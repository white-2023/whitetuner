# -*- coding: utf-8 -*-
"""
Flux2 Klein Edit Inference
图像编辑推理 - 从训练代码导入模块
"""

import os
import sys
import gc
import json
import torch
import time
from PIL import Image
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
hhytuner_dir = os.path.dirname(script_dir)
diffusers_dir = os.path.join(hhytuner_dir, "hhytuner_diffusers")
if diffusers_dir not in sys.path:
    sys.path.insert(0, diffusers_dir)

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2
from flux2_modules import Flux2Transformer2DModel, load_flux2_transformer_from_diffusers
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from optimum.quanto import freeze, qint8, quantize
from torchvision import transforms

# ===== 配置 =====
base_model_path = r"F:/models/FLUX.2-klein-base-9B"
trained_transformer_path = r""  # 训练后模型路径，留空则只使用基础模型

output_path = os.path.join(script_dir, "flux2_klein_edit.png")

# 输入图像（condition 图像列表）
condition_images = [
    r"",  # condition1 路径
    r"",  # condition2 路径 (可选)
]

# 编辑提示词
prompt = "Replace the background with a sunset over the ocean."
seed = 42

num_inference_steps = 50
guidance_scale = 3.5

quantize_text_encoder = True
quantize_transformer = False

text_encoder_layers = (9, 18, 27)
max_sequence_length = 512

torch_dtype = torch.bfloat16
device = "cuda"


def print_vram(msg=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {msg}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """计算 dynamic shifting 的 mu 参数"""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190
    b = m_10 - a * 10

    mu = a * num_steps + b
    return float(mu)


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """将 latent 转换为 patch 格式"""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
    return latents


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """打包 latent 为序列格式"""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    """根据位置 ID 解包 latent"""
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1
        flat_ids = h_ids * w + w_ids
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list, dim=0)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """将 patch 格式转换回原始 latent 格式"""
    batch_size, num_channels, height, width = latents.shape
    # num_channels = in_channels (patchified), height/width = 原始的 1/2
    # 需要还原为 num_channels/4 通道
    out_channels = num_channels // 4
    latents = latents.view(batch_size, out_channels, 4, height, width)
    latents = latents.view(batch_size, out_channels, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)  # b, c, h, 2, w, 2
    latents = latents.reshape(batch_size, out_channels, height * 2, width * 2)
    return latents


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """准备 latent 位置 ID
    
    官方实现: (B, C, H, W) -> (B, H*W, 4)
    坐标格式: (t, h, w, l) where t=0, l=0, h=[0..H-1], w=[0..W-1]
    """
    batch_size, _, height, width = latents.shape
    
    t = torch.arange(1)  # [0]
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)  # [0]
    
    # 使用与官方相同的 cartesian_prod
    latent_ids = torch.cartesian_prod(t, h, w, l)  # (H*W, 4)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H*W, 4)
    
    return latent_ids


def prepare_image_ids(image_latents: List[torch.Tensor], scale: int = 10) -> torch.Tensor:
    """为多个 condition 图像生成 4D 位置坐标
    
    官方实现: 每个 condition 图像有不同的 t 坐标 (scale + scale * idx)
    """
    # 创建每个 condition 图像的 t 坐标
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]
    
    image_latent_ids = []
    for latent, t_coord in zip(image_latents, t_coords):
        latent = latent.squeeze(0)  # (C, H, W)
        _, height, width = latent.shape
        
        # 使用与官方相同的 cartesian_prod
        x_ids = torch.cartesian_prod(t_coord, torch.arange(height), torch.arange(width), torch.arange(1))
        image_latent_ids.append(x_ids)
    
    image_latent_ids = torch.cat(image_latent_ids, dim=0)  # (total_tokens, 4)
    return image_latent_ids.unsqueeze(0)  # (1, total_tokens, 4)


def prepare_text_ids(encoder_hidden_states: torch.Tensor) -> torch.Tensor:
    """准备 text 位置 ID
    
    官方实现使用 cartesian_prod 创建 (t, h, w, l) 坐标：
    - t = 0 (时间维度，固定为0)
    - h = 0 (高度，固定为0，因为是文本)
    - w = 0 (宽度，固定为0，因为是文本)
    - l = 0, 1, 2, ..., L-1 (序列位置)
    """
    batch_size, seq_len, _ = encoder_hidden_states.shape
    
    out_ids = []
    for _ in range(batch_size):
        t = torch.arange(1)  # [0]
        h = torch.arange(1)  # [0]
        w = torch.arange(1)  # [0]
        l = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
        coords = torch.cartesian_prod(t, h, w, l)  # (seq_len, 4)
        out_ids.append(coords)
    
    return torch.stack(out_ids)  # (batch_size, seq_len, 4)


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int]) -> torch.Tensor:
    """加载并预处理图像"""
    image = Image.open(image_path).convert("RGB")
    
    # 调整大小
    image = image.resize(target_size, Image.LANCZOS)
    
    # 转换为 tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    return transform(image).unsqueeze(0)


def encode_prompt(
    text_encoder,
    tokenizer,
    prompt: str,
    device,
    dtype,
    text_encoder_layers: Tuple[int, ...] = (9, 18, 27),
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """编码文本提示词"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    
    hidden_states = torch.stack(
        [output.hidden_states[k] for k in text_encoder_layers],
        dim=1
    )
    hidden_states = hidden_states.to(dtype=dtype, device=device)
    
    batch_size, num_layers, seq_len, hidden_dim = hidden_states.shape
    prompt_embeds = hidden_states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
    
    return {
        'prompt_embeds': prompt_embeds,
        'attention_mask': attention_mask,
    }


def encode_image(
    vae,
    image_tensor: torch.Tensor,
    latents_bn_mean: torch.Tensor,
    latents_bn_std: torch.Tensor,
    device,
    dtype,
) -> torch.Tensor:
    """编码图像为 latent"""
    image_tensor = image_tensor.to(device, dtype)
    
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.mode()
    
    latent = patchify_latents(latent)
    
    bn_mean = latents_bn_mean.to(device, dtype)
    bn_std = latents_bn_std.to(device, dtype)
    latent = (latent - bn_mean) / bn_std
    
    return latent


def run_edit_inference(
    transformer,
    scheduler,
    prompt_embeds: torch.Tensor,
    condition_latents: List[torch.Tensor],
    output_height: int,
    output_width: int,
    num_steps: int,
    seed: int,
    device,
    dtype,
    latents_bn_mean: torch.Tensor,
    latents_bn_std: torch.Tensor,
    guidance_scale: float = 3.5,
) -> torch.Tensor:
    """运行 Edit 推理循环"""
    vae_scale_factor = 8
    
    # 从 transformer config 获取通道数
    in_channels = transformer.config.get("in_channels", 128)
    
    # 计算 latent 尺寸 (需要能被 2 整除用于 packing)
    latent_height = 2 * (output_height // (vae_scale_factor * 2))
    latent_width = 2 * (output_width // (vae_scale_factor * 2))
    patched_height = latent_height // 2
    patched_width = latent_width // 2
    
    # 生成初始噪声 (patchified 格式: in_channels = num_latent_channels * 4)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents_4d = torch.randn(
        (1, in_channels, patched_height, patched_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    
    # 准备 target latent IDs (在 pack 之前)
    target_latent_ids = prepare_latent_ids(latents_4d).to(device)
    
    # Pack target latents 为序列格式
    latents = pack_latents(latents_4d)
    
    # 准备 condition latent IDs
    condition_image_ids = prepare_image_ids(condition_latents, scale=10).to(device)
    
    # 合并 IDs
    img_ids = torch.cat([target_latent_ids, condition_image_ids], dim=1)
    
    # Pack condition latents
    packed_condition_latents = [pack_latents(lat) for lat in condition_latents]
    
    # Text IDs
    txt_ids = prepare_text_ids(prompt_embeds).to(device)
    
    # 计算 mu (用于 dynamic shifting)
    image_seq_len = latents.shape[1]  # packed latents 的序列长度
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    
    # 设置 timesteps
    scheduler.set_timesteps(num_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps
    scheduler.set_begin_index(0)
    
    # 获取 transformer 的 dtype
    transformer_dtype = next(transformer.parameters()).dtype
    
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        # Concatenate target latents with condition latents
        all_packed = [latents] + packed_condition_latents
        hidden_states = torch.cat(all_packed, dim=1)
        
        # timestep 归一化
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        
        # 官方实现: hidden_states.to(self.transformer.dtype)
        latent_model_input = hidden_states.to(transformer_dtype)
        
        with torch.no_grad():
            # Klein Edit 模型使用 guidance=None (不是 guidance-distilled)
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
        
        # 只取 target 部分的输出
        target_seq_len = latents.shape[1]
        noise_pred = noise_pred[:, :target_seq_len]
        
        # scheduler.step 在 packed 格式上操作
        latents_dtype = latents.dtype
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
    
    # Unpack latents 回 4D 格式
    latents = unpack_latents_with_ids(latents, target_latent_ids)
    
    # 反向 BatchNorm 归一化
    bn_mean = latents_bn_mean.to(device, dtype)
    bn_std = latents_bn_std.to(device, dtype)
    latents = latents * bn_std + bn_mean
    
    # Unpatchify: (B, 64, H/2, W/2) -> (B, 16, H, W)
    latents = unpatchify_latents(latents)
    
    return latents


def decode_latents(vae, latents: torch.Tensor, dtype) -> Image.Image:
    """解码 latent 为图像"""
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")[0]
    
    return Image.fromarray(image)


def main():
    # 检查 condition 图像
    valid_condition_images = [p for p in condition_images if p and os.path.exists(p)]
    if not valid_condition_images:
        print("[ERROR] 请提供至少一张 condition 图像!")
        print("请修改 condition_images 列表中的路径。")
        return
    
    # 获取第一张图像的尺寸作为输出尺寸
    first_image = Image.open(valid_condition_images[0])
    output_width, output_height = first_image.size
    # 确保尺寸是 16 的倍数
    output_width = (output_width // 16) * 16
    output_height = (output_height // 16) * 16
    first_image.close()
    
    print("=" * 60)
    print("Flux2 Klein Edit Inference")
    print("=" * 60)
    print(f"Prompt: {prompt[:80]}...")
    print(f"Output Size: {output_width}x{output_height}")
    print(f"Condition Images: {len(valid_condition_images)}")
    for i, p in enumerate(valid_condition_images):
        print(f"  [{i+1}] {os.path.basename(p)}")
    print(f"Steps: {num_inference_steps}, Seed: {seed}")
    print(f"Guidance Scale: {guidance_scale}")
    print("-" * 60)
    
    # Stage 1: 加载 Text Encoder
    print("\n[Stage 1] Loading Text Encoder (Qwen3)...")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        base_model_path,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    if quantize_text_encoder:
        print("Quantizing Text Encoder (qint8)...")
        quantize(text_encoder, weights=qint8, exclude=["*embed*", "*lm_head*"])
        freeze(text_encoder)
    
    text_encoder.to(device)
    
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        base_model_path,
        subfolder="tokenizer",
    )
    print_vram("After TE loaded")
    
    # Stage 2: 编码提示词
    print("\n[Stage 2] Encoding prompt...")
    prompt_data = encode_prompt(
        text_encoder, tokenizer, prompt, device, torch_dtype,
        text_encoder_layers=text_encoder_layers,
        max_length=max_sequence_length,
    )
    prompt_embeds = prompt_data['prompt_embeds']
    print(f"Prompt embeds shape: {prompt_embeds.shape}")
    
    # Stage 3: 卸载 Text Encoder
    print("\n[Stage 3] Unloading Text Encoder...")
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print_vram("After TE unloaded")
    
    # Stage 4: 加载 VAE
    print("\n[Stage 4] Loading VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(
        base_model_path,
        subfolder="vae",
        torch_dtype=torch_dtype,
    )
    vae.requires_grad_(False)
    vae.eval()
    
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    
    vae.to(device)
    print_vram("After VAE loaded")
    
    # Stage 5: 编码 condition 图像
    print("\n[Stage 5] Encoding condition images...")
    condition_latents = []
    for i, img_path in enumerate(valid_condition_images):
        print(f"  Encoding [{i+1}/{len(valid_condition_images)}]: {os.path.basename(img_path)}")
        img_tensor = load_and_preprocess_image(img_path, (output_width, output_height))
        latent = encode_image(vae, img_tensor, latents_bn_mean, latents_bn_std, device, torch_dtype)
        condition_latents.append(latent)
        print(f"    Latent shape: {latent.shape}")
    
    # Stage 6: 加载 Transformer
    transformer_path = trained_transformer_path if trained_transformer_path and os.path.exists(trained_transformer_path) else base_model_path
    is_trained = transformer_path != base_model_path
    
    print(f"\n[Stage 6] Loading {'trained' if is_trained else 'base'} Transformer...")
    if is_trained:
        transformer = load_flux2_transformer_from_diffusers(
            trained_transformer_path,
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        transformer = load_flux2_transformer_from_diffusers(
            base_model_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            device=device,
        )
    transformer.requires_grad_(False)
    transformer.eval()
    
    if quantize_transformer:
        print("Quantizing Transformer (qint8)...")
        quantize(transformer, weights=qint8, exclude=["*norm*", "proj_out*", "*embedder*"])
        freeze(transformer)
    
    print_vram("After Transformer loaded")
    
    # Stage 7: 创建 Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
    )
    
    # Stage 8: 运行推理
    print(f"\n[Stage 8] Running {'trained' if is_trained else 'base'} model inference...")
    start_time = time.time()
    output_latents = run_edit_inference(
        transformer, scheduler, prompt_embeds,
        condition_latents,
        output_height, output_width,
        num_inference_steps, seed,
        device, torch_dtype,
        latents_bn_mean, latents_bn_std,
        guidance_scale,
    )
    print(f"Inference done! Time: {time.time() - start_time:.2f}s")
    
    # Stage 9: 解码并保存
    print("\n[Stage 9] Decoding and saving...")
    output_image = decode_latents(vae, output_latents, torch_dtype)
    output_image.save(output_path)
    print(f"Output saved: {output_path}")
    
    # 清理
    del transformer, vae, output_latents
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

