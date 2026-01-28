# -*- coding: utf-8 -*-
"""
Flux2 Klein Text-to-Image Inference
从训练代码导入模块进行推理
"""

import os
import sys
import gc
import json
import torch
import time
from PIL import Image
from typing import Dict, Tuple, Optional
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
whitetuner_dir = os.path.dirname(script_dir)
diffusers_dir = os.path.join(whitetuner_dir, "whitetuner_diffusers")
if diffusers_dir not in sys.path:
    sys.path.insert(0, diffusers_dir)

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2
from flux2_modules import Flux2Transformer2DModel, load_flux2_transformer_from_diffusers
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from optimum.quanto import freeze, qint8, quantize

# ===== 配置 =====
base_model_path = r"/root/FLUX.2-klein-base-9B"
trained_transformer_path = r"/root/whitetuner/whitetuner_diffusers/output/final"  # 留空则只生成基础模型图像

output_path = os.path.join(script_dir, "flux2_klein.png")

prompt = "A cat holding a sign that says hello world"
negative_prompt = ""  # Flux2 Klein 不使用 negative prompt

height = 1024
width = 1024
seed = 42

num_inference_steps = 20
guidance_scale = 4.0  # CFG scale（和官方测试一样）

quantize_text_encoder = True
quantize_transformer = False

text_encoder_layers = (9, 18, 27)  # Qwen3 使用的层
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


def run_inference(
    transformer,
    scheduler,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    height: int,
    width: int,
    num_steps: int,
    seed: int,
    device,
    dtype,
    latents_bn_mean: torch.Tensor,
    latents_bn_std: torch.Tensor,
    guidance_scale: float = 3.5,
) -> torch.Tensor:
    """运行推理循环"""
    vae_scale_factor = 8
    do_classifier_free_guidance = guidance_scale > 1.0
    
    in_channels = transformer.config.get("in_channels", 128)
    
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))
    patched_height = latent_height // 2
    patched_width = latent_width // 2
    
    generator = torch.Generator(device=device).manual_seed(seed)
    latents_4d = torch.randn(
        (1, in_channels, patched_height, patched_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    
    latent_ids = prepare_latent_ids(latents_4d).to(device)
    latents = pack_latents(latents_4d)
    
    txt_ids = prepare_text_ids(prompt_embeds).to(device)
    negative_txt_ids = prepare_text_ids(negative_prompt_embeds).to(device) if do_classifier_free_guidance else None
    
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    
    scheduler.set_timesteps(num_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps
    scheduler.set_begin_index(0)
    
    print(f"[DEBUG] do_classifier_free_guidance: {do_classifier_free_guidance}")
    print(f"[DEBUG] Initial latents: shape={latents.shape}, dtype={latents.dtype}")
    
    # 获取 transformer 的 dtype（通过第一个参数的 dtype）
    transformer_dtype = next(transformer.parameters()).dtype
    
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        
        # 官方实现：latent_model_input = latents.to(self.transformer.dtype)
        latent_model_input = latents.to(transformer_dtype)
        
        with torch.no_grad():
            # 注意：transformer.forward 内部已经做了 hidden_states[:, num_txt_tokens:, ...]
            # 所以输出只包含 image tokens，不需要在这里再截断
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            
            # Debug: 检查 noise_pred 形状和值
            if i == 0:
                print(f"[DEBUG step 0] noise_pred shape: {noise_pred.shape}, latents shape: {latents.shape}")
                print(f"[DEBUG step 0] noise_pred: min={noise_pred.min():.4f}, max={noise_pred.max():.4f}, mean={noise_pred.mean():.4f}")
            
            if do_classifier_free_guidance:
                neg_noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_txt_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
                
                if i == 0:
                    print(f"[DEBUG step 0] neg_noise_pred: min={neg_noise_pred.min():.4f}, max={neg_noise_pred.max():.4f}")
                
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                
                if i == 0:
                    print(f"[DEBUG step 0] after CFG: min={noise_pred.min():.4f}, max={noise_pred.max():.4f}")
        
        latents_dtype = latents.dtype
        latents_before = latents.clone()
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
        
        # Debug: 检查 scheduler step 的效果
        if i == 0:
            print(f"[DEBUG step 0] latents before: min={latents_before.min():.4f}, max={latents_before.max():.4f}")
            print(f"[DEBUG step 0] latents after: min={latents.min():.4f}, max={latents.max():.4f}")
            print(f"[DEBUG step 0] timestep: {t.item():.4f}")
    
    print(f"[DEBUG] Final latents (packed): shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}")
    
    # Unpack latents 回 4D 格式
    latents = unpack_latents_with_ids(latents, latent_ids)
    print(f"[DEBUG] After unpack: shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}")
    
    # 反向 BatchNorm 归一化
    bn_mean = latents_bn_mean.to(device, dtype)
    bn_std = latents_bn_std.to(device, dtype)
    print(f"[DEBUG] bn_mean: shape={bn_mean.shape}, values={bn_mean.flatten()[:5]}")
    print(f"[DEBUG] bn_std: shape={bn_std.shape}, values={bn_std.flatten()[:5]}")
    latents = latents * bn_std + bn_mean
    print(f"[DEBUG] After BN denorm: shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}")
    
    # Unpatchify: (B, in_channels, H/2, W/2) -> (B, in_channels/4, H, W)
    latents = unpatchify_latents(latents)
    print(f"[DEBUG] After unpatchify: shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}")
    
    return latents


def decode_latents(vae, latents: torch.Tensor, dtype) -> Image.Image:
    """解码 latent 为图像"""
    print(f"[DEBUG decode] Input latents: shape={latents.shape}, dtype={latents.dtype}, min={latents.min():.4f}, max={latents.max():.4f}")
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    
    print(f"[DEBUG decode] After VAE decode: shape={image.shape}, min={image.min():.4f}, max={image.max():.4f}")
    
    image = (image / 2 + 0.5).clamp(0, 1)
    print(f"[DEBUG decode] After normalize: min={image.min():.4f}, max={image.max():.4f}")
    
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")[0]
    
    return Image.fromarray(image)


def main():
    name, ext = os.path.splitext(output_path)
    output_path_base = f"{name}_base{ext}"
    output_path_trained = f"{name}_trained{ext}"
    
    print("=" * 60)
    print("Flux2 Klein Text-to-Image Inference")
    print("=" * 60)
    print(f"Prompt: {prompt[:80]}...")
    print(f"Size: {width}x{height}, Steps: {num_inference_steps}, Seed: {seed}")
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
    
    # Stage 2: 编码提示词 (正面和负面)
    print("\n[Stage 2] Encoding prompts...")
    prompt_data = encode_prompt(
        text_encoder, tokenizer, prompt, device, torch_dtype,
        text_encoder_layers=text_encoder_layers,
        max_length=max_sequence_length,
    )
    prompt_embeds = prompt_data['prompt_embeds']
    print(f"Prompt embeds shape: {prompt_embeds.shape}")
    
    negative_prompt_data = encode_prompt(
        text_encoder, tokenizer, "", device, torch_dtype,
        text_encoder_layers=text_encoder_layers,
        max_length=max_sequence_length,
    )
    negative_prompt_embeds = negative_prompt_data['prompt_embeds']
    print(f"Negative prompt embeds shape: {negative_prompt_embeds.shape}")
    
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
    
    # 获取 BatchNorm 参数
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    
    vae.to(device)
    print_vram("After VAE loaded")
    
    # Stage 5: 加载原始 Transformer
    print("\n[Stage 5] Loading original Transformer...")
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
    
    # Stage 6: 创建 Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
    )
    
    # Stage 7: 基础模型推理
    print("\n[Stage 7] Running inference with original model...")
    start_time = time.time()
    latents_base = run_inference(
        transformer, scheduler, prompt_embeds, negative_prompt_embeds,
        height, width, num_inference_steps, seed,
        device, torch_dtype,
        latents_bn_mean, latents_bn_std,
        guidance_scale,
    )
    print(f"Inference done! Time: {time.time() - start_time:.2f}s")
    
    # Stage 8: 解码并保存基础模型图像
    print("\n[Stage 8] Decoding base model image...")
    image_base = decode_latents(vae, latents_base, torch_dtype)
    image_base.save(output_path_base)
    print(f"Base model output saved: {output_path_base}")
    
    del latents_base
    gc.collect()
    torch.cuda.empty_cache()
    
    # Stage 9: 训练后模型推理（如果提供了路径）
    if trained_transformer_path and os.path.exists(trained_transformer_path):
        print("\n" + "-" * 60)
        print("[Stage 9] Loading trained Transformer...")
        
        # 卸载原始 transformer
        del transformer
        gc.collect()
        torch.cuda.empty_cache()
        
        # 加载训练后的 transformer
        transformer_trained = load_flux2_transformer_from_diffusers(
            trained_transformer_path,
            torch_dtype=torch_dtype,
            device=device,
        )
        transformer_trained.requires_grad_(False)
        transformer_trained.eval()
        
        if quantize_transformer:
            print("Quantizing trained Transformer (qint8)...")
            quantize(transformer_trained, weights=qint8, exclude=["*norm*", "proj_out*", "*embedder*"])
            freeze(transformer_trained)
        
        print_vram("After trained Transformer loaded")
        
        # 重置 scheduler
        scheduler_trained = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_path,
            subfolder="scheduler",
        )
        
        print("\n[Stage 10] Running inference with trained model...")
        start_time = time.time()
        latents_trained = run_inference(
            transformer_trained, scheduler_trained, prompt_embeds, negative_prompt_embeds,
            height, width, num_inference_steps, seed,
            device, torch_dtype,
            latents_bn_mean, latents_bn_std,
            guidance_scale,
        )
        print(f"Inference done! Time: {time.time() - start_time:.2f}s")
        
        print("\n[Stage 11] Decoding trained model image...")
        image_trained = decode_latents(vae, latents_trained, torch_dtype)
        image_trained.save(output_path_trained)
        print(f"Trained model output saved: {output_path_trained}")
        
        del transformer_trained, latents_trained
    else:
        if trained_transformer_path:
            print(f"\n[WARNING] Trained model path not found: {trained_transformer_path}")
        print("Skipping trained model inference.")
    
    # 清理
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Base model: {output_path_base}")
    if trained_transformer_path and os.path.exists(trained_transformer_path):
        print(f"Trained:    {output_path_trained}")


if __name__ == "__main__":
    main()

