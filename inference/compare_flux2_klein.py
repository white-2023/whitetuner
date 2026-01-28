# -*- coding: utf-8 -*-
"""
对比测试：官方 diffusers pipeline vs 我们的推理代码
"""

import os
import sys

# 优先使用本地 diffusers (D:\ai\diffusers)
local_diffusers_path = r"D:\ai\diffusers\src"
if local_diffusers_path not in sys.path:
    sys.path.insert(0, local_diffusers_path)

import torch
import gc

# 配置
model_path = r"F:\models\FLUX.2-klein-base-9B"
prompt = "A cat holding a sign that says hello world"
height = 1024
width = 1024
seed = 42
num_inference_steps = 20
guidance_scale = 4.0

device = "cuda"
dtype = torch.bfloat16

script_dir = os.path.dirname(os.path.abspath(__file__))

def test_official_pipeline():
    """测试官方 diffusers pipeline"""
    print("\n" + "=" * 70)
    print("TESTING OFFICIAL DIFFUSERS PIPELINE")
    print("=" * 70)
    
    from diffusers import Flux2KleinPipeline
    
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    
    output_path = os.path.join(script_dir, "compare_official.png")
    image.save(output_path)
    print(f"\n[OFFICIAL] Image saved to: {output_path}")
    
    # 清理
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def test_our_inference():
    """测试我们的推理代码"""
    print("\n" + "=" * 70)
    print("TESTING OUR INFERENCE CODE")
    print("=" * 70)
    
    # 添加路径
    whitetuner_dir = os.path.dirname(script_dir)
    diffusers_dir = os.path.join(whitetuner_dir, "whitetuner_diffusers")
    if diffusers_dir not in sys.path:
        sys.path.insert(0, diffusers_dir)
    
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLFlux2
    from flux2_modules import load_flux2_transformer_from_diffusers
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
    from optimum.quanto import freeze, qint8, quantize
    from PIL import Image
    from tqdm import tqdm
    
    # 导入我们的辅助函数
    from flux2_klein_inference import (
        encode_prompt, compute_empirical_mu, prepare_latent_ids, prepare_text_ids,
        pack_latents, unpack_latents_with_ids, unpatchify_latents
    )
    
    text_encoder_layers = (9, 18, 27)
    max_sequence_length = 512
    
    # 加载 Text Encoder
    print("\n[OUR] Loading Text Encoder...")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    quantize(text_encoder, weights=qint8, exclude=["*embed*", "*lm_head*"])
    freeze(text_encoder)
    text_encoder.to(device)
    
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
    
    # 编码 prompt
    print("[OUR] Encoding prompt...")
    result = encode_prompt(text_encoder, tokenizer, prompt, device, dtype, text_encoder_layers, max_sequence_length)
    prompt_embeds = result['prompt_embeds']
    negative_result = encode_prompt(text_encoder, tokenizer, "", device, dtype, text_encoder_layers, max_sequence_length)
    negative_prompt_embeds = negative_result['prompt_embeds']
    
    print(f"[OUR DEBUG] prompt_embeds: shape={prompt_embeds.shape}")
    print(f"[OUR DEBUG] prompt_embeds stats: min={prompt_embeds.min():.4f}, max={prompt_embeds.max():.4f}, mean={prompt_embeds.mean():.4f}")
    print(f"[OUR DEBUG] negative_prompt_embeds stats: min={negative_prompt_embeds.min():.4f}, max={negative_prompt_embeds.max():.4f}, mean={negative_prompt_embeds.mean():.4f}")
    
    diff = (prompt_embeds - negative_prompt_embeds).abs()
    print(f"[OUR DEBUG] embeds diff stats: min={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}")
    
    # 卸载 text encoder
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    
    # 加载 VAE
    print("[OUR] Loading VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)
    
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    
    # 加载 Transformer
    print("[OUR] Loading Transformer...")
    transformer = load_flux2_transformer_from_diffusers(model_path, subfolder="transformer", torch_dtype=dtype, device=device)
    transformer.requires_grad_(False)
    transformer.eval()
    
    # 创建 Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    # 准备 latents
    vae_scale_factor = 8
    in_channels = transformer.config.get("in_channels", 128)
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))
    patched_height = latent_height // 2
    patched_width = latent_width // 2
    
    generator = torch.Generator(device=device).manual_seed(seed)
    latents_4d = torch.randn(
        (1, in_channels, patched_height, patched_width),
        generator=generator, device=device, dtype=dtype,
    )
    
    latent_ids = prepare_latent_ids(latents_4d).to(device)
    latents = pack_latents(latents_4d)
    
    txt_ids = prepare_text_ids(prompt_embeds).to(device)
    negative_txt_ids = prepare_text_ids(negative_prompt_embeds).to(device)
    
    print(f"[OUR DEBUG] Initial latents: shape={latents.shape}, dtype={latents.dtype}")
    print(f"[OUR DEBUG] text_ids: shape={txt_ids.shape}, sample={txt_ids[0, :3]}")
    print(f"[OUR DEBUG] latent_ids: shape={latent_ids.shape}, sample={latent_ids[0, :3]}")
    
    # 设置 timesteps
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps
    scheduler.set_begin_index(0)
    
    transformer_dtype = next(transformer.parameters()).dtype
    
    # 推理循环
    print("\n[OUR] Running inference...")
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        latent_model_input = latents.to(transformer_dtype)
        
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            
            if i == 0:
                print(f"[OUR DEBUG step 0] noise_pred: shape={noise_pred.shape}, min={noise_pred.min():.4f}, max={noise_pred.max():.4f}, mean={noise_pred.mean():.4f}")
            
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
                print(f"[OUR DEBUG step 0] neg_noise_pred: min={neg_noise_pred.min():.4f}, max={neg_noise_pred.max():.4f}")
                pred_diff = (noise_pred - neg_noise_pred).abs()
                print(f"[OUR DEBUG step 0] pos-neg diff: min={pred_diff.min():.4f}, max={pred_diff.max():.4f}, mean={pred_diff.mean():.4f}")
            
            noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
            
            if i == 0:
                print(f"[OUR DEBUG step 0] after CFG: min={noise_pred.min():.4f}, max={noise_pred.max():.4f}")
                print(f"[OUR DEBUG step 0] latents before step: min={latents.min():.4f}, max={latents.max():.4f}")
                print(f"[OUR DEBUG step 0] timestep: {t.item():.4f}")
        
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        if i == 0:
            print(f"[OUR DEBUG step 0] latents after step: min={latents.min():.4f}, max={latents.max():.4f}")
    
    # 后处理
    print(f"\n[OUR DEBUG] Final latents (packed): shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}")
    
    latents = unpack_latents_with_ids(latents, latent_ids)
    print(f"[OUR DEBUG] After unpack: shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}")
    
    bn_mean = latents_bn_mean.to(device, dtype)
    bn_std = latents_bn_std.to(device, dtype)
    print(f"[OUR DEBUG] bn_mean sample: {bn_mean.flatten()[:5]}")
    print(f"[OUR DEBUG] bn_std sample: {bn_std.flatten()[:5]}")
    
    latents = latents * bn_std + bn_mean
    print(f"[OUR DEBUG] After BN denorm: shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}")
    
    latents = unpatchify_latents(latents)
    print(f"[OUR DEBUG] After unpatchify: shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}")
    
    # VAE decode
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    print(f"[OUR DEBUG] After VAE decode: shape={image.shape}, min={image.min():.4f}, max={image.max():.4f}")
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")[0]
    image = Image.fromarray(image)
    
    output_path = os.path.join(script_dir, "compare_ours.png")
    image.save(output_path)
    print(f"\n[OUR] Image saved to: {output_path}")
    
    # 清理
    del transformer, vae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Running comparison test...")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt}")
    print(f"Size: {width}x{height}, Steps: {num_inference_steps}, Seed: {seed}")
    print(f"Guidance Scale: {guidance_scale}")
    
    # 先测试官方 pipeline
    test_official_pipeline()
    
    # 再测试我们的代码
    test_our_inference()
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Official: {os.path.join(script_dir, 'compare_official.png')}")
    print(f"Ours:     {os.path.join(script_dir, 'compare_ours.png')}")

