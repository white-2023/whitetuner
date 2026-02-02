# -*- coding: utf-8 -*-

import os
import sys
import gc
import torch
import time
import argparse
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "whitetuner_diffusers"))


def print_vram(msg=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {msg}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def encode_prompt(text_encoder, tokenizer, prompt_text, device):
    if not prompt_text or prompt_text.strip() == "":
        prompt_text = " "
    
    qwen_input_ids, attention_mask = tokenizer.encode_qwen(prompt_text, device=device)
    
    with torch.no_grad():
        text_embeds = text_encoder(qwen_input_ids, attention_mask=attention_mask)
    
    t5_input_ids = tokenizer.encode_t5(prompt_text, device=device)
    
    return text_embeds, t5_input_ids


def run_sampling(dit, context_pos, context_neg, latent_shape, num_steps, guidance_scale, shift, device, dtype):
    latent_c, latent_t, latent_h, latent_w = latent_shape
    latents = torch.randn(1, latent_c, latent_t, latent_h, latent_w, device=device, dtype=torch.float32)
    
    sigmas = []
    for i in range(num_steps):
        t = (num_steps - i) / num_steps
        sigma = time_snr_shift(shift, t)
        sigmas.append(sigma)
    sigmas.append(0.0)
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float64)
    
    print(f"  Sigmas[0]={sigmas[0].item():.4f}, Sigmas[-1]={sigmas[-2].item():.4f}")
    
    with torch.no_grad():
        for i in tqdm(range(len(sigmas) - 1), desc="Sampling"):
            current_sigma = sigmas[i]
            next_sigma = sigmas[i + 1]
            
            timestep = (current_sigma * 1.0).expand(latents.shape[0]).to(dtype)
            latent_model_input = latents.to(dtype)
            
            with torch.autocast(device_type="cuda", dtype=dtype):
                noise_pred = dit(
                    latent_model_input,
                    timesteps=timestep,
                    context=context_pos,
                )
            
            if guidance_scale > 1.0:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    noise_pred_uncond = dit(
                        latent_model_input,
                        timesteps=timestep,
                        context=context_neg,
                    )
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            denoised = latents - noise_pred.float() * current_sigma
            d = (latents - denoised) / current_sigma
            latents = latents + d * (next_sigma - current_sigma)
    
    return latents


def process_latents_for_vae(latents, device, dtype):
    latents_mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ]).view(1, 16, 1, 1, 1).to(device, dtype)
    latents_std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ]).view(1, 16, 1, 1, 1).to(device, dtype)
    return latents * latents_std + latents_mean


def decode_latents(vae, latents):
    with torch.no_grad():
        images = vae.decode([latents], already_processed=True)
        image = images[0]
    
    while image.dim() > 3:
        image = image.squeeze(1) if image.shape[1] == 1 else image.squeeze(0)
    
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().float().numpy()
    image = (image * 255).astype("uint8")
    
    return Image.fromarray(image)


def run_inference(
    base_model_path: str,
    trained_checkpoint_path: str = None,
    prompt: str = "A beautiful anime girl",
    negative_prompt: str = "",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
    shift: float = 3.0,
    seed: int = 42,
    output_dir: str = None,
):
    print("=" * 60)
    print("Anima Inference")
    print("=" * 60)
    
    device = "cuda"
    dtype = torch.bfloat16
    
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    te_path = os.path.join(base_model_path, "split_files", "text_encoders", "qwen_3_06b_base.safetensors")
    dit_path = os.path.join(base_model_path, "split_files", "diffusion_models", "anima-preview.safetensors")
    vae_path = os.path.join(base_model_path, "split_files", "vae", "qwen_image_vae.safetensors")
    
    trained_te_path = None
    trained_dit_path = None
    if trained_checkpoint_path:
        trained_te_path = os.path.join(trained_checkpoint_path, "text_encoder.safetensors")
        trained_dit_path = os.path.join(trained_checkpoint_path, "dit.safetensors")
        if not os.path.exists(trained_te_path):
            trained_te_path = None
            print(f"[INFO] No trained text encoder found at {trained_te_path}")
        if not os.path.exists(trained_dit_path):
            trained_dit_path = None
            print(f"[INFO] No trained DiT found at {trained_dit_path}")
    
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_orig = os.path.join(output_dir, "anima_original.png")
    output_path_trained = os.path.join(output_dir, "anima_trained.png")
    
    print(f"\nPrompt: {prompt[:80]}...")
    print(f"Negative: '{negative_prompt[:50]}...' " if len(negative_prompt) > 50 else f"Negative: '{negative_prompt}'")
    print(f"Size: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Shift: {shift}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    latent_c = 16
    latent_t = 1
    latent_h = height // 8
    latent_w = width // 8
    latent_shape = (latent_c, latent_t, latent_h, latent_w)
    
    from anima_modules.text_encoder import load_qwen3_text_encoder, AnimaTokenizer
    from anima_modules.model import load_anima_model
    from anima_modules.vae import load_anima_vae
    
    print("\n[Stage 1] Loading Text Encoder...")
    text_encoder = load_qwen3_text_encoder(te_path, device=device, dtype=dtype)
    text_encoder.eval()
    
    tokenizer = AnimaTokenizer()
    print_vram("After TE loaded")
    
    print("\n[Stage 2] Encoding prompts...")
    text_embeds_pos, t5_ids_pos = encode_prompt(text_encoder, tokenizer, prompt, device)
    text_embeds_neg, t5_ids_neg = encode_prompt(text_encoder, tokenizer, negative_prompt, device)
    print(f"Positive embeds: {text_embeds_pos.shape}, T5 max: {t5_ids_pos.max().item()}")
    print(f"Negative embeds: {text_embeds_neg.shape}, T5 max: {t5_ids_neg.max().item()}")
    
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print("Text Encoder unloaded")
    
    print("\n[Stage 3] Loading original DiT...")
    dit = load_anima_model(dit_path, device=device, dtype=dtype)
    dit.eval()
    print_vram("After DiT loaded")
    
    orig_weight_sample = None
    for name, param in dit.named_parameters():
        if "blocks.0" in name and "weight" in name:
            orig_weight_sample = param.data.clone().cpu()
            print(f"Original DiT weight ({name[:40]}...): mean={param.data.float().mean().item():.6f}, std={param.data.float().std().item():.6f}")
            break
    
    t5_ids_pos = t5_ids_pos.unsqueeze(0) if t5_ids_pos.ndim == 1 else t5_ids_pos
    t5_ids_neg = t5_ids_neg.unsqueeze(0) if t5_ids_neg.ndim == 1 else t5_ids_neg
    
    with torch.no_grad():
        context_pos = dit.preprocess_text_embeds(text_embeds_pos, t5_ids_pos)
        context_neg = dit.preprocess_text_embeds(text_embeds_neg, t5_ids_neg)
    
    if context_pos.shape[1] < 512:
        context_pos = torch.nn.functional.pad(context_pos, (0, 0, 0, 512 - context_pos.shape[1]))
    if context_neg.shape[1] < 512:
        context_neg = torch.nn.functional.pad(context_neg, (0, 0, 0, 512 - context_neg.shape[1]))
    
    print(f"Context pos: {context_pos.shape}, Context neg: {context_neg.shape}")
    
    print("\n[Stage 4] Sampling with original model...")
    start_time = time.time()
    latents_orig = run_sampling(dit, context_pos, context_neg, latent_shape, num_inference_steps, guidance_scale, shift, device, dtype)
    print(f"Sampling done! Time: {time.time() - start_time:.2f}s")
    print(f"Latents: mean={latents_orig.mean().item():.4f}, std={latents_orig.std().item():.4f}")
    
    latents_orig = process_latents_for_vae(latents_orig, device, latents_orig.dtype)
    
    del dit, context_pos, context_neg
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n[Stage 5] Loading VAE and decoding original...")
    vae = load_anima_vae(vae_path, device=device, dtype=dtype)
    
    image_orig = decode_latents(vae, latents_orig)
    image_orig.save(output_path_orig)
    print(f"Original model output saved: {output_path_orig}")
    
    del latents_orig, vae
    gc.collect()
    torch.cuda.empty_cache()
    print_vram("After first decode")
    
    if trained_dit_path or trained_te_path:
        print("\n" + "=" * 60)
        print("Running inference with TRAINED model")
        print("=" * 60)
        
        if trained_te_path:
            print("\n[Stage 6] Loading TRAINED Text Encoder...")
            text_encoder = load_qwen3_text_encoder(trained_te_path, device=device, dtype=dtype)
            text_encoder.eval()
            
            print("Encoding prompts with trained TE...")
            text_embeds_pos, t5_ids_pos = encode_prompt(text_encoder, tokenizer, prompt, device)
            text_embeds_neg, t5_ids_neg = encode_prompt(text_encoder, tokenizer, negative_prompt, device)
            
            del text_encoder
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("\n[Stage 6] Re-using original TE embeddings (no trained TE found)")
            text_encoder = load_qwen3_text_encoder(te_path, device=device, dtype=dtype)
            text_encoder.eval()
            text_embeds_pos, t5_ids_pos = encode_prompt(text_encoder, tokenizer, prompt, device)
            text_embeds_neg, t5_ids_neg = encode_prompt(text_encoder, tokenizer, negative_prompt, device)
            del text_encoder
            gc.collect()
            torch.cuda.empty_cache()
        
        if trained_dit_path:
            print("\n[Stage 7] Loading TRAINED DiT...")
            dit = load_anima_model(trained_dit_path, device=device, dtype=dtype)
        else:
            print("\n[Stage 7] Re-loading original DiT (no trained DiT found)")
            dit = load_anima_model(dit_path, device=device, dtype=dtype)
        dit.eval()
        print_vram("After trained DiT loaded")
        
        for name, param in dit.named_parameters():
            if "blocks.0" in name and "weight" in name:
                trained_weight_sample = param.data.clone().cpu()
                print(f"Trained DiT weight ({name[:40]}...): mean={param.data.float().mean().item():.6f}, std={param.data.float().std().item():.6f}")
                if orig_weight_sample is not None:
                    diff = (orig_weight_sample.float() - trained_weight_sample.float()).abs().mean().item()
                    print(f"Weight difference (original vs trained): {diff:.6f}")
                    if diff < 1e-6:
                        print("[WARNING] Weights are almost identical! Training may not have changed the model.")
                break
        
        t5_ids_pos = t5_ids_pos.unsqueeze(0) if t5_ids_pos.ndim == 1 else t5_ids_pos
        t5_ids_neg = t5_ids_neg.unsqueeze(0) if t5_ids_neg.ndim == 1 else t5_ids_neg
        
        with torch.no_grad():
            context_pos = dit.preprocess_text_embeds(text_embeds_pos, t5_ids_pos)
            context_neg = dit.preprocess_text_embeds(text_embeds_neg, t5_ids_neg)
        
        if context_pos.shape[1] < 512:
            context_pos = torch.nn.functional.pad(context_pos, (0, 0, 0, 512 - context_pos.shape[1]))
        if context_neg.shape[1] < 512:
            context_neg = torch.nn.functional.pad(context_neg, (0, 0, 0, 512 - context_neg.shape[1]))
        
        print("\n[Stage 8] Sampling with trained model...")
        start_time = time.time()
        latents_trained = run_sampling(dit, context_pos, context_neg, latent_shape, num_inference_steps, guidance_scale, shift, device, dtype)
        print(f"Sampling done! Time: {time.time() - start_time:.2f}s")
        print(f"Latents: mean={latents_trained.mean().item():.4f}, std={latents_trained.std().item():.4f}")
        
        latents_trained = process_latents_for_vae(latents_trained, device, latents_trained.dtype)
        
        del dit, context_pos, context_neg
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\n[Stage 9] Loading VAE and decoding trained...")
        vae = load_anima_vae(vae_path, device=device, dtype=dtype)
        
        image_trained = decode_latents(vae, latents_trained)
        image_trained.save(output_path_trained)
        print(f"Trained model output saved: {output_path_trained}")
        
        del latents_trained, vae
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Original: {output_path_orig}")
    if trained_dit_path or trained_te_path:
        print(f"Trained:  {output_path_trained}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Anima Inference")
    parser.add_argument("--base_model_path", type=str, default=r"F:\models\circlestone-labs-Anima",
                        help="Base model path")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Trained checkpoint path (e.g., output/checkpoints/checkpoint-1000)")
    parser.add_argument("--prompt", type=str, 
                        default="masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top and denim shorts is standing outdoors. She's holding a rectangular sign out in front of her that reads \"ANIMA\". She's looking at the viewer with a smile. The background features some trees and blue sky with clouds.",
                        help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale (CFG)")
    parser.add_argument("--shift", type=float, default=3.0, help="Time shift for flow matching")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    run_inference(
        base_model_path=args.base_model_path,
        trained_checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
