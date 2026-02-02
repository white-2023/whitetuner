import os
import sys
import gc
import torch
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_inference():
    print("=" * 60)
    print("Anima Inference (Cosmos2/diffusers style)")
    print("=" * 60)
    
    device = "cuda"
    dtype = torch.bfloat16
    
    model_path = r"F:\models\circlestone-labs-Anima"
    te_path = os.path.join(model_path, "split_files", "text_encoders", "qwen_3_06b_base.safetensors")
    dit_path = os.path.join(model_path, "split_files", "diffusion_models", "anima-preview.safetensors")
    vae_path = os.path.join(model_path, "split_files", "vae", "qwen_image_vae.safetensors")
    
    prompt = "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top and denim shorts is standing outdoors. She's holding a rectangular sign out in front of her that reads \"ANIMA\". She's looking at the viewer with a smile. The background features some trees and blue sky with clouds."
    negative_prompt = ""
    guidance_scale = 7.0
    
    print(f"\nPrompt: {prompt[:100]}...")
    print(f"Negative: '{negative_prompt}'")
    print(f"Guidance Scale: {guidance_scale}")
    
    height, width = 1024, 1024
    num_inference_steps = 20
    
    shift = 3.0
    
    latent_c = 16
    latent_t = 1
    latent_h = height // 8
    latent_w = width // 8
    
    print("\n[1/4] Loading Text Encoder and encoding text...")
    from anima_modules.text_encoder import load_qwen3_text_encoder, AnimaTokenizer
    text_encoder = load_qwen3_text_encoder(te_path, device=device, dtype=dtype)
    text_encoder.eval()
    
    tokenizer = AnimaTokenizer()
    
    def encode_prompt(prompt_text):
        if not prompt_text or prompt_text.strip() == "":
            prompt_text = " "
        
        qwen_input_ids, attention_mask = tokenizer.encode_qwen(prompt_text, device=device)
        
        print(f"  Qwen token ids (first 20): {qwen_input_ids[0][:20].tolist()}")
        print(f"  Qwen token ids length: {qwen_input_ids.shape[1]}")
        
        with torch.no_grad():
            text_embeds = text_encoder(qwen_input_ids, attention_mask=attention_mask)
        
        print(f"  Text encoder output: shape={text_embeds.shape}, mean={text_embeds.float().mean().item():.4f}, std={text_embeds.float().std().item():.4f}")
        
        t5_input_ids = tokenizer.encode_t5(prompt_text, device=device)
        
        return text_embeds, t5_input_ids
    
    text_embeds_pos, t5_input_ids_pos = encode_prompt(prompt)
    text_embeds_neg, t5_input_ids_neg = encode_prompt(negative_prompt)
    print(f"Positive text embeds: {text_embeds_pos.shape}, T5 ids max: {t5_input_ids_pos.max().item()}")
    print(f"Negative text embeds: {text_embeds_neg.shape}, T5 ids max: {t5_input_ids_neg.max().item()}")
    
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Text Encoder unloaded")
    
    print("\n[2/4] Loading DiT and sampling...")
    from anima_modules.model import load_anima_model
    dit = load_anima_model(dit_path, device=device, dtype=dtype)
    dit.eval()
    
    print("\nProcessing text through LLM Adapter...")
    print(f"  text_embeds_pos: shape={text_embeds_pos.shape}, mean={text_embeds_pos.float().mean().item():.4f}, std={text_embeds_pos.float().std().item():.4f}")
    print(f"  t5_input_ids_pos: shape={t5_input_ids_pos.shape}, min={t5_input_ids_pos.min().item()}, max={t5_input_ids_pos.max().item()}")
    
    t5_ids_pos = t5_input_ids_pos.unsqueeze(0) if t5_input_ids_pos.ndim == 1 else t5_input_ids_pos
    t5_ids_neg = t5_input_ids_neg.unsqueeze(0) if t5_input_ids_neg.ndim == 1 else t5_input_ids_neg
    
    with torch.no_grad():
        context_pos = dit.preprocess_text_embeds(text_embeds_pos, t5_ids_pos)
        context_neg = dit.preprocess_text_embeds(text_embeds_neg, t5_ids_neg)
    
    if context_pos.shape[1] < 512:
        context_pos = torch.nn.functional.pad(context_pos, (0, 0, 0, 512 - context_pos.shape[1]))
    if context_neg.shape[1] < 512:
        context_neg = torch.nn.functional.pad(context_neg, (0, 0, 0, 512 - context_neg.shape[1]))
    
    print(f"  Positive context after: shape={context_pos.shape}, mean={context_pos.mean().item():.4f}, std={context_pos.std().item():.4f}")
    print(f"  Negative context after: shape={context_neg.shape}, mean={context_neg.mean().item():.4f}, std={context_neg.std().item():.4f}")
    
    del text_embeds_pos, t5_input_ids_pos, text_embeds_neg, t5_input_ids_neg
    gc.collect()
    torch.cuda.empty_cache()
    
    def time_snr_shift(alpha, t):
        if alpha == 1.0:
            return t
        return alpha * t / (1 + (alpha - 1) * t)
    
    print(f"\nSampling settings:")
    print(f"  Resolution: {height}x{width}")
    print(f"  Latent shape: ({latent_c}, {latent_t}, {latent_h}, {latent_w})")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance Scale: {guidance_scale}")
    print(f"  Shift: {shift}")
    
    latents = torch.randn(1, latent_c, latent_t, latent_h, latent_w, device=device, dtype=torch.float32)
    
    sigmas = []
    for i in range(num_inference_steps):
        t = (num_inference_steps - i) / num_inference_steps
        sigma = time_snr_shift(shift, t)
        sigmas.append(sigma)
    sigmas.append(0.0)
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float64)
    
    print(f"  Sigmas[0]={sigmas[0].item():.4f}, Sigmas[-2]={sigmas[-2].item():.4f}")
    
    print("\nSampling (Flow Matching with CONST)...")
    print(f"  Initial latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    
    with torch.no_grad():
        for i in tqdm(range(len(sigmas) - 1)):
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
            
            if i == 0:
                print(f"  Step 0: sigma={current_sigma.item():.4f}")
                print(f"    noise_pred: mean={noise_pred.mean().item():.4f}, std={noise_pred.std().item():.4f}")
                print(f"    denoised: mean={denoised.mean().item():.4f}, std={denoised.std().item():.4f}")
            
            d = (latents - denoised) / current_sigma
            latents = latents + d * (next_sigma - current_sigma)
            
            if i == 0:
                print(f"    latents after step: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    
    print(f"  Final latents (before process_out): mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    
    latents_mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ]).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ]).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std + latents_mean
    
    print(f"  Final latents (after process_out): mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    
    del dit, context_pos, context_neg
    gc.collect()
    torch.cuda.empty_cache()
    print("DiT unloaded")
    
    print("\n[3/4] Loading VAE and decoding...")
    
    from anima_modules.vae import AnimaVAE, load_anima_vae
    vae = load_anima_vae(vae_path, device=device, dtype=dtype)
    print("VAE loaded successfully")
    
    print(f"  Latents before VAE: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    
    with torch.no_grad():
        images = vae.decode([latents], already_processed=True)
        image = images[0]
    
    print(f"  - Decoded image shape: {image.shape}")
    
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    print("VAE unloaded")
    
    while image.dim() > 3:
        image = image.squeeze(1) if image.shape[1] == 1 else image.squeeze(0)
    
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().float().numpy()
    image = (image * 255).astype("uint8")
    
    output_dir = r"D:\test"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "anima_inference_test.png")
    Image.fromarray(image).save(output_path)
    print(f"\nImage saved to: {output_path}")
    
    print("\n[SUCCESS] Inference completed!")

if __name__ == "__main__":
    run_inference()
