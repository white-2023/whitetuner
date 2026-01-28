# -*- coding: utf-8 -*-

import os
import sys
import gc
import torch
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "whitetuner_diffusers"))

from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from qwen_modules import QwenImageTransformer2DModel
from optimum.quanto import quantize, freeze, qint8
from base_trainer import pack_latents, unpack_latents
from tqdm import tqdm

base_model_path = r"/root/Qwen-Image-2512"
trained_transformer_path = r"/root/test/output/checkpoints/checkpoint-2000/transformer"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "qwen_image.png")

prompt = "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm."
negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
}
aspect_ratio = "1:1"
width, height = aspect_ratios[aspect_ratio]

seed = 42
num_inference_steps = 50
vae_scale_factor = 8

quantize_text_encoder = True
quantize_transformer = True

torch_dtype = torch.bfloat16
device = "cuda"


def print_vram(msg=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] {msg}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")


def encode_prompt(text_encoder, tokenizer, prompt, device, dtype):
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        prompt_embeds = outputs.hidden_states[-1]
    
    valid_length = attention_mask.sum().item()
    prompt_embeds = prompt_embeds[:, :valid_length, :]
    attention_mask = attention_mask[:, :valid_length]
    
    return prompt_embeds, attention_mask


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def run_inference(transformer, scheduler, prompt_embeds, attention_mask, height, width, num_steps, seed, device, dtype):
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    num_channels = 16
    
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, num_channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    
    image_seq_len = (latent_height // 2) * (latent_width // 2)
    mu = calculate_shift(image_seq_len)
    scheduler.set_timesteps(num_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps
    
    txt_seq_lens = [attention_mask.sum().item()]
    img_shapes = [[(1, latent_height // 2, latent_width // 2)]]
    
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        packed_latents = pack_latents(latents, 1, num_channels, latent_height, latent_width)
        
        timestep_normalized = t.float() / 1000.0
        timestep_tensor = timestep_normalized.unsqueeze(0)
        
        with torch.no_grad():
            model_output = transformer(
                hidden_states=packed_latents,
                timestep=timestep_tensor,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=attention_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )
        
        model_output = unpack_latents(model_output, height, width, vae_scale_factor)
        if model_output.dim() == 5:
            model_output = model_output.squeeze(2)
        
        latents = scheduler.step(model_output, t, latents, return_dict=False)[0]
    
    return latents


def decode_latents(vae, latents, dtype):
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, dtype)
    
    latents = latents.unsqueeze(2)
    latents = latents / latents_std + latents_mean
    
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    
    image = image[:, :, 0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")[0]
    
    from PIL import Image
    return Image.fromarray(image)


name, ext = os.path.splitext(output_path)
output_path_1 = f"{name}_model1{ext}"
output_path_2 = f"{name}_model2{ext}"

print(f"Prompt: {prompt[:50]}...")
print(f"Size: {width}x{height}, Steps: {num_inference_steps}, Seed: {seed}")
print("-" * 60)

print("\n[Stage 1] Loading Text Encoder...")
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    subfolder="text_encoder",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)
text_encoder.requires_grad_(False)
text_encoder.eval()
text_encoder.model.visual = None
print("Removed visual module")

if quantize_text_encoder:
    print("Quantizing Text Encoder (qint8)...")
    quantize(text_encoder, weights=qint8, exclude=["*embed*", "*lm_head*"])
    freeze(text_encoder)

text_encoder.to(device)

tokenizer = Qwen2Tokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
print_vram("After TE loaded")

print("\n[Stage 2] Encoding prompt...")
prompt_embeds, attention_mask = encode_prompt(text_encoder, tokenizer, prompt, device, torch_dtype)
print(f"Prompt embeds shape: {prompt_embeds.shape}")

print("\n[Stage 3] Unloading Text Encoder...")
del text_encoder, tokenizer
gc.collect()
torch.cuda.empty_cache()
print_vram("After TE unloaded")

print("\n[Stage 4] Loading original Transformer...")
transformer = QwenImageTransformer2DModel.from_pretrained(
    base_model_path,
    subfolder="transformer",
    torch_dtype=torch_dtype,
)
transformer.requires_grad_(False)
transformer.eval()

orig_weight_sample = None
for name, param in transformer.named_parameters():
    if "transformer_blocks.0" in name and "weight" in name:
        orig_weight_sample = param.data.clone().cpu()
        print(f"Original model weight sample ({name}): mean={param.data.float().mean().item():.6f}, std={param.data.float().std().item():.6f}")
        break

if quantize_transformer:
    print("Quantizing Transformer (qint8)...")
    quantize(transformer, weights=qint8, exclude=["*norm*", "proj_out*", "*embedder*"])
    freeze(transformer)

transformer.to(device)
print_vram("After original Transformer loaded")

scheduler_1 = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True)

print("\n[Stage 5] Running inference with original model...")
start_time = time.time()
latents_1 = run_inference(transformer, scheduler_1, prompt_embeds, attention_mask, height, width, num_inference_steps, seed, device, torch_dtype)
print(f"Inference done! Time: {time.time() - start_time:.2f}s")
print(f"latents_1: mean={latents_1.mean().item():.4f}, std={latents_1.std().item():.4f}")

print("\n[Stage 6] Unloading Transformer, loading VAE for first decode...")
del transformer
gc.collect()
torch.cuda.empty_cache()

vae = AutoencoderKLQwenImage.from_pretrained(
    base_model_path,
    subfolder="vae",
    torch_dtype=torch_dtype,
)
vae.requires_grad_(False)
vae.eval()
vae.to(device)

image_1 = decode_latents(vae, latents_1, torch_dtype)
image_1.save(output_path_1)
print(f"Original model output saved: {output_path_1}")

del latents_1, vae
gc.collect()
torch.cuda.empty_cache()
print_vram("After first decode and cleanup")

print("\n[Stage 7] Loading trained Transformer...")
transformer = QwenImageTransformer2DModel.from_pretrained(
    trained_transformer_path,
    torch_dtype=torch_dtype,
)
transformer.requires_grad_(False)
transformer.eval()

for name, param in transformer.named_parameters():
    if "transformer_blocks.0" in name and "weight" in name:
        trained_weight_sample = param.data.clone().cpu()
        print(f"Trained model weight sample ({name}): mean={param.data.float().mean().item():.6f}, std={param.data.float().std().item():.6f}")
        if orig_weight_sample is not None:
            diff = (orig_weight_sample.float() - trained_weight_sample.float()).abs().mean().item()
            print(f"Weight difference (original vs trained): {diff:.6f}")
            if diff < 1e-6:
                print("[WARNING] Weights are almost identical! Training may not have changed the model.")
        break

if quantize_transformer:
    print("Quantizing trained Transformer (qint8)...")
    quantize(transformer, weights=qint8, exclude=["*norm*", "proj_out*", "*embedder*"])
    freeze(transformer)

transformer.to(device)
print_vram("After trained Transformer loaded")

scheduler_2 = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True)

print("\n[Stage 8] Running inference with trained model...")
start_time = time.time()
latents_2 = run_inference(transformer, scheduler_2, prompt_embeds, attention_mask, height, width, num_inference_steps, seed, device, torch_dtype)
print(f"Inference done! Time: {time.time() - start_time:.2f}s")
print(f"latents_2: mean={latents_2.mean().item():.4f}, std={latents_2.std().item():.4f}")

print("\n[Stage 9] Unloading Transformer, loading VAE for second decode...")
del transformer
gc.collect()
torch.cuda.empty_cache()

vae = AutoencoderKLQwenImage.from_pretrained(
    base_model_path,
    subfolder="vae",
    torch_dtype=torch_dtype,
)
vae.requires_grad_(False)
vae.eval()
vae.to(device)

image_2 = decode_latents(vae, latents_2, torch_dtype)
image_2.save(output_path_2)
print(f"Trained model output saved: {output_path_2}")

print("\n" + "=" * 60)
print("Done!")
print(f"Original: {output_path_1}")
print(f"Trained:  {output_path_2}")
