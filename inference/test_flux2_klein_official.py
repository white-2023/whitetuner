# -*- coding: utf-8 -*-
import os
import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "/root/FLUX.2-klein-base-9B"

pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype)
pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=50,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]

output_path = os.path.join(script_dir, "flux-klein-official.png")
image.save(output_path)
print(f"Done! Saved to {output_path}")

