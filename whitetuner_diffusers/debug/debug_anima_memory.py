import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import gc

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def get_gpu_reserved():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**3
    return 0

def test_epoch_memory_leak():
    print("=" * 60)
    print("Testing Epoch Boundary Memory Leak")
    print("=" * 60)
    
    from anima_modules.model import load_anima_model
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Loading Anima model to CPU...")
    model = load_anima_model(dit_path, device="cpu", dtype=dtype)
    
    blocks_to_swap = 20
    print(f"Enabling block swap: {blocks_to_swap} blocks")
    model.enable_block_swap(
        blocks_to_swap=blocks_to_swap,
        device=device,
        supports_backward=True,
        use_pinned_memory=False,
    )
    
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    model.train()
    model.requires_grad_(True)
    
    from transformers import Adafactor
    from adafactor_fused import patch_adafactor_fused
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adafactor(
        trainable_params,
        lr=1e-5,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    patch_adafactor_fused(optimizer)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_baseline = get_gpu_memory()
    print(f"\nBaseline memory: {mem_baseline:.3f}GB")
    
    B = 1
    C = 16
    T, H, W = 1, 64, 64
    
    epoch_size = 10
    num_epochs = 5
    
    print(f"\nSimulating {num_epochs} epochs x {epoch_size} steps = {num_epochs * epoch_size} total steps")
    
    epoch_end_memory = []
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1} ---")
        
        for step in range(epoch_size):
            latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (B,), device=device)
            context = torch.randn(B, 512, 1024, device=device, dtype=dtype)
            
            sigma = timesteps.float() / 1000.0
            sigma = sigma.view(-1, 1, 1, 1, 1)
            noisy_latents = (1 - sigma) * latents + sigma * noise
            
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = model(noisy_latents, timesteps=timesteps, context=context)
            
            target = noise - latents
            loss = F.mse_loss(output.float(), target.float())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            del latents, noise, noisy_latents, output, target, loss, context, sigma, timesteps
        
        gc.collect()
        torch.cuda.empty_cache()
        
        mem_after_epoch = get_gpu_memory()
        mem_reserved = get_gpu_reserved()
        epoch_end_memory.append(mem_after_epoch)
        print(f"Epoch {epoch + 1} end: allocated={mem_after_epoch:.3f}GB, reserved={mem_reserved:.3f}GB")
    
    print(f"\n--- Summary ---")
    for i, mem in enumerate(epoch_end_memory):
        delta = mem - epoch_end_memory[0] if i > 0 else 0
        print(f"Epoch {i + 1}: {mem:.3f}GB (delta: {delta:+.3f}GB)")
    
    total_growth = epoch_end_memory[-1] - epoch_end_memory[0]
    print(f"\nTotal memory growth: {total_growth:+.3f}GB")
    
    if abs(total_growth) < 0.1:
        print("[PASS] No epoch boundary memory leak!")
    else:
        print("[WARNING] Memory growing at epoch boundaries!")


def test_memory_leak():
    print("=" * 60)
    print("Testing Anima Memory Leak")
    print("=" * 60)
    
    from anima_modules.model import load_anima_model
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Loading Anima model to CPU...")
    model = load_anima_model(dit_path, device="cpu", dtype=dtype)
    
    blocks_to_swap = 20
    print(f"Enabling block swap: {blocks_to_swap} blocks")
    model.enable_block_swap(
        blocks_to_swap=blocks_to_swap,
        device=device,
        supports_backward=True,
        use_pinned_memory=False,
    )
    
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    model.train()
    model.requires_grad_(True)
    
    from transformers import Adafactor
    from adafactor_fused import patch_adafactor_fused
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adafactor(
        trainable_params,
        lr=1e-5,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    patch_adafactor_fused(optimizer)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_baseline = get_gpu_memory()
    print(f"\nBaseline memory: {mem_baseline:.3f}GB")
    
    B = 1
    C = 16
    T, H, W = 1, 64, 64
    
    mem_history = []
    
    num_steps = 10
    print(f"\nRunning {num_steps} training steps...")
    
    for step in range(num_steps):
        latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (B,), device=device)
        context = torch.randn(B, 512, 1024, device=device, dtype=dtype)
        
        sigma = timesteps.float() / 1000.0
        sigma = sigma.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - sigma) * latents + sigma * noise
        
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = model(noisy_latents, timesteps=timesteps, context=context)
        
        target = noise - latents
        loss = F.mse_loss(output.float(), target.float())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        del latents, noise, noisy_latents, output, target, loss, context, sigma, timesteps
        
        mem_after = get_gpu_memory()
        mem_history.append(mem_after)
        
        print(f"Step {step+1}: {mem_after:.3f}GB (delta: {mem_after - mem_baseline:+.3f}GB)")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_final = get_gpu_memory()
    print(f"\nFinal memory after gc: {mem_final:.3f}GB")
    
    mem_growth = mem_history[-1] - mem_history[0]
    print(f"\nMemory growth over {num_steps} steps: {mem_growth:+.3f}GB")
    
    if abs(mem_growth) < 0.1:
        print("[PASS] No significant memory leak detected!")
    else:
        print("[WARNING] Memory appears to be growing!")
        
        print("\nAnalyzing potential causes...")
        
        print("\n1. Checking optimizer state size...")
        opt_state_size = 0
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    opt_state_size += v.numel() * v.element_size()
        print(f"   Optimizer state: {opt_state_size / 1024**3:.3f}GB")
        
        print("\n2. Checking model parameters...")
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"   Model parameters: {param_size / 1024**3:.3f}GB")
        
        print("\n3. Checking gradient size...")
        grad_size = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
        print(f"   Gradients: {grad_size / 1024**3:.3f}GB")


def test_memory_without_block_swap():
    print("\n" + "=" * 60)
    print("Testing Memory WITHOUT Block Swap")
    print("=" * 60)
    
    from anima_modules.model import load_anima_model
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Loading Anima model to GPU...")
    model = load_anima_model(dit_path, device=device, dtype=dtype)
    model.train()
    model.requires_grad_(True)
    
    import bitsandbytes as bnb
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-6,
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_baseline = get_gpu_memory()
    print(f"\nBaseline memory: {mem_baseline:.3f}GB")
    
    B = 1
    C = 16
    T, H, W = 1, 32, 32
    
    mem_history = []
    
    num_steps = 5
    print(f"\nRunning {num_steps} training steps (smaller resolution)...")
    
    for step in range(num_steps):
        latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (B,), device=device)
        context = torch.randn(B, 256, 1024, device=device, dtype=dtype)
        
        sigma = timesteps.float() / 1000.0
        sigma = sigma.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - sigma) * latents + sigma * noise
        
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = model(noisy_latents, timesteps=timesteps, context=context)
        
        target = noise - latents
        loss = F.mse_loss(output.float(), target.float())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        del latents, noise, noisy_latents, output, target, loss, context, sigma, timesteps
        
        mem_after = get_gpu_memory()
        mem_history.append(mem_after)
        
        print(f"Step {step+1}: {mem_after:.3f}GB (delta: {mem_after - mem_baseline:+.3f}GB)")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_final = get_gpu_memory()
    print(f"\nFinal memory after gc: {mem_final:.3f}GB")
    
    mem_growth = mem_history[-1] - mem_history[0]
    print(f"\nMemory growth over {num_steps} steps: {mem_growth:+.3f}GB")
    
    if abs(mem_growth) < 0.1:
        print("[PASS] No significant memory leak detected!")
    else:
        print("[WARNING] Memory appears to be growing!")


if __name__ == "__main__":
    test_epoch_memory_leak()
