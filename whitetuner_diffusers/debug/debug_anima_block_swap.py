import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

def test_block_swap_simple():
    print("=" * 60)
    print("Test 1: Block Swap Basic Functionality")
    print("=" * 60)
    
    from anima_modules.model import load_anima_model
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Loading Anima model to CPU (for block swap)...")
    model = load_anima_model(dit_path, device="cpu", dtype=dtype)
    
    print(f"Number of blocks: {len(model.blocks)}")
    
    blocks_to_swap = 20
    print(f"\nEnabling block swap: {blocks_to_swap} blocks")
    model.enable_block_swap(
        blocks_to_swap=blocks_to_swap,
        device=device,
        supports_backward=True,
        use_pinned_memory=False,
    )
    
    model.move_to_device_except_swap_blocks(device)
    model.prepare_block_swap_before_forward()
    
    num_on_gpu = 0
    num_on_cpu = 0
    for i, block in enumerate(model.blocks):
        first_param = next(block.parameters())
        if first_param.device.type == "cuda":
            num_on_gpu += 1
        else:
            num_on_cpu += 1
    
    print(f"Blocks on GPU: {num_on_gpu}")
    print(f"Blocks on CPU: {num_on_cpu}")
    
    expected_on_gpu = len(model.blocks) - blocks_to_swap
    if num_on_gpu == expected_on_gpu:
        print(f"[PASS] Block distribution correct!")
    else:
        print(f"[FAIL] Expected {expected_on_gpu} on GPU, got {num_on_gpu}")
        return False
    
    return True


def test_block_swap_forward():
    print("\n" + "=" * 60)
    print("Test 2: Block Swap Forward Pass")
    print("=" * 60)
    
    from anima_modules.model import load_anima_model
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return None
    
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
    
    B = 1
    C = 16
    T, H, W = 1, 64, 64
    latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    timesteps = torch.tensor([500], device=device)
    context = torch.randn(B, 512, 1024, device=device, dtype=dtype)
    
    print(f"\nForward pass...")
    print(f"  latents: {latents.shape}")
    print(f"  timesteps: {timesteps}")
    print(f"  context: {context.shape}")
    
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**3
    
    with torch.autocast(device_type="cuda", dtype=dtype):
        output = model(latents, timesteps=timesteps, context=context)
    
    mem_after = torch.cuda.memory_allocated() / 1024**3
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Memory: {mem_before:.2f}GB -> {mem_after:.2f}GB (peak: {mem_peak:.2f}GB)")
    
    if output.shape == latents.shape:
        print(f"[PASS] Forward pass successful!")
        return True
    else:
        print(f"[FAIL] Output shape mismatch!")
        return False


def test_block_swap_backward():
    print("\n" + "=" * 60)
    print("Test 3: Block Swap Backward Pass")
    print("=" * 60)
    
    from anima_modules.model import load_anima_model
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return None
    
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
    
    B = 1
    C = 16
    T, H, W = 1, 64, 64
    latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500], device=device)
    context = torch.randn(B, 512, 1024, device=device, dtype=dtype)
    
    sigma = 0.5
    noisy_latents = (1 - sigma) * latents + sigma * noise
    
    print(f"\nForward pass...")
    torch.cuda.reset_peak_memory_stats()
    
    with torch.autocast(device_type="cuda", dtype=dtype):
        output = model(noisy_latents, timesteps=timesteps, context=context)
    
    target = noise - latents
    loss = F.mse_loss(output.float(), target.float())
    print(f"Loss: {loss.item():.6f}")
    
    print(f"\nBackward pass...")
    loss.backward()
    
    grad_count = 0
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            grad_norm += param.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    
    print(f"Parameters with gradients: {grad_count}")
    print(f"Total gradient norm: {grad_norm:.6f}")
    
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak memory: {mem_peak:.2f}GB")
    
    if grad_count > 0 and grad_norm > 0:
        print(f"[PASS] Backward pass successful!")
        return True
    else:
        print(f"[FAIL] No gradients computed!")
        return False


def test_adafactor_with_block_swap():
    print("\n" + "=" * 60)
    print("Test 4: Adafactor + Block Swap")
    print("=" * 60)
    
    from anima_modules.model import load_anima_model
    from transformers import Adafactor
    from adafactor_fused import patch_adafactor_fused
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return None
    
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
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = Adafactor(
        trainable_params,
        lr=1e-5,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    patch_adafactor_fused(optimizer)
    print(f"Adafactor optimizer created (fused)")
    
    first_param = trainable_params[0]
    weight_before = first_param.data.clone()
    
    B = 1
    C = 16
    T, H, W = 1, 64, 64
    latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500], device=device)
    context = torch.randn(B, 512, 1024, device=device, dtype=dtype)
    
    sigma = 0.5
    noisy_latents = (1 - sigma) * latents + sigma * noise
    
    print(f"\nTraining step 1...")
    
    with torch.autocast(device_type="cuda", dtype=dtype):
        output = model(noisy_latents, timesteps=timesteps, context=context)
    
    target = noise - latents
    loss = F.mse_loss(output.float(), target.float())
    print(f"Loss: {loss.item():.6f}")
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    weight_after = first_param.data
    weight_diff = (weight_after - weight_before).abs().max().item()
    
    print(f"Weight max diff: {weight_diff:.10f}")
    
    if weight_diff > 0:
        print(f"[PASS] Adafactor + Block Swap works!")
        return True
    else:
        print(f"[FAIL] Weights not updated!")
        return False


if __name__ == "__main__":
    print("Testing Anima Block Swap\n")
    
    results = {}
    
    try:
        results["test1"] = test_block_swap_simple()
    except Exception as e:
        print(f"\n[ERROR] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        results["test1"] = False
    
    try:
        results["test2"] = test_block_swap_forward()
    except Exception as e:
        print(f"\n[ERROR] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        results["test2"] = False
    
    try:
        results["test3"] = test_block_swap_backward()
    except Exception as e:
        print(f"\n[ERROR] Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        results["test3"] = False
    
    try:
        results["test4"] = test_adafactor_with_block_swap()
    except Exception as e:
        print(f"\n[ERROR] Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        results["test4"] = False
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Test 1 (Block Distribution): {'PASS' if results.get('test1') else 'SKIP' if results.get('test1') is None else 'FAIL'}")
    print(f"Test 2 (Forward Pass): {'PASS' if results.get('test2') else 'SKIP' if results.get('test2') is None else 'FAIL'}")
    print(f"Test 3 (Backward Pass): {'PASS' if results.get('test3') else 'SKIP' if results.get('test3') is None else 'FAIL'}")
    print(f"Test 4 (Adafactor + Block Swap): {'PASS' if results.get('test4') else 'SKIP' if results.get('test4') is None else 'FAIL'}")
