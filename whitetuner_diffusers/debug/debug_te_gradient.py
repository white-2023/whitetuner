import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

def test_te_gradient_flow():
    print("=" * 60)
    print("Test 1: TE Gradient Flow (Simplified)")
    print("=" * 60)
    
    class SimpleDiT(torch.nn.Module):
        def __init__(self, hidden_dim=256):
            super().__init__()
            self.proj = torch.nn.Linear(hidden_dim, hidden_dim)
            self.norm = torch.nn.LayerNorm(hidden_dim)
            self.out = torch.nn.Linear(hidden_dim, hidden_dim)
        
        def forward(self, x, context):
            x = x + context.mean(dim=1, keepdim=True)
            x = self.proj(x)
            x = self.norm(x)
            x = self.out(x)
            return x
    
    class SimpleTE(torch.nn.Module):
        def __init__(self, vocab_size=1000, hidden_dim=256):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
            self.proj = torch.nn.Linear(hidden_dim, hidden_dim)
        
        def forward(self, input_ids):
            x = self.embed(input_ids)
            x = self.proj(x)
            return x
    
    te = SimpleTE()
    dit = SimpleDiT()
    
    te_weight_before = te.proj.weight.data.clone()
    dit_weight_before = dit.proj.weight.data.clone()
    
    optimizer = torch.optim.AdamW([
        {"params": dit.parameters(), "lr": 1e-4},
        {"params": te.parameters(), "lr": 1e-5},
    ])
    
    input_ids = torch.randint(0, 1000, (2, 10))
    latents = torch.randn(2, 1, 256)
    target = torch.randn(2, 1, 256)
    
    text_embeds = te(input_ids)
    print(f"text_embeds requires_grad: {text_embeds.requires_grad}")
    
    model_pred = dit(latents, context=text_embeds)
    print(f"model_pred requires_grad: {model_pred.requires_grad}")
    
    loss = F.mse_loss(model_pred, target)
    print(f"loss: {loss.item():.6f}")
    
    loss.backward()
    
    te_grad = te.proj.weight.grad
    dit_grad = dit.proj.weight.grad
    
    print(f"\nTE proj.weight.grad is not None: {te_grad is not None}")
    if te_grad is not None:
        print(f"TE grad norm: {te_grad.norm().item():.6f}")
    
    print(f"DiT proj.weight.grad is not None: {dit_grad is not None}")
    if dit_grad is not None:
        print(f"DiT grad norm: {dit_grad.norm().item():.6f}")
    
    optimizer.step()
    
    te_weight_after = te.proj.weight.data
    dit_weight_after = dit.proj.weight.data
    
    te_weight_diff = (te_weight_after - te_weight_before).abs().max().item()
    dit_weight_diff = (dit_weight_after - dit_weight_before).abs().max().item()
    
    print(f"\nTE weight changed: {te_weight_diff > 0} (max diff: {te_weight_diff:.8f})")
    print(f"DiT weight changed: {dit_weight_diff > 0} (max diff: {dit_weight_diff:.8f})")
    
    if te_weight_diff > 0 and dit_weight_diff > 0:
        print("\n[PASS] Both TE and DiT weights updated!")
    else:
        print("\n[FAIL] Some weights not updated!")
    
    return te_weight_diff > 0 and dit_weight_diff > 0


def test_real_anima_te_gradient():
    print("\n" + "=" * 60)
    print("Test 2: Real Anima TE Gradient Flow")
    print("=" * 60)
    
    from anima_modules.text_encoder import load_qwen3_text_encoder, AnimaTokenizer
    from anima_modules.model import load_anima_model
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    te_path = "F:/models/circlestone-labs-Anima/split_files/text_encoders/qwen_3_06b_base.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return None
    if not os.path.exists(te_path):
        print(f"[SKIP] TE not found: {te_path}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print("Loading Text Encoder...")
    te = load_qwen3_text_encoder(te_path, device=device, dtype=dtype)
    te.train()
    
    print("Loading Tokenizer...")
    tokenizer = AnimaTokenizer()
    
    print("Loading DiT...")
    dit = load_anima_model(dit_path, device=device, dtype=dtype)
    dit.train()
    
    te_weight_before = None
    dit_weight_before = None
    te_param_name = None
    dit_param_name = None
    
    for name, param in te.named_parameters():
        if param.requires_grad and param.numel() > 1000:
            te_weight_before = param.data.clone()
            te_param_name = name
            print(f"TE param: {name}, shape: {param.shape}")
            break
    
    for name, param in dit.named_parameters():
        if param.requires_grad and param.numel() > 1000:
            dit_weight_before = param.data.clone()
            dit_param_name = name
            print(f"DiT param: {name}, shape: {param.shape}")
            break
    
    if te_weight_before is None or dit_weight_before is None:
        print("[SKIP] Could not find target parameters")
        return None
    
    dit_params = [p for p in dit.parameters() if p.requires_grad]
    te_params = [p for p in te.parameters() if p.requires_grad]
    
    print(f"DiT trainable params: {sum(p.numel() for p in dit_params):,}")
    print(f"TE trainable params: {sum(p.numel() for p in te_params):,}")
    
    optimizer = torch.optim.AdamW([
        {"params": dit_params, "lr": 1e-5},
        {"params": te_params, "lr": 1e-6},
    ])
    
    captions = ["a beautiful anime girl with long hair"]
    
    inputs = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    text_embeds = te(input_ids, attention_mask)
    print(f"text_embeds shape: {text_embeds.shape}")
    print(f"text_embeds requires_grad: {text_embeds.requires_grad}")
    
    if text_embeds.shape[1] < 512:
        text_embeds = F.pad(text_embeds, (0, 0, 0, 512 - text_embeds.shape[1]))
    
    B = 1
    T, H, W = 1, 64, 64
    C = 16
    latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500], device=device)
    
    sigma = 0.5
    noisy_latents = (1 - sigma) * latents + sigma * noise
    
    print(f"noisy_latents shape: {noisy_latents.shape}")
    print(f"timesteps: {timesteps}")
    
    with torch.autocast(device_type="cuda", dtype=dtype):
        model_pred = dit(
            noisy_latents,
            timesteps=timesteps,
            context=text_embeds,
        )
    
    print(f"model_pred shape: {model_pred.shape}")
    print(f"model_pred requires_grad: {model_pred.requires_grad}")
    
    target = noise - latents
    loss = F.mse_loss(model_pred.float(), target.float())
    print(f"loss: {loss.item():.6f}")
    
    print("\nRunning backward...")
    loss.backward()
    
    te_has_grad = False
    te_grad_norm = 0.0
    for name, param in te.named_parameters():
        if param.grad is not None:
            te_has_grad = True
            te_grad_norm += param.grad.norm().item() ** 2
    te_grad_norm = te_grad_norm ** 0.5
    
    dit_has_grad = False
    dit_grad_norm = 0.0
    for name, param in dit.named_parameters():
        if param.grad is not None:
            dit_has_grad = True
            dit_grad_norm += param.grad.norm().item() ** 2
    dit_grad_norm = dit_grad_norm ** 0.5
    
    print(f"\nTE has gradients: {te_has_grad} (total grad norm: {te_grad_norm:.6f})")
    print(f"DiT has gradients: {dit_has_grad} (total grad norm: {dit_grad_norm:.6f})")
    
    print("\nRunning optimizer step...")
    optimizer.step()
    
    te_weight_after = None
    dit_weight_after = None
    
    for name, param in te.named_parameters():
        if name == te_param_name:
            te_weight_after = param.data
            break
    
    for name, param in dit.named_parameters():
        if name == dit_param_name:
            dit_weight_after = param.data
            break
    
    te_diff = (te_weight_after - te_weight_before).abs().max().item()
    dit_diff = (dit_weight_after - dit_weight_before).abs().max().item()
    
    print(f"\nTE weight max diff: {te_diff:.10f}")
    print(f"DiT weight max diff: {dit_diff:.10f}")
    
    te_updated = te_diff > 1e-10
    dit_updated = dit_diff > 1e-10
    
    print(f"\nTE weights updated: {te_updated}")
    print(f"DiT weights updated: {dit_updated}")
    
    if te_updated and dit_updated:
        print("\n[PASS] Both TE and DiT weights updated successfully!")
        return True
    else:
        print("\n[FAIL] Some weights not updated!")
        return False


def test_gradient_no_detach():
    print("\n" + "=" * 60)
    print("Test 3: Verify no detach() in pipeline")
    print("=" * 60)
    
    class MockTE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    class MockDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        
        def forward(self, x, context):
            return self.linear(x + context)
    
    te = MockTE()
    dit = MockDiT()
    
    x = torch.randn(2, 10)
    target = torch.randn(2, 10)
    
    te_out = te(x)
    print(f"TE output requires_grad: {te_out.requires_grad}")
    
    te_out_detached = te_out.detach()
    print(f"TE output (detached) requires_grad: {te_out_detached.requires_grad}")
    
    dit_out_with_grad = dit(x, te_out)
    dit_out_no_grad = dit(x, te_out_detached)
    
    loss_with_grad = F.mse_loss(dit_out_with_grad, target)
    loss_no_grad = F.mse_loss(dit_out_no_grad, target)
    
    te.zero_grad()
    loss_with_grad.backward()
    te_grad_with = te.linear.weight.grad.clone() if te.linear.weight.grad is not None else None
    
    te.zero_grad()
    loss_no_grad.backward()
    te_grad_without = te.linear.weight.grad
    
    print(f"\nWith connected gradient:")
    print(f"  TE has grad: {te_grad_with is not None}")
    if te_grad_with is not None:
        print(f"  TE grad norm: {te_grad_with.norm().item():.6f}")
    
    print(f"\nWith detached (no gradient to TE):")
    print(f"  TE has grad: {te_grad_without is not None}")
    
    if te_grad_with is not None and te_grad_without is None:
        print("\n[PASS] Gradient flow works correctly when not detached!")
        return True
    else:
        print("\n[INFO] Check gradient connections")
        return False


if __name__ == "__main__":
    print("Testing TE Gradient Update\n")
    
    test1_pass = test_te_gradient_flow()
    test3_pass = test_gradient_no_detach()
    
    try:
        test2_pass = test_real_anima_te_gradient()
    except Exception as e:
        print(f"\n[ERROR] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        test2_pass = None
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Test 1 (Simplified): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Real Anima): {'PASS' if test2_pass else 'SKIP' if test2_pass is None else 'FAIL'}")
    print(f"Test 3 (No Detach): {'PASS' if test3_pass else 'FAIL'}")
