import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

def test_loss_calculation():
    print("=" * 60)
    print("Testing Anima Loss Calculation")
    print("=" * 60)
    
    from anima_modules.model import load_anima_model
    
    dit_path = "F:/models/circlestone-labs-Anima/split_files/diffusion_models/anima-preview.safetensors"
    
    if not os.path.exists(dit_path):
        print(f"[SKIP] DiT not found: {dit_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Loading Anima model...")
    model = load_anima_model(dit_path, device=device, dtype=dtype)
    model.eval()
    
    B = 1
    C = 16
    T, H, W = 1, 64, 64
    
    latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    context = torch.randn(B, 512, 1024, device=device, dtype=dtype)
    
    test_timesteps = [100, 500, 900]
    
    for t_val in test_timesteps:
        timesteps = torch.tensor([t_val], device=device)
        sigma = timesteps.float() / 1000.0
        sigma_5d = sigma.view(-1, 1, 1, 1, 1).to(dtype)
        
        noisy_latents = (1 - sigma_5d) * latents + sigma_5d * noise
        
        print(f"\n--- Timestep {t_val}, sigma={sigma.item():.3f} ---")
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            print("\n[Method 1] Current training (no input scaling):")
            model_pred_1 = model(noisy_latents, timesteps=timesteps, context=context)
        
        target_1 = noise - latents
        loss_1 = F.mse_loss(model_pred_1.float(), target_1.float())
        print(f"  Loss: {loss_1.item():.4f}")
        print(f"  Pred range: [{model_pred_1.min().item():.3f}, {model_pred_1.max().item():.3f}]")
        print(f"  Target range: [{target_1.min().item():.3f}, {target_1.max().item():.3f}]")
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            print("\n[Method 2] With ComfyUI input scaling:")
            sigma_scaled = (sigma / (sigma + 1)).to(dtype)
            scaled_input = noisy_latents * (1.0 - sigma_scaled.view(-1, 1, 1, 1, 1))
            model_pred_2 = model(scaled_input, timesteps=timesteps, context=context)
        
        target_2 = noise - latents
        loss_2 = F.mse_loss(model_pred_2.float(), target_2.float())
        print(f"  Loss: {loss_2.item():.4f}")
        print(f"  Pred range: [{model_pred_2.min().item():.3f}, {model_pred_2.max().item():.3f}]")
        
        print("\n[Method 3] Direct noise prediction (epsilon):")
        target_3 = noise
        loss_3 = F.mse_loss(model_pred_1.float(), target_3.float())
        print(f"  Loss: {loss_3.item():.4f}")
        
        print("\n[Method 4] Direct x0 prediction:")
        target_4 = latents
        loss_4 = F.mse_loss(model_pred_1.float(), target_4.float())
        print(f"  Loss: {loss_4.item():.4f}")
        
        print("\n[Method 5] Scaled v-prediction (target / sigma):")
        sigma_val = sigma.item()
        if sigma_val > 0.01:
            target_5 = (noise - latents) * sigma_val
            loss_5 = F.mse_loss(model_pred_1.float(), target_5.float())
            print(f"  Loss: {loss_5.item():.4f}")
            print(f"  Target range: [{target_5.min().item():.3f}, {target_5.max().item():.3f}]")
        
        print("\n[Method 6] Reconstructed v from model output:")
        sigma_scaled = (sigma / (sigma + 1)).item()
        reconstructed_x0 = noisy_latents * (1 - sigma_scaled) * (1 - sigma_val) - model_pred_1 * sigma_val
        recon_loss = F.mse_loss(reconstructed_x0.float(), latents.float())
        print(f"  Reconstruction loss (should be low if correct): {recon_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("- Loss around 2.0 suggests random prediction (no correlation)")
    print("- Loss << 1.0 suggests correct target alignment")
    print("- If all methods have similar high loss, the model or input format may be wrong")
    print("=" * 60)


def test_perfect_prediction():
    print("\n" + "=" * 60)
    print("Testing Perfect Prediction Baseline")
    print("=" * 60)
    
    B, C, T, H, W = 1, 16, 1, 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    latents = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    
    target = noise - latents
    
    perfect_pred = target
    loss_perfect = F.mse_loss(perfect_pred.float(), target.float())
    print(f"Perfect prediction loss: {loss_perfect.item():.6f}")
    
    noisy_pred = target + 0.1 * torch.randn_like(target)
    loss_noisy = F.mse_loss(noisy_pred.float(), target.float())
    print(f"Slightly noisy prediction loss: {loss_noisy.item():.6f}")
    
    random_pred = torch.randn_like(target)
    loss_random = F.mse_loss(random_pred.float(), target.float())
    print(f"Random prediction loss: {loss_random.item():.4f}")
    
    print(f"\nTarget (v = noise - latents) stats:")
    print(f"  Mean: {target.mean().item():.4f}")
    print(f"  Std: {target.std().item():.4f}")
    print(f"  Range: [{target.min().item():.3f}, {target.max().item():.3f}]")
    
    print(f"\nExpected random loss (2 * var):")
    var = target.var().item()
    print(f"  Variance: {var:.4f}")
    print(f"  2 * Variance: {2 * var:.4f}")


if __name__ == "__main__":
    test_perfect_prediction()
    test_loss_calculation()
