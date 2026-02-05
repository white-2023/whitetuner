import torch
import sys
sys.path.insert(0, 'D:/ai/whitetuner/whitetuner_diffusers')

from diffusers import FlowMatchEulerDiscreteScheduler

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def test_scheduler():
    print("=" * 60)
    print("测试 FlowMatchEulerDiscreteScheduler")
    print("=" * 60)
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
    )
    
    latent_height = 128
    latent_width = 128
    image_seq_len = (latent_height // 2) * (latent_width // 2)
    mu = calculate_shift(image_seq_len)
    
    print(f"\nlatent size: {latent_height}x{latent_width}")
    print(f"image_seq_len: {image_seq_len}")
    print(f"mu (shift): {mu}")
    
    num_steps = 30
    scheduler.set_timesteps(num_steps, device='cpu', mu=mu)
    
    print(f"\n设置 {num_steps} 步后:")
    print(f"  timesteps type: {type(scheduler.timesteps)}")
    print(f"  timesteps shape: {scheduler.timesteps.shape}")
    print(f"  timesteps: {scheduler.timesteps[:5].tolist()} ... {scheduler.timesteps[-5:].tolist()}")
    
    if hasattr(scheduler, 'sigmas'):
        print(f"\n  sigmas type: {type(scheduler.sigmas)}")
        print(f"  sigmas shape: {scheduler.sigmas.shape}")
        print(f"  sigmas: {scheduler.sigmas[:5].tolist()} ... {scheduler.sigmas[-5:].tolist()}")
    else:
        print("\n  没有 sigmas 属性!")
    
    print("\n" + "=" * 60)
    print("测试 scheduler.step 的行为")
    print("=" * 60)
    
    batch_size = 1
    channels = 16
    noise = torch.randn(batch_size, channels, latent_height, latent_width)
    
    print(f"\n模拟推理过程:")
    current = noise.clone()
    trajectory = [current.clone()]
    
    for i, t in enumerate(scheduler.timesteps[:5]):
        fake_velocity = torch.randn_like(current) * 0.1
        result = scheduler.step(fake_velocity, t, current, return_dict=True)
        current = result.prev_sample
        trajectory.append(current.clone())
        
        print(f"  step {i}: timestep={t.item():.2f}")
        if hasattr(result, 'prev_sample'):
            print(f"    prev_sample shape: {result.prev_sample.shape}")
    
    print("\n" + "=" * 60)
    print("验证 velocity 计算公式")
    print("=" * 60)
    
    scheduler.set_timesteps(num_steps, device='cpu', mu=mu)
    
    if hasattr(scheduler, 'sigmas'):
        sigmas = scheduler.sigmas
        print(f"\nsigmas 长度: {len(sigmas)}")
        print(f"timesteps 长度: {len(scheduler.timesteps)}")
        
        for i in range(min(5, len(scheduler.timesteps))):
            t = scheduler.timesteps[i]
            sigma = sigmas[i] if i < len(sigmas) else None
            sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else 0
            
            print(f"\n  step {i}:")
            print(f"    timestep: {t.item():.4f}")
            print(f"    sigma: {sigma.item() if sigma is not None else 'N/A':.4f}")
            print(f"    sigma_next: {sigma_next.item() if isinstance(sigma_next, torch.Tensor) else sigma_next:.4f}")
            print(f"    delta_sigma: {(sigma_next - sigma).item() if sigma is not None else 'N/A':.4f}")
    
    print("\n" + "=" * 60)
    print("验证 target velocity = (next_latent - current_latent) / delta_sigma")
    print("=" * 60)
    
    current = noise.clone()
    
    for i in range(min(3, len(scheduler.timesteps))):
        t = scheduler.timesteps[i]
        
        true_velocity = torch.randn_like(current)
        
        result = scheduler.step(true_velocity, t, current, return_dict=True)
        next_latent = result.prev_sample
        
        sigma = scheduler.sigmas[i]
        sigma_next = scheduler.sigmas[i + 1] if i + 1 < len(scheduler.sigmas) else torch.tensor(0.0)
        delta_sigma = sigma_next - sigma
        
        if abs(delta_sigma.item()) > 1e-8:
            recovered_velocity = (next_latent - current) / delta_sigma
            
            diff = (recovered_velocity - true_velocity).abs().mean().item()
            print(f"\n  step {i}:")
            print(f"    delta_sigma: {delta_sigma.item():.6f}")
            print(f"    true_velocity mean: {true_velocity.mean().item():.6f}")
            print(f"    recovered_velocity mean: {recovered_velocity.mean().item():.6f}")
            print(f"    差异 (应该接近 0): {diff:.10f}")
        
        current = next_latent

if __name__ == "__main__":
    test_scheduler()
