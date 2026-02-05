import torch
import sys
sys.path.insert(0, 'D:/ai/whitetuner/whitetuner_diffusers')

from diffusers import FlowMatchEulerDiscreteScheduler

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def test_student_steps(teacher_steps, student_steps):
    print(f"\n{'='*70}")
    print(f"测试: 教师 {teacher_steps} 步, 学生 {student_steps} 步")
    print('='*70)
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
    )
    
    latent_height = 128
    latent_width = 128
    image_seq_len = (latent_height // 2) * (latent_width // 2)
    mu = calculate_shift(image_seq_len)
    
    batch_size = 1
    channels = 16
    noise = torch.randn(batch_size, channels, latent_height, latent_width)
    
    scheduler.set_timesteps(teacher_steps, device='cpu', mu=mu)
    teacher_timesteps = scheduler.timesteps.clone()
    
    teacher_trajectory = [noise.clone()]
    current = noise.clone()
    for i, t in enumerate(teacher_timesteps):
        fake_velocity = -current * 0.5
        current = scheduler.step(fake_velocity, t, current, return_dict=False)[0]
        teacher_trajectory.append(current.clone())
    
    print(f"教师轨迹长度: {len(teacher_trajectory)}")
    
    scheduler.set_timesteps(student_steps, device='cpu', mu=mu)
    student_timesteps = scheduler.timesteps
    student_sigmas = scheduler.sigmas
    
    print(f"学生 timesteps ({len(student_timesteps)}): {[f'{t.item():.1f}' for t in student_timesteps]}")
    print(f"学生 sigmas ({len(student_sigmas)}): {[f'{s.item():.4f}' for s in student_sigmas]}")
    
    all_passed = True
    total_diff = 0
    
    for s in range(student_steps):
        student_t = student_timesteps[s]
        student_t_next = student_timesteps[s + 1] if s + 1 < len(student_timesteps) else torch.tensor(0.0)
        
        idx_current = torch.argmin((teacher_timesteps - student_t).abs()).item()
        if s + 1 < len(student_timesteps):
            idx_next = torch.argmin((teacher_timesteps - student_t_next).abs()).item()
            idx_next = min(idx_next + 1, len(teacher_trajectory) - 1)
        else:
            idx_next = len(teacher_trajectory) - 1
        
        student_input = teacher_trajectory[idx_current]
        target_latent = teacher_trajectory[idx_next]
        
        sigma_current = student_sigmas[s]
        sigma_next = student_sigmas[s + 1] if s + 1 < len(student_sigmas) else torch.tensor(0.0)
        delta_sigma = sigma_next - sigma_current
        
        if abs(delta_sigma.item()) < 1e-8:
            print(f"  步骤 {s}: delta_sigma 太小，跳过")
            continue
        
        target_velocity = (target_latent - student_input) / delta_sigma
        
        student_output = scheduler.step(target_velocity, student_t, student_input, return_dict=False)[0]
        
        diff = (student_output - target_latent).abs().mean().item()
        total_diff += diff
        
        status = "PASS" if diff < 1e-6 else "FAIL"
        if diff >= 1e-6:
            all_passed = False
        
        print(f"  步骤 {s}: idx [{idx_current}]->[{idx_next}], delta_sigma={delta_sigma.item():.4f}, diff={diff:.2e} [{status}]")
    
    scheduler.set_timesteps(student_steps, device='cpu', mu=mu)
    current = noise.clone()
    
    for s, t in enumerate(scheduler.timesteps):
        student_t_next = scheduler.timesteps[s + 1] if s + 1 < len(scheduler.timesteps) else torch.tensor(0.0)
        
        idx_current = torch.argmin((teacher_timesteps - t).abs()).item()
        if s + 1 < len(scheduler.timesteps):
            idx_next = torch.argmin((teacher_timesteps - student_t_next).abs()).item()
            idx_next = min(idx_next + 1, len(teacher_trajectory) - 1)
        else:
            idx_next = len(teacher_trajectory) - 1
        
        target_latent = teacher_trajectory[idx_next]
        
        sigma_current = scheduler.sigmas[s]
        sigma_next = scheduler.sigmas[s + 1] if s + 1 < len(scheduler.sigmas) else torch.tensor(0.0)
        delta_sigma = sigma_next - sigma_current
        
        if abs(delta_sigma.item()) < 1e-8:
            continue
        
        target_velocity = (target_latent - current) / delta_sigma
        current = scheduler.step(target_velocity, t, current, return_dict=False)[0]
    
    final_diff = (current - teacher_trajectory[-1]).abs().mean().item()
    
    print(f"\n  完整推理后与教师最终结果差异: {final_diff:.2e}")
    
    if all_passed and final_diff < 1e-5:
        print(f"  结果: 全部通过!")
        return True
    else:
        print(f"  结果: 存在问题!")
        return False

def main():
    print("测试不同学生步数的兼容性")
    print("="*70)
    
    teacher_steps = 50
    test_cases = [4, 5, 6, 7, 8, 10, 12, 15, 20]
    
    results = {}
    for student_steps in test_cases:
        passed = test_student_steps(teacher_steps, student_steps)
        results[student_steps] = passed
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    for steps, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  学生 {steps} 步: [{status}]")
    
    print("\n" + "="*70)
    print("测试边界情况")
    print("="*70)
    
    test_student_steps(30, 4)
    test_student_steps(100, 8)
    test_student_steps(20, 10)

if __name__ == "__main__":
    main()
