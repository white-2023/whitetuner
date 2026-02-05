import torch
import sys
sys.path.insert(0, 'D:/ai/whitetuner/whitetuner_diffusers')

from diffusers import FlowMatchEulerDiscreteScheduler

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def test_velocity_matching_v2():
    print("=" * 70)
    print("测试速度场匹配逻辑 V2 (使用学生的 sigma)")
    print("=" * 70)
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
    )
    
    latent_height = 128
    latent_width = 128
    image_seq_len = (latent_height // 2) * (latent_width // 2)
    mu = calculate_shift(image_seq_len)
    
    teacher_steps = 74
    student_steps = 4
    
    print(f"\n配置:")
    print(f"  教师步数: {teacher_steps}")
    print(f"  学生步数: {student_steps}")
    
    batch_size = 1
    channels = 16
    noise = torch.randn(batch_size, channels, latent_height, latent_width)
    
    print("\n" + "=" * 70)
    print("步骤 1: 教师推理，生成完整轨迹")
    print("=" * 70)
    
    scheduler.set_timesteps(teacher_steps, device='cpu', mu=mu)
    teacher_timesteps = scheduler.timesteps.clone()
    
    teacher_trajectory = [noise.clone()]
    
    current = noise.clone()
    for i, t in enumerate(teacher_timesteps):
        fake_velocity = -current * 0.5
        current = scheduler.step(fake_velocity, t, current, return_dict=False)[0]
        teacher_trajectory.append(current.clone())
    
    print(f"教师轨迹长度: {len(teacher_trajectory)}")
    print(f"教师 timesteps 范围: [{teacher_timesteps[0].item():.2f}, ..., {teacher_timesteps[-1].item():.2f}]")
    
    print("\n" + "=" * 70)
    print("步骤 2: 学生 scheduler 设置")
    print("=" * 70)
    
    scheduler.set_timesteps(student_steps, device='cpu', mu=mu)
    student_timesteps = scheduler.timesteps
    student_sigmas = scheduler.sigmas
    
    print(f"学生 timesteps: {student_timesteps.tolist()}")
    print(f"学生 sigmas: {student_sigmas.tolist()}")
    
    print("\n" + "=" * 70)
    print("步骤 3: 按照新逻辑计算采样点和 target_velocity")
    print("=" * 70)
    
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
        
        target_velocity = (target_latent - student_input) / delta_sigma
        
        print(f"\n学生步骤 {s}:")
        print(f"  学生 timestep: {student_t.item():.2f} -> {student_t_next.item():.2f}")
        print(f"  教师轨迹索引: [{idx_current}] -> [{idx_next}]")
        print(f"  教师 timestep: {teacher_timesteps[idx_current].item():.2f} (最近) -> trajectory[{idx_next}]")
        print(f"  学生 sigma: {sigma_current.item():.4f} -> {sigma_next.item():.4f}")
        print(f"  delta_sigma: {delta_sigma.item():.4f}")
        print(f"  target_velocity norm: {target_velocity.norm().item():.4f}")
    
    print("\n" + "=" * 70)
    print("步骤 4: 验证学生用 target_velocity 能否到达目标")
    print("=" * 70)
    
    scheduler.set_timesteps(student_steps, device='cpu', mu=mu)
    student_timesteps = scheduler.timesteps
    student_sigmas = scheduler.sigmas
    
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
        
        student_input = teacher_trajectory[idx_current].clone()
        target_latent = teacher_trajectory[idx_next]
        
        sigma_current = student_sigmas[s]
        sigma_next = student_sigmas[s + 1] if s + 1 < len(student_sigmas) else torch.tensor(0.0)
        delta_sigma = sigma_next - sigma_current
        
        target_velocity = (target_latent - student_input) / delta_sigma
        
        student_output = scheduler.step(target_velocity, student_t, student_input, return_dict=False)[0]
        
        diff_to_target = (student_output - target_latent).abs().mean().item()
        total_diff += diff_to_target
        
        print(f"\n学生步骤 {s}:")
        print(f"  scheduler.step 后与教师目标差异: {diff_to_target:.10f}")
        
        if diff_to_target < 1e-6:
            print(f"  验证通过! 学生能正确到达目标")
        else:
            print(f"  警告: 差异较大!")
    
    print(f"\n总差异: {total_diff:.10f}")
    if total_diff < 1e-5:
        print("所有步骤验证通过!")
    
    print("\n" + "=" * 70)
    print("步骤 5: 完整推理测试")
    print("=" * 70)
    
    scheduler.set_timesteps(student_steps, device='cpu', mu=mu)
    
    current = noise.clone()
    print(f"\n起点 (noise) norm: {current.norm().item():.4f}")
    
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
        
        target_velocity = (target_latent - current) / delta_sigma
        
        current = scheduler.step(target_velocity, t, current, return_dict=False)[0]
        print(f"  步骤 {s}: current norm = {current.norm().item():.4f}")
    
    final_diff = (current - teacher_trajectory[-1]).abs().mean().item()
    print(f"\n最终结果与教师最终结果差异: {final_diff:.10f}")

if __name__ == "__main__":
    test_velocity_matching_v2()
