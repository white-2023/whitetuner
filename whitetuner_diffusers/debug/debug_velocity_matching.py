import torch
import sys
sys.path.insert(0, 'D:/ai/whitetuner/whitetuner_diffusers')

from diffusers import FlowMatchEulerDiscreteScheduler

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def test_velocity_matching():
    print("=" * 70)
    print("测试速度场匹配逻辑")
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
    print(f"  latent size: {latent_height}x{latent_width}")
    
    batch_size = 1
    channels = 16
    noise = torch.randn(batch_size, channels, latent_height, latent_width)
    
    print("\n" + "=" * 70)
    print("步骤 1: 教师推理，生成完整轨迹")
    print("=" * 70)
    
    scheduler.set_timesteps(teacher_steps, device='cpu', mu=mu)
    teacher_timesteps = scheduler.timesteps
    teacher_sigmas = scheduler.sigmas.clone()
    
    print(f"\n教师 timesteps 长度: {len(teacher_timesteps)}")
    print(f"教师 sigmas 长度: {len(teacher_sigmas)}")
    print(f"教师 sigmas 范围: [{teacher_sigmas[0].item():.4f}, ..., {teacher_sigmas[-1].item():.4f}]")
    
    teacher_trajectory = [noise.clone()]
    
    current = noise.clone()
    for i, t in enumerate(teacher_timesteps):
        fake_velocity = -current * 0.5
        current = scheduler.step(fake_velocity, t, current, return_dict=False)[0]
        teacher_trajectory.append(current.clone())
    
    print(f"\n教师轨迹长度: {len(teacher_trajectory)}")
    print(f"  trajectory[0] (noise) norm: {teacher_trajectory[0].norm().item():.4f}")
    print(f"  trajectory[-1] (final) norm: {teacher_trajectory[-1].norm().item():.4f}")
    
    print("\n" + "=" * 70)
    print("步骤 2: 计算学生采样点")
    print("=" * 70)
    
    sample_indices_teacher = []
    for s in range(student_steps + 1):
        idx = int(s * teacher_steps / student_steps)
        idx = min(idx, teacher_steps)
        sample_indices_teacher.append(idx)
    
    print(f"\n学生 {student_steps} 步对应的教师轨迹索引: {sample_indices_teacher}")
    print(f"  (教师 {teacher_steps} 步 / 学生 {student_steps} 步 = 每 {teacher_steps/student_steps:.1f} 步采样)")
    
    for s in range(student_steps + 1):
        idx = sample_indices_teacher[s]
        sigma = teacher_sigmas[idx].item()
        print(f"  学生点 {s}: 教师轨迹索引 {idx}, sigma = {sigma:.4f}")
    
    print("\n" + "=" * 70)
    print("步骤 3: 计算每个学生步骤的目标速度场")
    print("=" * 70)
    
    scheduler.set_timesteps(student_steps, device='cpu', mu=mu)
    student_timesteps = scheduler.timesteps
    student_sigmas = scheduler.sigmas.clone()
    
    print(f"\n学生 timesteps: {student_timesteps.tolist()}")
    print(f"学生 sigmas: {student_sigmas.tolist()}")
    
    for s in range(student_steps):
        start_idx = sample_indices_teacher[s]
        end_idx = sample_indices_teacher[s + 1]
        
        student_input = teacher_trajectory[start_idx]
        target_latent = teacher_trajectory[end_idx]
        
        sigma_start = teacher_sigmas[start_idx].item()
        sigma_end = teacher_sigmas[end_idx].item()
        delta_sigma = sigma_end - sigma_start
        
        target_velocity = (target_latent - student_input) / delta_sigma
        
        student_t = student_timesteps[s]
        student_sigma = student_sigmas[s].item()
        student_sigma_next = student_sigmas[s + 1].item()
        student_delta = student_sigma_next - student_sigma
        
        print(f"\n学生步骤 {s}:")
        print(f"  教师轨迹: [{start_idx}] -> [{end_idx}]")
        print(f"  教师 sigma: {sigma_start:.4f} -> {sigma_end:.4f}, delta = {delta_sigma:.4f}")
        print(f"  学生 timestep: {student_t.item():.2f}")
        print(f"  学生 sigma: {student_sigma:.4f} -> {student_sigma_next:.4f}, delta = {student_delta:.4f}")
        print(f"  target_velocity norm: {target_velocity.norm().item():.4f}")
        print(f"  target_velocity mean: {target_velocity.mean().item():.6f}")
    
    print("\n" + "=" * 70)
    print("步骤 4: 验证学生用 target_velocity 能否到达目标")
    print("=" * 70)
    
    scheduler.set_timesteps(student_steps, device='cpu', mu=mu)
    
    for s in range(student_steps):
        start_idx = sample_indices_teacher[s]
        end_idx = sample_indices_teacher[s + 1]
        
        student_input = teacher_trajectory[start_idx].clone()
        target_latent = teacher_trajectory[end_idx]
        
        sigma_start = teacher_sigmas[start_idx].item()
        sigma_end = teacher_sigmas[end_idx].item()
        delta_sigma = sigma_end - sigma_start
        
        target_velocity = (target_latent - student_input) / delta_sigma
        
        t = student_timesteps[s]
        student_output = scheduler.step(target_velocity, t, student_input, return_dict=False)[0]
        
        student_delta = student_sigmas[s + 1].item() - student_sigmas[s].item()
        expected_output = student_input + target_velocity * student_delta
        
        diff_to_expected = (student_output - expected_output).abs().mean().item()
        diff_to_target = (student_output - target_latent).abs().mean().item()
        
        print(f"\n学生步骤 {s}:")
        print(f"  scheduler.step 后与 expected 差异: {diff_to_expected:.10f} (应该接近 0)")
        print(f"  scheduler.step 后与教师目标差异: {diff_to_target:.6f}")
        print(f"  教师 delta_sigma: {delta_sigma:.4f}, 学生 delta_sigma: {student_delta:.4f}")
        
        if abs(delta_sigma - student_delta) > 0.01:
            print(f"  注意: delta_sigma 不匹配! 这是正常的，因为教师和学生步数不同")

if __name__ == "__main__":
    test_velocity_matching()
