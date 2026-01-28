"""
调试 Block Swap 模式 vs 普通模式的训练差异
对比：
1. 相同输入的 forward 输出是否一致
2. 相同输入的 backward 梯度是否一致
3. 优化器更新后的参数是否一致
4. Loss 值是否一致
"""

import os
import sys
import gc
import copy
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compare_tensors(t1, t2, name="tensor", rtol=1e-4, atol=1e-4):
    """比较两个 tensor 是否相近"""
    if t1 is None and t2 is None:
        return True, f"{name}: 都为 None"
    if t1 is None or t2 is None:
        return False, f"{name}: 一个为 None"
    
    t1 = t1.float()
    t2 = t2.float()
    
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    return is_close, f"{name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, close={is_close}"


def test_forward_consistency(model_path: str):
    """测试 forward 输出一致性"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("\n" + "=" * 70)
    print("测试 1: Forward 输出一致性")
    print("=" * 70)
    
    # 固定随机种子
    torch.manual_seed(42)
    
    # 加载模型
    print("\n>>> 加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device="cpu")
    
    # 保存初始权重
    initial_state = {k: v.clone() for k, v in transformer.state_dict().items()}
    
    # 准备输入
    num_attention_heads = transformer.config.get("num_attention_heads", 48)
    attention_head_dim = transformer.config.get("attention_head_dim", 128)
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64
    txt_seq_len = 32
    
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim, dtype=dtype)
    timestep = torch.tensor([0.5], dtype=dtype)
    
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, dtype=dtype)
    for i in range(img_seq_len):
        h = i // width
        w = i % width
        img_ids[i] = torch.tensor([0, 0, h, w], dtype=dtype)
    txt_ids = torch.zeros(txt_seq_len, 4, dtype=dtype)
    
    results = {}
    
    for mode_name, use_block_swap in [("普通模式", False), ("Block Swap", True)]:
        print(f"\n>>> 测试 {mode_name}...")
        cleanup()
        
        # 重新加载模型
        transformer.load_state_dict(initial_state)
        transformer.to(device)
        
        blocks_to_swap = transformer.num_single_blocks - 2 if use_block_swap else 0
        
        if use_block_swap:
            transformer.enable_block_swap(
                blocks_to_swap=blocks_to_swap,
                device=device,
                supports_backward=True,
                use_pinned_memory=False,
            )
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        
        transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_block_swap)
        
        # Forward
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = transformer(
                    hidden_states=hidden_states.to(device),
                    encoder_hidden_states=encoder_hidden_states.to(device),
                    timestep=timestep.to(device),
                    img_ids=img_ids.to(device),
                    txt_ids=txt_ids.to(device),
                    guidance=None,
                )
        
        results[mode_name] = output['sample'].cpu().clone()
        print(f"  输出: shape={output['sample'].shape}, mean={output['sample'].mean():.6f}, std={output['sample'].std():.6f}")
        
        if use_block_swap:
            transformer.cleanup_offloader()
    
    # 比较结果
    is_close, msg = compare_tensors(results["普通模式"], results["Block Swap"], "Forward 输出")
    print(f"\n>>> 比较结果: {msg}")
    
    return is_close


def test_backward_consistency(model_path: str):
    """测试 backward 梯度一致性"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("\n" + "=" * 70)
    print("测试 2: Backward 梯度一致性")
    print("=" * 70)
    
    # 固定随机种子
    torch.manual_seed(42)
    
    # 加载模型
    print("\n>>> 加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device="cpu")
    
    # 保存初始权重
    initial_state = {k: v.clone() for k, v in transformer.state_dict().items()}
    
    # 准备输入
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64
    txt_seq_len = 32
    
    torch.manual_seed(123)
    hidden_states_base = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim, dtype=dtype)
    timestep = torch.tensor([0.5], dtype=dtype)
    
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, dtype=dtype)
    for i in range(img_seq_len):
        h = i // width
        w = i % width
        img_ids[i] = torch.tensor([0, 0, h, w], dtype=dtype)
    txt_ids = torch.zeros(txt_seq_len, 4, dtype=dtype)
    
    grad_results = {}
    loss_results = {}
    
    for mode_name, use_block_swap in [("普通模式", False), ("Block Swap", True)]:
        print(f"\n>>> 测试 {mode_name}...")
        cleanup()
        
        # 重新加载模型
        transformer.load_state_dict(initial_state)
        transformer.to(device)
        transformer.requires_grad_(True)
        
        blocks_to_swap = transformer.num_single_blocks - 2 if use_block_swap else 0
        
        if use_block_swap:
            transformer.enable_block_swap(
                blocks_to_swap=blocks_to_swap,
                device=device,
                supports_backward=True,
                use_pinned_memory=False,
            )
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        
        transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_block_swap)
        
        hidden_states = hidden_states_base.clone().to(device).requires_grad_(True)
        
        # Forward + Backward
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states.to(device),
                timestep=timestep.to(device),
                img_ids=img_ids.to(device),
                txt_ids=txt_ids.to(device),
                guidance=None,
            )
        
        loss = output['sample'].float().mean()
        loss_results[mode_name] = loss.item()
        print(f"  Loss: {loss.item():.6f}")
        
        loss.backward()
        
        # 收集梯度
        grads = {}
        for name, param in transformer.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.cpu().clone()
        
        grad_results[mode_name] = grads
        print(f"  收集了 {len(grads)} 个参数的梯度")
        
        # 打印一些梯度统计
        total_grad_norm = 0
        for name, grad in list(grads.items())[:5]:
            grad_norm = grad.float().norm().item()
            total_grad_norm += grad_norm ** 2
            print(f"    {name}: norm={grad_norm:.6e}")
        
        if use_block_swap:
            transformer.cleanup_offloader()
    
    # 比较 loss
    loss_diff = abs(loss_results["普通模式"] - loss_results["Block Swap"])
    print(f"\n>>> Loss 差异: {loss_diff:.6e}")
    
    # 比较梯度
    print("\n>>> 梯度比较:")
    all_close = True
    grad_diffs = []
    
    common_keys = set(grad_results["普通模式"].keys()) & set(grad_results["Block Swap"].keys())
    print(f"  共同参数数量: {len(common_keys)}")
    
    for name in list(common_keys)[:10]:  # 只检查前 10 个
        g1 = grad_results["普通模式"][name]
        g2 = grad_results["Block Swap"][name]
        is_close, msg = compare_tensors(g1, g2, name)
        if not is_close:
            all_close = False
        grad_diffs.append((name, (g1 - g2).abs().max().item()))
    
    # 排序并打印最大差异
    grad_diffs.sort(key=lambda x: -x[1])
    print("\n  最大梯度差异 (前 5 个):")
    for name, diff in grad_diffs[:5]:
        print(f"    {name}: {diff:.6e}")
    
    return all_close, loss_diff


def test_optimizer_step(model_path: str):
    """测试优化器更新一致性"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    import bitsandbytes as bnb
    import transformers.optimization
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("\n" + "=" * 70)
    print("测试 3: 优化器更新一致性")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print("\n>>> 加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device="cpu")
    
    initial_state = {k: v.clone() for k, v in transformer.state_dict().items()}
    
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64
    txt_seq_len = 32
    
    torch.manual_seed(123)
    hidden_states_base = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim, dtype=dtype)
    timestep = torch.tensor([0.5], dtype=dtype)
    
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, dtype=dtype)
    for i in range(img_seq_len):
        h = i // width
        w = i % width
        img_ids[i] = torch.tensor([0, 0, h, w], dtype=dtype)
    txt_ids = torch.zeros(txt_seq_len, 4, dtype=dtype)
    
    param_results = {}
    
    for mode_name, use_block_swap, use_adafactor in [
        ("AdamW8bit", False, False),
        ("Adafactor (Block Swap)", True, True),
        ("Adafactor (无 Block Swap)", False, True),  # 额外测试：只换优化器
    ]:
        print(f"\n>>> 测试 {mode_name}...")
        cleanup()
        
        transformer.load_state_dict(initial_state)
        transformer.to(device)
        transformer.requires_grad_(True)
        
        blocks_to_swap = transformer.num_single_blocks - 2 if use_block_swap else 0
        
        if use_block_swap:
            transformer.enable_block_swap(
                blocks_to_swap=blocks_to_swap,
                device=device,
                supports_backward=True,
                use_pinned_memory=False,
            )
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        
        transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_block_swap)
        
        # 创建优化器
        trainable_params = [p for p in transformer.parameters() if p.requires_grad]
        lr = 1e-4
        
        if use_adafactor:
            optimizer = transformers.optimization.Adafactor(
                trainable_params,
                lr=lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
            print(f"  使用 Adafactor 优化器")
        else:
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-6,
            )
            print(f"  使用 AdamW8bit 优化器")
        
        # 记录初始参数
        initial_param_sample = {}
        for name, param in list(transformer.named_parameters())[:3]:
            initial_param_sample[name] = param.data.cpu().clone()
        
        hidden_states = hidden_states_base.clone().to(device).requires_grad_(True)
        
        # Forward + Backward
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states.to(device),
                timestep=timestep.to(device),
                img_ids=img_ids.to(device),
                txt_ids=txt_ids.to(device),
                guidance=None,
            )
        
        loss = output['sample'].float().mean()
        print(f"  Loss: {loss.item():.6f}")
        
        loss.backward()
        
        # 优化器 step
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录更新后的参数
        updated_param_sample = {}
        for name, param in list(transformer.named_parameters())[:3]:
            updated_param_sample[name] = param.data.cpu().clone()
        
        # 计算参数变化
        print("  参数变化:")
        for name in initial_param_sample:
            before = initial_param_sample[name]
            after = updated_param_sample[name]
            diff = (after - before).abs()
            print(f"    {name}: max_change={diff.max().item():.6e}, mean_change={diff.mean().item():.6e}")
        
        param_results[mode_name] = updated_param_sample
        
        if use_block_swap:
            transformer.cleanup_offloader()
    
    # 比较结果
    print("\n>>> 参数更新比较:")
    for name in list(param_results["AdamW8bit"].keys()):
        p_adam = param_results["AdamW8bit"][name]
        p_adafactor = param_results["Adafactor (Block Swap)"][name]
        p_adafactor_no_swap = param_results["Adafactor (无 Block Swap)"][name]
        
        diff_adafactor = (p_adam - p_adafactor).abs().max().item()
        diff_adafactor_no_swap = (p_adam - p_adafactor_no_swap).abs().max().item()
        diff_swap_effect = (p_adafactor - p_adafactor_no_swap).abs().max().item()
        
        print(f"  {name}:")
        print(f"    AdamW8bit vs Adafactor(swap): {diff_adafactor:.6e}")
        print(f"    AdamW8bit vs Adafactor(no swap): {diff_adafactor_no_swap:.6e}")
        print(f"    Block Swap 影响: {diff_swap_effect:.6e}")


def test_loss_curve(model_path: str, num_steps: int = 10):
    """测试多步训练的 loss 曲线"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    import bitsandbytes as bnb
    import transformers.optimization
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("\n" + "=" * 70)
    print("测试 4: Loss 曲线对比")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print("\n>>> 加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device="cpu")
    
    initial_state = {k: v.clone() for k, v in transformer.state_dict().items()}
    
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64
    txt_seq_len = 32
    
    # 预生成多步输入
    inputs = []
    for step in range(num_steps):
        torch.manual_seed(1000 + step)
        inputs.append({
            'hidden_states': torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype),
            'encoder_hidden_states': torch.randn(batch_size, txt_seq_len, joint_attention_dim, dtype=dtype),
            'timestep': torch.tensor([0.5], dtype=dtype),
        })
    
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, dtype=dtype)
    for i in range(img_seq_len):
        h = i // width
        w = i % width
        img_ids[i] = torch.tensor([0, 0, h, w], dtype=dtype)
    txt_ids = torch.zeros(txt_seq_len, 4, dtype=dtype)
    
    loss_curves = {}
    
    for mode_name, use_block_swap, use_adafactor in [
        ("AdamW8bit", False, False),
        ("Adafactor (Block Swap)", True, True),
    ]:
        print(f"\n>>> 测试 {mode_name}...")
        cleanup()
        
        transformer.load_state_dict(initial_state)
        transformer.to(device)
        transformer.requires_grad_(True)
        
        blocks_to_swap = transformer.num_single_blocks - 2 if use_block_swap else 0
        
        if use_block_swap:
            transformer.enable_block_swap(
                blocks_to_swap=blocks_to_swap,
                device=device,
                supports_backward=True,
                use_pinned_memory=False,
            )
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        
        transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_block_swap)
        
        trainable_params = [p for p in transformer.parameters() if p.requires_grad]
        lr = 1e-4
        
        if use_adafactor:
            optimizer = transformers.optimization.Adafactor(
                trainable_params,
                lr=lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        else:
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-6,
            )
        
        losses = []
        
        for step in range(num_steps):
            if use_block_swap:
                transformer.prepare_block_swap_before_forward()
            
            inp = inputs[step]
            hidden_states = inp['hidden_states'].to(device).requires_grad_(True)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=inp['encoder_hidden_states'].to(device),
                    timestep=inp['timestep'].to(device),
                    img_ids=img_ids.to(device),
                    txt_ids=txt_ids.to(device),
                    guidance=None,
                )
            
            loss = output['sample'].float().mean()
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"  Step {step + 1}: loss={loss.item():.6f}")
        
        loss_curves[mode_name] = losses
        
        if use_block_swap:
            transformer.cleanup_offloader()
    
    # 打印对比
    print("\n>>> Loss 曲线对比:")
    print(f"{'Step':<6} {'AdamW8bit':<15} {'Adafactor(swap)':<15} {'差异':<15}")
    print("-" * 50)
    for step in range(num_steps):
        l1 = loss_curves["AdamW8bit"][step]
        l2 = loss_curves["Adafactor (Block Swap)"][step]
        diff = l2 - l1
        print(f"{step + 1:<6} {l1:<15.6f} {l2:<15.6f} {diff:<15.6f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="调试 Block Swap 训练差异")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--test", type=str, default="all", 
                       choices=["forward", "backward", "optimizer", "curve", "all"])
    parser.add_argument("--num_steps", type=int, default=10, help="Loss 曲线测试步数")
    args = parser.parse_args()
    
    if args.test in ["forward", "all"]:
        test_forward_consistency(args.model_path)
    
    if args.test in ["backward", "all"]:
        test_backward_consistency(args.model_path)
    
    if args.test in ["optimizer", "all"]:
        test_optimizer_step(args.model_path)
    
    if args.test in ["curve", "all"]:
        test_loss_curve(args.model_path, args.num_steps)


if __name__ == "__main__":
    main()

