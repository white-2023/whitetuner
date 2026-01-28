"""
调试 Block Swap 22 vs 23 的区别
分析为什么 23 个 block swap 会报错
"""

import torch
import sys
sys.path.insert(0, "D:/ai/whitetuner/whitetuner_diffusers")


def analyze_block_swap_logic(num_blocks: int, blocks_to_swap: int):
    """分析 block swap 的逻辑，打印每个 block 的 hook 行为"""
    print(f"\n{'=' * 70}")
    print(f"分析 Block Swap 逻辑: num_blocks={num_blocks}, blocks_to_swap={blocks_to_swap}")
    print(f"{'=' * 70}")
    
    # 初始状态
    blocks_on_gpu = num_blocks - blocks_to_swap
    print(f"\n[初始状态]")
    print(f"  GPU 上的 blocks: 0 ~ {blocks_on_gpu - 1} (共 {blocks_on_gpu} 个)")
    print(f"  CPU 上的 blocks: {blocks_on_gpu} ~ {num_blocks - 1} (共 {blocks_to_swap} 个)")
    
    # Forward 阶段的交换
    print(f"\n[Forward 阶段] - 每个 block 执行后触发的交换")
    forward_swaps = []
    for block_idx in range(num_blocks):
        # submit_move_blocks_forward 的逻辑 (supports_backward=True)
        if block_idx >= blocks_to_swap:
            continue
        block_idx_to_cpu = block_idx
        block_idx_to_cuda = num_blocks - blocks_to_swap + block_idx
        forward_swaps.append((block_idx, block_idx_to_cpu, block_idx_to_cuda))
        print(f"  Block {block_idx} 执行后: block {block_idx_to_cpu} → CPU, block {block_idx_to_cuda} → GPU")
    
    # Forward 结束后的状态
    cpu_blocks = set(range(blocks_to_swap))  # 被交换出去的
    gpu_blocks = set(range(blocks_to_swap, num_blocks))  # 被交换回来的 + 从未移动的
    print(f"\n[Forward 结束后状态]")
    print(f"  CPU 上的 blocks: {sorted(cpu_blocks)}")
    print(f"  GPU 上的 blocks: {sorted(gpu_blocks)}")
    
    # Backward 阶段的 hook 分析（使用修复后的逻辑）
    print(f"\n[Backward 阶段] - 每个 block 的 backward hook 行为")
    print(f"  (backward 顺序: {num_blocks - 1} → 0)")
    
    backward_hooks = []
    num_gpu_blocks = num_blocks - blocks_to_swap
    
    for block_index in range(num_blocks):
        num_blocks_propagated = num_blocks - block_index - 1
        
        # 修复边界情况：当 GPU 上只有 1 个 block 时，使用简化逻辑
        if num_gpu_blocks == 1:
            if block_index > 0:
                swapping = True
                block_idx_to_cpu = block_index
                block_idx_to_cuda = block_index - 1
            else:
                swapping = False
                block_idx_to_cpu = 0
                block_idx_to_cuda = 0
        else:
            swapping = num_blocks_propagated > 0 and num_blocks_propagated <= blocks_to_swap
            block_idx_to_cpu = num_blocks - num_blocks_propagated
            block_idx_to_cuda = blocks_to_swap - num_blocks_propagated
        
        waiting = block_index > 0 and block_index <= blocks_to_swap
        block_idx_to_wait = block_index - 1
        
        if not swapping and not waiting:
            continue
        
        backward_hooks.append({
            'block_index': block_index,
            'swapping': swapping,
            'waiting': waiting,
            'swap_to_cpu': block_idx_to_cpu if swapping else None,
            'swap_to_cuda': block_idx_to_cuda if swapping else None,
            'wait_for': block_idx_to_wait if waiting else None,
        })
    
    # 按 backward 执行顺序打印
    for hook in sorted(backward_hooks, key=lambda x: -x['block_index']):
        print(f"  Block {hook['block_index']:2d}: ", end="")
        if hook['swapping']:
            print(f"swap(block {hook['swap_to_cpu']} → CPU, block {hook['swap_to_cuda']} → GPU) ", end="")
        if hook['waiting']:
            print(f"wait(block {hook['wait_for']})", end="")
        print()
    
    # 找出问题：哪些 block 在 backward 时可能权重不在 GPU
    print(f"\n[问题分析] - 检查 backward 时权重位置")
    
    # 模拟 backward 过程
    current_cpu = set(cpu_blocks)
    current_gpu = set(gpu_blocks)
    pending_moves = {}  # block_idx_to_cuda -> (from_cpu_block)
    
    print(f"  Backward 开始时: GPU={sorted(current_gpu)}, CPU={sorted(current_cpu)}")
    
    for block_index in range(num_blocks - 1, -1, -1):
        # 检查当前 block 是否需要 recompute（gradient checkpointing）
        if block_index not in current_gpu:
            print(f"\n  ⚠️  Block {block_index} 需要 recompute 但权重在 CPU!")
            print(f"      这会导致 RuntimeError: Expected all tensors to be on the same device")
        
        # 找到这个 block 的 hook 信息
        hook = None
        for h in backward_hooks:
            if h['block_index'] == block_index:
                hook = h
                break
        
        if hook is None:
            continue
        
        # 执行 swap 操作（提交到后台线程）
        if hook['swapping']:
            pending_moves[hook['swap_to_cuda']] = hook['swap_to_cpu']
        
        # 执行 wait 操作
        if hook['waiting']:
            wait_block = hook['wait_for']
            if wait_block in pending_moves:
                # 完成移动
                from_cpu_block = pending_moves.pop(wait_block)
                current_cpu.add(from_cpu_block)
                current_gpu.discard(from_cpu_block)
                current_gpu.add(wait_block)
                current_cpu.discard(wait_block)
    
    return forward_swaps, backward_hooks


def test_block_swap_with_model(blocks_to_swap: int, model_path: str = None):
    """实际测试 block swap"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    if model_path is None:
        model_path = "F:/models/FLUX.2-klein-base-9B"
    
    print(f"\n{'=' * 70}")
    print(f"实际测试 Block Swap: blocks_to_swap={blocks_to_swap}")
    print(f"{'=' * 70}")
    
    print("\n加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device="cpu")
    
    num_single_blocks = transformer.num_single_blocks
    print(f"模型有 {num_single_blocks} 个 single blocks")
    
    # 启用 block swap
    print(f"\n启用 block swap: {blocks_to_swap} blocks...")
    try:
        transformer.enable_block_swap(
            blocks_to_swap=blocks_to_swap,
            device=device,
            supports_backward=True,
            use_pinned_memory=False,
        )
    except AssertionError as e:
        print(f"❌ 启用 block swap 失败: {e}")
        return False
    
    # 移动模型
    transformer.move_to_device_except_swap_blocks(device)
    transformer.prepare_block_swap_before_forward()
    
    # 启用 gradient checkpointing
    print("启用 gradient checkpointing (with activation cpu offloading)...")
    transformer.enable_gradient_checkpointing(activation_cpu_offloading=True)
    
    # 准备输入
    num_attention_heads = transformer.config.get("num_attention_heads", 48)
    attention_head_dim = transformer.config.get("attention_head_dim", 128)
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64  # 减小以加速测试
    txt_seq_len = 32
    
    hidden_states = torch.randn(batch_size, img_seq_len, in_channels, device=device, dtype=dtype, requires_grad=True)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim, device=device, dtype=dtype)
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    
    img_ids = torch.zeros(img_seq_len, 4, device=device, dtype=dtype)
    for i in range(img_seq_len):
        h = i // 8
        w = i % 8
        img_ids[i] = torch.tensor([0, 0, h, w], dtype=dtype)
    
    txt_ids = torch.zeros(txt_seq_len, 4, device=device, dtype=dtype)
    
    # 测试 forward + backward
    print("\n执行 forward + backward...")
    try:
        # Forward
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
            )
        print(f"  Forward 成功: output shape={output['sample'].shape}")
        
        # Backward - 不能在这里调用 prepare_block_swap_before_forward！
        # 因为 backward 需要在 forward 交换后的状态下执行
        loss = output['sample'].mean()
        loss.backward()
        print(f"  Backward 成功: grad norm={hidden_states.grad.norm():.4f}")
    except Exception as e:
        print(f"❌ Forward/Backward 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 清理
    transformer.cleanup_offloader()
    del transformer
    torch.cuda.empty_cache()
    
    return True


def simulate_backward_execution(num_blocks: int, blocks_to_swap: int):
    """模拟 backward 执行过程，追踪每个 block 的权重位置"""
    print(f"\n{'=' * 70}")
    print(f"模拟 Backward 执行: num_blocks={num_blocks}, blocks_to_swap={blocks_to_swap}")
    print(f"{'=' * 70}")
    
    # Forward 结束后的初始状态
    cpu_blocks = set(range(blocks_to_swap))
    gpu_blocks = set(range(blocks_to_swap, num_blocks))
    
    # 收集所有 hook 信息（使用修复后的逻辑）
    hooks = {}
    num_gpu_blocks = num_blocks - blocks_to_swap
    
    for block_index in range(num_blocks):
        num_blocks_propagated = num_blocks - block_index - 1
        
        # 修复边界情况：当 GPU 上只有 1 个 block 时，使用简化逻辑
        if num_gpu_blocks == 1:
            if block_index > 0:
                swapping = True
                swap_to_cpu = block_index
                swap_to_cuda = block_index - 1
            else:
                swapping = False
                swap_to_cpu = None
                swap_to_cuda = None
        else:
            swapping = num_blocks_propagated > 0 and num_blocks_propagated <= blocks_to_swap
            swap_to_cpu = num_blocks - num_blocks_propagated if swapping else None
            swap_to_cuda = blocks_to_swap - num_blocks_propagated if swapping else None
        
        waiting = block_index > 0 and block_index <= blocks_to_swap
        
        if swapping or waiting:
            hooks[block_index] = {
                'swapping': swapping,
                'waiting': waiting,
                'swap_to_cpu': swap_to_cpu,
                'swap_to_cuda': swap_to_cuda,
                'wait_for': block_index - 1 if waiting else None,
            }
    
    # 模拟 futures 字典
    futures = {}  # block_idx_to_cuda -> True (表示已提交)
    
    print("\n[Backward 执行顺序] (从 block 47 → 0)")
    print("-" * 70)
    
    errors = []
    
    # Backward 从 47 → 0
    for block_index in range(num_blocks - 1, -1, -1):
        # Step 1: Recompute (需要权重在 GPU)
        if block_index not in gpu_blocks:
            error_msg = f"Block {block_index} recompute 失败: 权重在 CPU!"
            errors.append((block_index, "recompute", error_msg))
            print(f"  ❌ Block {block_index}: RECOMPUTE 失败 - 权重在 CPU!")
            print(f"     当前 GPU blocks: {sorted(gpu_blocks)}")
            print(f"     当前 CPU blocks: {sorted(cpu_blocks)}")
            continue
        
        # Step 2: Backward (假设成功)
        
        # Step 3: Backward Hook
        if block_index in hooks:
            hook = hooks[block_index]
            actions = []
            
            # 处理 swapping
            if hook['swapping']:
                to_cpu = hook['swap_to_cpu']
                to_cuda = hook['swap_to_cuda']
                futures[to_cuda] = (to_cpu, to_cuda)  # 记录待处理的移动
                actions.append(f"提交 block {to_cpu}→CPU, block {to_cuda}→GPU")
            
            # 处理 waiting
            if hook['waiting']:
                wait_block = hook['wait_for']
                if wait_block in futures:
                    # 完成移动
                    from_cpu, to_cuda = futures.pop(wait_block)
                    cpu_blocks.add(from_cpu)
                    gpu_blocks.discard(from_cpu)
                    gpu_blocks.add(to_cuda)
                    cpu_blocks.discard(to_cuda)
                    actions.append(f"等待 block {wait_block} 完成 (block {to_cuda} → GPU)")
                else:
                    actions.append(f"等待 block {wait_block} 但 futures 中不存在!")
                    errors.append((block_index, "wait", f"futures[{wait_block}] 不存在"))
            
            if actions:
                print(f"  Block {block_index:2d}: recompute ✓ → backward ✓ → hook: {', '.join(actions)}")
        else:
            print(f"  Block {block_index:2d}: recompute ✓ → backward ✓ → (无 hook)")
    
    print()
    if errors:
        print("=" * 70)
        print("⚠️  发现问题:")
        print("=" * 70)
        for block_idx, stage, msg in errors:
            print(f"  Block {block_idx} ({stage}): {msg}")
    else:
        print("✅ 模拟执行完成，无错误")
    
    return errors


def analyze_recompute_issue():
    """分析 gradient checkpointing recompute 时的问题"""
    print("\n" + "=" * 70)
    print("问题分析: Gradient Checkpointing + Block Swap")
    print("=" * 70)
    
    print("""
问题根源:
---------
1. Gradient checkpointing 在 backward 时需要 recompute forward
2. Backward hook 是在 block 的 backward 完成后才触发的
3. Block N 的 recompute 发生在 Block (N+1) 的 hook 执行之后

关键时序问题 (blocks_to_swap=23):
---------------------------------
1. Forward 结束后: Block 22 在 CPU, Block 23 在 GPU
2. Backward 到 Block 46 时: Hook 提交 "Block 22 → GPU" 到后台线程
3. Backward 到 Block 23 时: Hook 执行 wait(Block 22)，确保 Block 22 在 GPU
4. 然后 Block 22 的 recompute 开始执行

问题在于:
---------
- 虽然 Block 23 的 hook 会 wait(Block 22)
- 但 Block 22 的移动是在 Block 46 的 hook 中异步提交的
- 如果 CUDA stream 同步有问题，Block 22 的权重可能还没完全到 GPU

可能的修复方案:
---------------
1. 在 block.forward() 开始时检查并等待权重到位
2. 修改 backward hook 确保提前提交移动操作
3. 增加额外的同步点
""")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="调试 Block Swap 问题")
    parser.add_argument("--mode", type=str, default="analyze", choices=["analyze", "simulate", "test", "all"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--blocks", type=int, nargs='+', default=[22, 23])
    parser.add_argument("--num_blocks", type=int, default=24, help="single blocks 数量 (FLUX.2 Klein 9B=48, 小模型=24)")
    args = parser.parse_args()
    
    num_single_blocks = args.num_blocks  # 从参数获取
    
    if args.mode in ["analyze", "all"]:
        # 分析逻辑差异
        for blocks_to_swap in args.blocks:
            analyze_block_swap_logic(num_single_blocks, blocks_to_swap)
        
        analyze_recompute_issue()
    
    if args.mode in ["simulate", "all"]:
        # 模拟 backward 执行
        print("\n" + "#" * 70)
        print("# 模拟 Backward 执行过程")
        print("#" * 70)
        for blocks_to_swap in args.blocks:
            errors = simulate_backward_execution(num_single_blocks, blocks_to_swap)
            if errors:
                print(f"\n💡 blocks_to_swap={blocks_to_swap} 会出现问题!")
            else:
                print(f"\n✅ blocks_to_swap={blocks_to_swap} 理论上应该正常")
    
    if args.mode in ["test", "all"]:
        # 实际测试
        for blocks_to_swap in args.blocks:
            print(f"\n\n{'#' * 70}")
            print(f"# 测试 blocks_to_swap = {blocks_to_swap}")
            print(f"{'#' * 70}")
            
            success = test_block_swap_with_model(blocks_to_swap, args.model_path)
            if success:
                print(f"\n✅ blocks_to_swap={blocks_to_swap} 测试通过")
            else:
                print(f"\n❌ blocks_to_swap={blocks_to_swap} 测试失败")


if __name__ == "__main__":
    main()

