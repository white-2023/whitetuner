"""
调试 Activation CPU Offloading 的显存效果
测试启用/禁用 activation_cpu_offloading 时的显存占用差异
"""

import os
import sys
import gc
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_gpu_memory_mb():
    """获取当前 GPU 显存使用量 (MB)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_memory_reserved_mb():
    """获取 GPU 保留显存 (MB)"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0


def cleanup():
    """清理 GPU 缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_activation_offload(
    model_path: str,
    use_cpu_offload: bool,
    batch_size: int = 1,
    img_seq_len: int = 256,  # 模拟 512x512 图像的 latent
    txt_seq_len: int = 128,
):
    """测试一次 forward + backward，记录显存使用"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print(f"\n{'=' * 70}")
    print(f"测试配置: activation_cpu_offloading={use_cpu_offload}")
    print(f"  batch_size={batch_size}, img_seq_len={img_seq_len}, txt_seq_len={txt_seq_len}")
    print(f"{'=' * 70}")
    
    cleanup()
    mem_start = get_gpu_memory_mb()
    print(f"\n[起始] 显存: {mem_start:.1f} MB")
    
    # 加载模型
    print("\n>>> 加载模型...")
    transformer = load_flux2_transformer_from_diffusers(
        model_path,
        torch_dtype=dtype,
        device="cpu",
    )
    transformer.to(device)
    transformer.requires_grad_(True)
    
    mem_after_load = get_gpu_memory_mb()
    print(f"[加载模型后] 显存: {mem_after_load:.1f} MB (+{mem_after_load - mem_start:.1f} MB)")
    
    # 启用 gradient checkpointing
    print(f"\n>>> 启用 Gradient Checkpointing (activation_cpu_offloading={use_cpu_offload})...")
    transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_cpu_offload)
    
    # 准备输入
    num_attention_heads = transformer.config.get("num_attention_heads", 48)
    attention_head_dim = transformer.config.get("attention_head_dim", 128)
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    print(f"\n>>> 准备输入数据...")
    hidden_states = torch.randn(
        batch_size, img_seq_len, in_channels,
        device=device, dtype=dtype, requires_grad=True
    )
    encoder_hidden_states = torch.randn(
        batch_size, txt_seq_len, joint_attention_dim,
        device=device, dtype=dtype
    )
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    
    # 创建位置 ID
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, device=device, dtype=dtype)
    for i in range(img_seq_len):
        h = i // width
        w = i % width
        img_ids[i] = torch.tensor([0, 0, h, w], dtype=dtype)
    
    txt_ids = torch.zeros(txt_seq_len, 4, device=device, dtype=dtype)
    
    mem_before_forward = get_gpu_memory_mb()
    print(f"[Forward 前] 显存: {mem_before_forward:.1f} MB")
    
    # Forward
    print("\n>>> 执行 Forward...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
        )
    
    torch.cuda.synchronize()
    mem_after_forward = get_gpu_memory_mb()
    mem_reserved_after_forward = get_gpu_memory_reserved_mb()
    print(f"[Forward 后] 显存: {mem_after_forward:.1f} MB (+{mem_after_forward - mem_before_forward:.1f} MB)")
    print(f"            保留: {mem_reserved_after_forward:.1f} MB")
    
    # Backward
    print("\n>>> 执行 Backward...")
    loss = output['sample'].mean()
    loss.backward()
    
    torch.cuda.synchronize()
    mem_after_backward = get_gpu_memory_mb()
    mem_reserved_after_backward = get_gpu_memory_reserved_mb()
    print(f"[Backward 后] 显存: {mem_after_backward:.1f} MB")
    print(f"             保留: {mem_reserved_after_backward:.1f} MB")
    
    # 峰值显存
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    print(f"\n[峰值] 分配: {peak_mem:.1f} MB, 保留: {peak_reserved:.1f} MB")
    
    # 清理
    del transformer, hidden_states, encoder_hidden_states, output, loss
    cleanup()
    
    return {
        'use_cpu_offload': use_cpu_offload,
        'mem_after_load': mem_after_load,
        'mem_after_forward': mem_after_forward,
        'mem_after_backward': mem_after_backward,
        'peak_mem': peak_mem,
        'peak_reserved': peak_reserved,
    }


def test_manual_offload():
    """手动测试 activation offload 的效果"""
    print("\n" + "=" * 70)
    print("手动测试: 对比 activation 保存位置")
    print("=" * 70)
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    cleanup()
    
    # 创建一个简单的模型来测试
    class SimpleBlock(torch.nn.Module):
        def __init__(self, dim=4096):
            super().__init__()
            self.linear1 = torch.nn.Linear(dim, dim * 4, bias=False)
            self.linear2 = torch.nn.Linear(dim * 4, dim, bias=False)
            self.norm = torch.nn.LayerNorm(dim)
        
        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.linear1(x)
            x = F.gelu(x)
            x = self.linear2(x)
            return x + residual
    
    dim = 4096
    num_blocks = 24
    seq_len = 512
    batch_size = 1
    
    print(f"\n配置: dim={dim}, num_blocks={num_blocks}, seq_len={seq_len}")
    
    # 创建模型
    blocks = torch.nn.ModuleList([SimpleBlock(dim) for _ in range(num_blocks)])
    blocks.to(device, dtype)
    blocks.requires_grad_(True)
    
    mem_model = get_gpu_memory_mb()
    print(f"模型加载后显存: {mem_model:.1f} MB")
    
    # 测试 1: 不使用 checkpoint
    print("\n--- 测试 1: 不使用 Gradient Checkpointing ---")
    cleanup()
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    for block in blocks:
        x = block(x)
    loss = x.mean()
    
    peak_no_ckpt = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Forward 峰值: {peak_no_ckpt:.1f} MB")
    
    loss.backward()
    peak_no_ckpt_bwd = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Backward 峰值: {peak_no_ckpt_bwd:.1f} MB")
    
    del x, loss
    cleanup()
    
    # 测试 2: 使用 checkpoint，不 offload
    print("\n--- 测试 2: Gradient Checkpointing (无 CPU Offload) ---")
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    for block in blocks:
        x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
    loss = x.mean()
    
    peak_ckpt = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Forward 峰值: {peak_ckpt:.1f} MB")
    
    loss.backward()
    peak_ckpt_bwd = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Backward 峰值: {peak_ckpt_bwd:.1f} MB")
    
    del x, loss
    cleanup()
    
    # 测试 3: 使用 checkpoint + CPU offload wrapper
    print("\n--- 测试 3: Gradient Checkpointing + CPU Offload Wrapper ---")
    torch.cuda.reset_peak_memory_stats()
    
    def to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, (tuple, list)):
            return type(obj)(to_cpu(o) for o in obj)
        return obj
    
    def to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, (tuple, list)):
            return type(obj)(to_device(o, device) for o in obj)
        return obj
    
    def create_cpu_offload_wrapper(func, device):
        def wrapper(*inputs):
            cuda_inputs = to_device(inputs, device)
            outputs = func(*cuda_inputs)
            return to_cpu(outputs)
        return wrapper
    
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    for block in blocks:
        wrapped_block = create_cpu_offload_wrapper(block, device)
        x = torch.utils.checkpoint.checkpoint(wrapped_block, x, use_reentrant=False)
    
    # x 现在在 CPU 上，需要移回 GPU
    x = x.to(device)
    loss = x.mean()
    
    peak_offload = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Forward 峰值: {peak_offload:.1f} MB")
    
    loss.backward()
    peak_offload_bwd = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Backward 峰值: {peak_offload_bwd:.1f} MB")
    
    # 总结
    print("\n" + "=" * 70)
    print("总结:")
    print("=" * 70)
    print(f"  无 Checkpoint:              Forward {peak_no_ckpt:.1f} MB, Backward {peak_no_ckpt_bwd:.1f} MB")
    print(f"  Checkpoint (无 Offload):    Forward {peak_ckpt:.1f} MB, Backward {peak_ckpt_bwd:.1f} MB")
    print(f"  Checkpoint + CPU Offload:   Forward {peak_offload:.1f} MB, Backward {peak_offload_bwd:.1f} MB")
    print()
    print(f"  Checkpoint 节省: {peak_no_ckpt_bwd - peak_ckpt_bwd:.1f} MB")
    print(f"  CPU Offload 额外节省: {peak_ckpt_bwd - peak_offload_bwd:.1f} MB")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="调试 Activation CPU Offloading")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--mode", type=str, default="simple", choices=["simple", "full", "compare"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_seq_len", type=int, default=256)
    parser.add_argument("--txt_seq_len", type=int, default=128)
    args = parser.parse_args()
    
    if args.mode == "simple":
        # 简单测试：使用模拟的简单模型
        test_manual_offload()
    
    elif args.mode == "full" and args.model_path:
        # 完整测试：使用实际模型
        results = []
        for use_offload in [False, True]:
            torch.cuda.reset_peak_memory_stats()
            result = test_activation_offload(
                args.model_path,
                use_cpu_offload=use_offload,
                batch_size=args.batch_size,
                img_seq_len=args.img_seq_len,
                txt_seq_len=args.txt_seq_len,
            )
            results.append(result)
        
        # 对比结果
        print("\n" + "=" * 70)
        print("对比结果:")
        print("=" * 70)
        print(f"{'配置':<30} {'Forward后':<15} {'Backward后':<15} {'峰值':<15}")
        print("-" * 70)
        for r in results:
            config = f"cpu_offload={r['use_cpu_offload']}"
            print(f"{config:<30} {r['mem_after_forward']:.1f} MB{'':<6} {r['mem_after_backward']:.1f} MB{'':<6} {r['peak_mem']:.1f} MB")
        
        diff = results[0]['peak_mem'] - results[1]['peak_mem']
        print(f"\nCPU Offload 节省: {diff:.1f} MB")
    
    elif args.mode == "compare" and args.model_path:
        # 对比模式：测试实际模型的两种配置
        print("\n" + "#" * 70)
        print("# 对比测试: activation_cpu_offloading 效果")
        print("#" * 70)
        
        for use_offload in [False, True]:
            torch.cuda.reset_peak_memory_stats()
            test_activation_offload(
                args.model_path,
                use_cpu_offload=use_offload,
                batch_size=args.batch_size,
                img_seq_len=args.img_seq_len,
                txt_seq_len=args.txt_seq_len,
            )
    
    else:
        print("请使用 --mode simple 或提供 --model_path")


if __name__ == "__main__":
    main()

