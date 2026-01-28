"""
调试 activation_cpu_offloading 是否导致梯度翻倍的问题

测试配置:
1. 普通 gradient checkpointing (无 CPU offloading)
2. gradient checkpointing + activation_cpu_offloading
3. 无 gradient checkpointing (基准)

隔离变量:
- Block Swap: OFF (排除 block swap 的影响)
- 只关注 activation_cpu_offloading 的影响
"""

import os
import sys
import gc
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class SimpleBlock(nn.Module):
    """简单的测试 block"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, *args, **kwargs):
        return self.norm(self.linear(x) + x)


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


def create_cpu_offloading_wrapper(func, device):
    """复制自 flux2_model.py"""
    def wrapper(orig_func):
        def custom_forward(*inputs):
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            cpu_outputs = to_cpu(outputs)
            return cpu_outputs
        return custom_forward
    return wrapper(func)


def test_simple_model():
    """测试简单模型"""
    print("\n" + "=" * 70)
    print("测试简单模型 - 梯度翻倍问题诊断")
    print("=" * 70)
    
    device = torch.device("cuda:0")
    dtype = torch.float32  # 使用 float32 便于精确比较
    
    dim = 64
    seq_len = 16
    batch_size = 2
    num_blocks = 4
    
    torch.manual_seed(42)
    
    # 创建模型
    blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(num_blocks)])
    blocks.to(device)
    
    # 保存初始权重
    initial_state = {k: v.clone() for k, v in blocks.state_dict().items()}
    
    # 测试输入
    torch.manual_seed(123)
    x_base = torch.randn(batch_size, seq_len, dim, dtype=dtype)
    
    results = {}
    
    for mode_name, use_ckpt, use_cpu_offload in [
        ("无 Checkpointing", False, False),
        ("Checkpointing 无 CPU Offload", True, False),
        ("Checkpointing + CPU Offload", True, True),
    ]:
        print(f"\n>>> 测试: {mode_name}")
        cleanup()
        
        # 重置模型
        blocks.load_state_dict(initial_state)
        blocks.to(device)
        
        x = x_base.clone().to(device).requires_grad_(True)
        
        for block in blocks:
            if use_ckpt:
                if use_cpu_offload:
                    wrapped_block = create_cpu_offloading_wrapper(block, device)
                    x = torch.utils.checkpoint.checkpoint(wrapped_block, x, use_reentrant=False)
                    x = x.to(device)  # 输出在 CPU 上，需要移回
                else:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        loss = x.mean()
        print(f"  Loss: {loss.item():.6f}")
        
        loss.backward()
        
        # 收集梯度
        grads = {}
        for name, param in blocks.named_parameters():
            if param.grad is not None:
                grads[name] = {
                    'norm': param.grad.norm().item(),
                    'mean': param.grad.mean().item(),
                    'data': param.grad.cpu().clone(),
                }
        
        results[mode_name] = {
            'loss': loss.item(),
            'grads': grads,
            'x_grad_norm': x.grad.norm().item() if x.grad is not None else 0,
        }
        
        print(f"  Input grad norm: {results[mode_name]['x_grad_norm']:.6e}")
        for name, g in list(grads.items())[:2]:
            print(f"  {name}: norm={g['norm']:.6e}")
    
    # 比较结果
    print("\n" + "-" * 70)
    print("梯度比较:")
    print("-" * 70)
    
    base = results["无 Checkpointing"]
    for mode_name in ["Checkpointing 无 CPU Offload", "Checkpointing + CPU Offload"]:
        print(f"\n>>> {mode_name} vs 无 Checkpointing:")
        mode = results[mode_name]
        
        input_grad_ratio = mode['x_grad_norm'] / base['x_grad_norm']
        print(f"  Input grad ratio: {input_grad_ratio:.4f}")
        
        for name in list(base['grads'].keys())[:3]:
            if name in mode['grads']:
                ratio = mode['grads'][name]['norm'] / base['grads'][name]['norm']
                diff = (mode['grads'][name]['data'] - base['grads'][name]['data']).abs().max().item()
                print(f"  {name}: ratio={ratio:.4f}, max_diff={diff:.6e}")


def test_flux2_model(model_path: str):
    """测试 FLUX2 模型 - 隔离测试各种配置"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("\n" + "=" * 70)
    print("测试 FLUX2 模型 - 梯度翻倍问题诊断")
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
    
    results = {}
    
    # 测试多种配置，隔离 Block Swap 和 activation_cpu_offloading 的影响
    test_configs = [
        # (mode_name, use_ckpt, use_cpu_offload, use_block_swap)
        ("无 Checkpointing", False, False, False),
        ("Ckpt 无 CPU Offload 无 BlockSwap", True, False, False),
        ("Ckpt + CPU Offload 无 BlockSwap", True, True, False),
        ("Ckpt 无 CPU Offload + BlockSwap", True, False, True),
        ("Ckpt + CPU Offload + BlockSwap", True, True, True),
    ]
    
    for mode_name, use_ckpt, use_cpu_offload, use_block_swap in test_configs:
        print(f"\n>>> 测试: {mode_name}")
        cleanup()
        
        transformer.load_state_dict(initial_state)
        
        # 清理之前的 offloader
        if hasattr(transformer, 'offloader') and transformer.offloader is not None:
            transformer.cleanup_offloader()
        transformer.blocks_to_swap = 0
        
        if use_block_swap:
            blocks_to_swap = transformer.num_single_blocks - 2
            transformer.enable_block_swap(
                blocks_to_swap=blocks_to_swap,
                device=device,
                supports_backward=True,
                use_pinned_memory=False,
            )
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        else:
            transformer.to(device)
        
        transformer.requires_grad_(True)
        
        # 设置 gradient checkpointing
        if use_ckpt:
            transformer.enable_gradient_checkpointing(activation_cpu_offloading=use_cpu_offload)
        else:
            transformer.disable_gradient_checkpointing()
        
        hidden_states = hidden_states_base.clone().to(device).requires_grad_(True)
        
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
        
        grads = {}
        for name, param in transformer.named_parameters():
            if param.grad is not None:
                grads[name] = {
                    'norm': param.grad.float().norm().item(),
                    'data': param.grad.cpu().clone(),
                }
        
        results[mode_name] = {
            'loss': loss.item(),
            'grads': grads,
        }
        
        print(f"  收集了 {len(grads)} 个参数的梯度")
        for name, g in list(grads.items())[:3]:
            print(f"  {name}: norm={g['norm']:.6e}")
        
        # 清理 offloader
        if use_block_swap and hasattr(transformer, 'offloader') and transformer.offloader is not None:
            transformer.cleanup_offloader()
    
    # 比较结果
    print("\n" + "-" * 70)
    print("梯度比较 (以'无 Checkpointing'为基准):")
    print("-" * 70)
    
    if "无 Checkpointing" not in results:
        print("缺少基准测试结果")
        return
    
    base = results["无 Checkpointing"]
    
    for mode_name in results.keys():
        if mode_name == "无 Checkpointing":
            continue
        
        print(f"\n>>> {mode_name}:")
        mode = results[mode_name]
        
        ratios = []
        for name in list(base['grads'].keys())[:5]:
            if name in mode['grads'] and base['grads'][name]['norm'] > 1e-10:
                ratio = mode['grads'][name]['norm'] / base['grads'][name]['norm']
                ratios.append(ratio)
                diff = (mode['grads'][name]['data'].float() - base['grads'][name]['data'].float()).abs().max().item()
                print(f"  {name}: ratio={ratio:.4f}, max_diff={diff:.6e}")
        
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"  >>> 平均 ratio: {avg_ratio:.4f}")
            if abs(avg_ratio - 1.0) > 0.1:
                print(f"  ⚠️ 警告: 梯度比例异常！")
            if abs(avg_ratio - 2.0) < 0.1:
                print(f"  ⚠️ 疑似梯度翻倍问题！")


def test_checkpoint_wrapper_isolation():
    """隔离测试 checkpoint wrapper 的行为"""
    print("\n" + "=" * 70)
    print("隔离测试: checkpoint + CPU offload wrapper")
    print("=" * 70)
    
    device = torch.device("cuda:0")
    
    # 简单的线性层
    linear = nn.Linear(16, 16).to(device)
    
    torch.manual_seed(42)
    x_base = torch.randn(2, 8, 16, device=device, requires_grad=True)
    
    # 测试 1: 直接调用
    x1 = x_base.clone().detach().requires_grad_(True)
    y1 = linear(x1)
    loss1 = y1.mean()
    loss1.backward()
    grad1 = linear.weight.grad.clone()
    linear.zero_grad()
    
    # 测试 2: checkpoint (无 offload)
    x2 = x_base.clone().detach().requires_grad_(True)
    y2 = torch.utils.checkpoint.checkpoint(linear, x2, use_reentrant=False)
    loss2 = y2.mean()
    loss2.backward()
    grad2 = linear.weight.grad.clone()
    linear.zero_grad()
    
    # 测试 3: checkpoint + CPU offload wrapper
    x3 = x_base.clone().detach().requires_grad_(True)
    wrapped = create_cpu_offloading_wrapper(linear, device)
    y3 = torch.utils.checkpoint.checkpoint(wrapped, x3, use_reentrant=False)
    y3 = y3.to(device)  # 从 CPU 移回 GPU
    loss3 = y3.mean()
    loss3.backward()
    grad3 = linear.weight.grad.clone()
    linear.zero_grad()
    
    print(f"\n直接调用:")
    print(f"  loss={loss1.item():.6f}, grad_norm={grad1.norm().item():.6e}")
    
    print(f"\nCheckpoint (无 offload):")
    print(f"  loss={loss2.item():.6f}, grad_norm={grad2.norm().item():.6e}")
    print(f"  vs 直接: ratio={grad2.norm().item() / grad1.norm().item():.4f}")
    
    print(f"\nCheckpoint + CPU Offload:")
    print(f"  loss={loss3.item():.6f}, grad_norm={grad3.norm().item():.6e}")
    print(f"  vs 直接: ratio={grad3.norm().item() / grad1.norm().item():.4f}")
    
    # 检查梯度差异
    print(f"\n梯度差异:")
    print(f"  直接 vs ckpt: {(grad1 - grad2).abs().max().item():.6e}")
    print(f"  直接 vs ckpt+offload: {(grad1 - grad3).abs().max().item():.6e}")
    print(f"  ckpt vs ckpt+offload: {(grad2 - grad3).abs().max().item():.6e}")


def test_multiple_blocks_checkpoint():
    """测试多个 block 串联时的 checkpoint 行为"""
    print("\n" + "=" * 70)
    print("测试: 多个 block 串联的 checkpoint 行为")
    print("=" * 70)
    
    device = torch.device("cuda:0")
    num_blocks = 4
    dim = 32
    
    blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(num_blocks)]).to(device)
    
    torch.manual_seed(42)
    x_base = torch.randn(2, 8, dim, device=device)
    
    # 保存初始状态
    init_state = {k: v.clone() for k, v in blocks.state_dict().items()}
    
    results = []
    
    for mode, use_ckpt, use_offload in [
        ("直接", False, False),
        ("Ckpt", True, False),
        ("Ckpt+Offload", True, True),
    ]:
        blocks.load_state_dict(init_state)
        blocks.zero_grad()  # 清零梯度！
        
        x = x_base.clone().requires_grad_(True)
        
        for i, block in enumerate(blocks):
            if use_ckpt:
                if use_offload:
                    wrapped = create_cpu_offloading_wrapper(block, device)
                    x = torch.utils.checkpoint.checkpoint(wrapped, x, use_reentrant=False)
                    x = x.to(device)
                else:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        loss = x.mean()
        loss.backward()
        
        # 收集第一个 block 的梯度
        grad_norm = blocks[0].linear.weight.grad.norm().item()
        
        results.append({
            'mode': mode,
            'loss': loss.item(),
            'grad_norm': grad_norm,
        })
        
        print(f"\n{mode}:")
        print(f"  loss={loss.item():.6f}")
        print(f"  block[0].linear.weight grad_norm={grad_norm:.6e}")
    
    # 比较
    print("\n比较:")
    base = results[0]
    for r in results[1:]:
        ratio = r['grad_norm'] / base['grad_norm']
        print(f"  {r['mode']} vs {base['mode']}: ratio={ratio:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="调试梯度翻倍问题")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--test", type=str, default="all", 
                       choices=["simple", "flux2", "wrapper", "multi", "all"])
    args = parser.parse_args()
    
    if args.test in ["wrapper", "all"]:
        test_checkpoint_wrapper_isolation()
    
    if args.test in ["multi", "all"]:
        test_multiple_blocks_checkpoint()
    
    if args.test in ["simple", "all"]:
        test_simple_model()
    
    if args.test in ["flux2", "all"] and args.model_path:
        test_flux2_model(args.model_path)


if __name__ == "__main__":
    main()

