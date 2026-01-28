"""
直接测试 FLUX2 模型的 gradient checkpointing 行为
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


def test_flux2_checkpointing(model_path: str):
    """测试 FLUX2 模型的 checkpointing 行为"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("=" * 70)
    print("FLUX2 Gradient Checkpointing 诊断")
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
    
    # 测试参数
    test_params = [
        ("time_guidance_embed.timestep_embedder.linear_1.weight", "time embed"),
        ("double_stream_modulation_img.linear.weight", "mod img"),
        ("single_stream_modulation.linear.weight", "single mod"),
        ("transformer_blocks.0.attn.to_q.weight", "double block 0"),
        ("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", "single block 0"),
    ]
    
    results = {}
    
    for mode_name, use_ckpt in [("无 Checkpointing", False), ("有 Checkpointing", True)]:
        print(f"\n>>> 测试: {mode_name}")
        cleanup()
        
        transformer.load_state_dict(initial_state)
        transformer.to(device)
        transformer.requires_grad_(True)
        
        # 确保没有 block swap
        transformer.blocks_to_swap = 0
        if hasattr(transformer, 'offloader') and transformer.offloader is not None:
            transformer.cleanup_offloader()
        
        # 设置 gradient checkpointing
        if use_ckpt:
            transformer.enable_gradient_checkpointing(activation_cpu_offloading=False)
        else:
            transformer.disable_gradient_checkpointing()
        
        hidden_states = hidden_states_base.clone().to(device).requires_grad_(True)
        
        # 添加 hook 来监控中间计算
        forward_count = {}
        def make_forward_hook(name):
            def hook(module, input, output):
                if name not in forward_count:
                    forward_count[name] = 0
                forward_count[name] += 1
            return hook
        
        # 注册 hooks
        hooks = []
        hooks.append(transformer.time_guidance_embed.register_forward_hook(
            make_forward_hook("time_guidance_embed")))
        hooks.append(transformer.double_stream_modulation_img.register_forward_hook(
            make_forward_hook("double_stream_modulation_img")))
        hooks.append(transformer.single_stream_modulation.register_forward_hook(
            make_forward_hook("single_stream_modulation")))
        hooks.append(transformer.transformer_blocks[0].register_forward_hook(
            make_forward_hook("transformer_blocks.0")))
        hooks.append(transformer.single_transformer_blocks[0].register_forward_hook(
            make_forward_hook("single_transformer_blocks.0")))
        
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
        
        # Forward 计数
        print(f"  Forward 调用次数:")
        for name, count in forward_count.items():
            print(f"    {name}: {count}")
        
        # 清除 forward hooks
        for h in hooks:
            h.remove()
        
        # Backward
        forward_count.clear()
        
        loss.backward()
        
        # 收集梯度
        grads = {}
        for param_name, display_name in test_params:
            param = dict(transformer.named_parameters())[param_name]
            if param.grad is not None:
                grads[display_name] = param.grad.float().norm().item()
        
        results[mode_name] = grads
        
        print(f"  梯度 norm:")
        for name, norm in grads.items():
            print(f"    {name}: {norm:.6e}")
    
    # 比较
    print("\n" + "-" * 70)
    print("梯度比较 (有 Checkpointing / 无 Checkpointing):")
    print("-" * 70)
    
    for name in results["无 Checkpointing"]:
        no_ckpt = results["无 Checkpointing"][name]
        with_ckpt = results["有 Checkpointing"][name]
        ratio = with_ckpt / no_ckpt if no_ckpt > 0 else 0
        status = "OK" if abs(ratio - 1.0) < 0.1 else "ABNORMAL"
        print(f"  {name}: ratio={ratio:.4f} {status}")


def test_ckpt_func_comparison(model_path: str):
    """对比原始 _gradient_checkpointing_func 和简化版本"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("\n" + "=" * 70)
    print("对比原始 _gradient_checkpointing_func 和简化版本")
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
    
    # 保存原始方法的引用
    original_ckpt_func = transformer._gradient_checkpointing_func
    
    # 简化版本（不调用 create_cpu_offloading_wrapper）
    def simple_ckpt_func(self, block, *args):
        return torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
    
    import types
    
    for mode_name, use_original in [
        ("原始 _gradient_checkpointing_func", True),
        ("简化版本 (无 wrapper)", False),
    ]:
        print(f"\n>>> 测试: {mode_name}")
        cleanup()
        
        transformer.load_state_dict(initial_state)
        transformer.to(device)
        transformer.requires_grad_(True)
        transformer.blocks_to_swap = 0
        if hasattr(transformer, 'offloader') and transformer.offloader is not None:
            transformer.cleanup_offloader()
        
        if use_original:
            # 使用原始方法
            transformer._gradient_checkpointing_func = types.MethodType(
                lambda self, block, *args: torch.utils.checkpoint.checkpoint(
                    block, *args, use_reentrant=False
                ) if not self.activation_cpu_offloading else self.__class__._gradient_checkpointing_func(self, block, *args),
                transformer
            )
            # 实际上直接用原始方法
            transformer._gradient_checkpointing_func = original_ckpt_func
        else:
            # 使用简化版本
            transformer._gradient_checkpointing_func = types.MethodType(simple_ckpt_func, transformer)
        
        transformer.enable_gradient_checkpointing(activation_cpu_offloading=False)
        
        # 打印关键状态
        print(f"  activation_cpu_offloading: {transformer.activation_cpu_offloading}")
        print(f"  gradient_checkpointing: {transformer.gradient_checkpointing}")
        
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
        loss.backward()
        
        grad_norm = transformer.time_guidance_embed.timestep_embedder.linear_1.weight.grad.float().norm().item()
        results[mode_name] = grad_norm
        print(f"  time_embed grad: {grad_norm:.6e}")
    
    print("\n比较:")
    base = list(results.values())[1]  # 简化版本作为基准
    for mode, norm in results.items():
        ratio = norm / base if base > 0 else 0
        print(f"  {mode}: {norm:.6e} (ratio={ratio:.4f})")


def test_individual_components(model_path: str):
    """分别测试各个组件"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("\n" + "=" * 70)
    print("分别测试各个组件")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    print("\n>>> 加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device=device)
    
    # 只测试 time_guidance_embed
    print("\n>>> 测试 time_guidance_embed (不涉及 block):")
    
    temb_input = torch.randn(1, dtype=dtype, device=device) * 1000
    guidance = None
    
    for use_ckpt in [False, True]:
        transformer.zero_grad()
        
        if use_ckpt:
            out = torch.utils.checkpoint.checkpoint(
                transformer.time_guidance_embed, temb_input, guidance,
                use_reentrant=False
            )
        else:
            out = transformer.time_guidance_embed(temb_input, guidance)
        
        loss = out.float().mean()
        loss.backward()
        
        grad_norm = transformer.time_guidance_embed.timestep_embedder.linear_1.weight.grad.float().norm().item()
        mode = "ckpt" if use_ckpt else "直接"
        print(f"  {mode}: grad_norm={grad_norm:.6e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test", type=str, default="all", choices=["all", "ckpt", "compare", "individual"])
    args = parser.parse_args()
    
    if args.test in ["all", "ckpt"]:
        test_flux2_checkpointing(args.model_path)
    
    if args.test in ["all", "compare"]:
        test_ckpt_func_comparison(args.model_path)
    
    if args.test in ["all", "individual"]:
        test_individual_components(args.model_path)


if __name__ == "__main__":
    main()

