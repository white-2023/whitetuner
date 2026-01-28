"""
训练健康检查：诊断训练效果差的原因
"""

import os
import sys
import gc
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_model_weights(model_path: str):
    """检查模型权重是否正常"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    print("=" * 70)
    print("检查 1: 模型权重统计")
    print("=" * 70)
    
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=torch.bfloat16, device="cpu")
    
    print("\n关键层权重统计:")
    key_layers = [
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "x_embedder.weight",
        "context_embedder.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
    ]
    
    for name in key_layers:
        try:
            param = dict(transformer.named_parameters())[name]
            w = param.float()
            print(f"  {name}:")
            print(f"    shape={tuple(w.shape)}, mean={w.mean():.6f}, std={w.std():.6f}")
            print(f"    min={w.min():.6f}, max={w.max():.6f}")
            
            # 检查异常值
            if w.std() < 1e-6:
                print(f"    [WARNING] 权重方差过小！")
            if torch.isnan(w).any():
                print(f"    [ERROR] 包含 NaN！")
            if torch.isinf(w).any():
                print(f"    [ERROR] 包含 Inf！")
        except KeyError:
            print(f"  {name}: [NOT FOUND]")


def check_gradient_flow(model_path: str):
    """检查梯度流是否正常"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    print("\n" + "=" * 70)
    print("检查 2: 梯度流")
    print("=" * 70)
    
    cleanup()
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device=device)
    transformer.requires_grad_(True)
    transformer.enable_gradient_checkpointing(activation_cpu_offloading=False)
    
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64
    txt_seq_len = 32
    
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype, device=device, requires_grad=True)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim, dtype=dtype, device=device)
    timestep = torch.tensor([0.5], dtype=dtype, device=device)
    
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, dtype=dtype, device=device)
    for i in range(img_seq_len):
        img_ids[i] = torch.tensor([0, 0, i // width, i % width], dtype=dtype)
    txt_ids = torch.zeros(txt_seq_len, 4, dtype=dtype, device=device)
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
        )
    
    loss = output['sample'].float().mean()
    loss.backward()
    
    print("\n梯度统计（按模块）:")
    
    # 按模块分组统计
    module_grads = {}
    for name, param in transformer.named_parameters():
        if param.grad is not None:
            module = name.split('.')[0]
            if module not in module_grads:
                module_grads[module] = []
            module_grads[module].append(param.grad.float().norm().item())
    
    for module, norms in module_grads.items():
        avg_norm = sum(norms) / len(norms)
        max_norm = max(norms)
        min_norm = min(norms)
        print(f"  {module}: avg={avg_norm:.6e}, max={max_norm:.6e}, min={min_norm:.6e}")
        
        if avg_norm < 1e-10:
            print(f"    [WARNING] 梯度过小！")
        if avg_norm > 1e3:
            print(f"    [WARNING] 梯度过大！")
    
    # 检查是否有参数没有梯度
    no_grad_count = sum(1 for _, p in transformer.named_parameters() if p.grad is None)
    total_count = sum(1 for _ in transformer.parameters())
    print(f"\n参数梯度覆盖: {total_count - no_grad_count}/{total_count}")


def simulate_training_step(model_path: str, learning_rate: float = 1e-5):
    """模拟一步训练，检查参数更新"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    import bitsandbytes as bnb
    
    print("\n" + "=" * 70)
    print(f"检查 3: 模拟训练步 (lr={learning_rate})")
    print("=" * 70)
    
    cleanup()
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device=device)
    transformer.requires_grad_(True)
    transformer.enable_gradient_checkpointing(activation_cpu_offloading=False)
    
    # 保存初始权重
    initial_weights = {}
    for name, param in transformer.named_parameters():
        initial_weights[name] = param.data.float().clone()
    
    # 创建优化器
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-6,
    )
    
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64
    txt_seq_len = 32
    
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype, device=device, requires_grad=True)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim, dtype=dtype, device=device)
    timestep = torch.tensor([0.5], dtype=dtype, device=device)
    
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, dtype=dtype, device=device)
    for i in range(img_seq_len):
        img_ids[i] = torch.tensor([0, 0, i // width, i % width], dtype=dtype)
    txt_ids = torch.zeros(txt_seq_len, 4, dtype=dtype, device=device)
    
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
    
    # 模拟 MSE loss
    target = torch.randn_like(output['sample'])
    loss = F.mse_loss(output['sample'].float(), target.float())
    
    print(f"\nStep 0 - Loss: {loss.item():.6f}")
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # 检查权重变化
    print("\n权重变化:")
    key_layers = [
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
    ]
    
    for name in key_layers:
        try:
            param = dict(transformer.named_parameters())[name]
            before = initial_weights[name]
            after = param.data.float()
            diff = (after - before).abs()
            
            print(f"  {name}:")
            print(f"    max_change={diff.max().item():.6e}, mean_change={diff.mean().item():.6e}")
            
            if diff.max().item() < 1e-10:
                print(f"    [WARNING] 权重几乎没变化！")
            if diff.max().item() > 0.1:
                print(f"    [WARNING] 权重变化过大！")
        except KeyError:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for simulation")
    args = parser.parse_args()
    
    check_model_weights(args.model_path)
    check_gradient_flow(args.model_path)
    simulate_training_step(args.model_path, args.lr)


if __name__ == "__main__":
    main()

