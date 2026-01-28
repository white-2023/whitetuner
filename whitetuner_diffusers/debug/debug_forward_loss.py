"""
对比 Block Swap 模式 vs 普通模式的 Forward 输出和 Loss
问题：启用 Block Swap 后起始 loss 和普通模式不一样
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
        torch.cuda.synchronize()


def test_forward_output(model_path: str):
    """测试 forward 输出是否一致"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("=" * 70)
    print("测试: Block Swap vs 普通模式的 Forward 输出")
    print("=" * 70)
    
    # 固定随机种子
    torch.manual_seed(42)
    
    print("\n>>> 加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device="cpu")
    
    # 保存初始权重
    initial_state = {k: v.clone() for k, v in transformer.state_dict().items()}
    
    # 准备固定输入
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64
    txt_seq_len = 32
    
    torch.manual_seed(123)  # 固定输入的随机种子
    hidden_states_base = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
    encoder_hidden_states_base = torch.randn(batch_size, txt_seq_len, joint_attention_dim, dtype=dtype)
    timestep_base = torch.tensor([0.5], dtype=dtype)
    
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, dtype=dtype)
    for i in range(img_seq_len):
        img_ids[i] = torch.tensor([0, 0, i // width, i % width], dtype=dtype)
    txt_ids = torch.zeros(txt_seq_len, 4, dtype=dtype)
    
    # 固定 target 用于计算 loss
    torch.manual_seed(456)
    target_base = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
    
    results = {}
    
    # 测试配置
    configs = [
        ("普通模式 (无 Block Swap)", 0),
        ("Block Swap = 1", 1),
        ("Block Swap = 10", 10),
        ("Block Swap = 22", 22),
    ]
    
    for mode_name, blocks_to_swap in configs:
        print(f"\n>>> 测试: {mode_name}")
        cleanup()
        
        # 重新加载模型到初始状态
        transformer.load_state_dict(initial_state)
        
        if blocks_to_swap > 0:
            transformer.enable_block_swap(
                blocks_to_swap=blocks_to_swap,
                device=device,
                supports_backward=False,  # 只测试 forward
                use_pinned_memory=False,
            )
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        else:
            transformer.to(device)
        
        # 不启用 gradient checkpointing，只测试纯 forward
        transformer.disable_gradient_checkpointing()
        
        # 准备输入
        hidden_states = hidden_states_base.clone().to(device)
        encoder_hidden_states = encoder_hidden_states_base.clone().to(device)
        timestep = timestep_base.clone().to(device)
        target = target_base.clone().to(device)
        
        # Forward (no grad)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    img_ids=img_ids.to(device),
                    txt_ids=txt_ids.to(device),
                    guidance=None,
                )
        
        model_output = output['sample']
        
        # 计算 loss
        loss = F.mse_loss(model_output.float(), target.float())
        
        # 输出统计
        output_mean = model_output.float().mean().item()
        output_std = model_output.float().std().item()
        output_min = model_output.float().min().item()
        output_max = model_output.float().max().item()
        
        results[mode_name] = {
            'loss': loss.item(),
            'output_mean': output_mean,
            'output_std': output_std,
            'output_min': output_min,
            'output_max': output_max,
            'output': model_output.cpu().clone(),
        }
        
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Output: mean={output_mean:.6f}, std={output_std:.6f}")
        print(f"  Output: min={output_min:.6f}, max={output_max:.6f}")
        
        # 清理 offloader
        if blocks_to_swap > 0:
            transformer.cleanup_offloader()
    
    # 比较结果
    print("\n" + "=" * 70)
    print("对比结果:")
    print("=" * 70)
    
    base_mode = "普通模式 (无 Block Swap)"
    base_output = results[base_mode]['output']
    base_loss = results[base_mode]['loss']
    
    print(f"\n基准: {base_mode}")
    print(f"  Loss: {base_loss:.6f}")
    
    for mode_name, data in results.items():
        if mode_name == base_mode:
            continue
        
        # 计算输出差异
        output_diff = (data['output'] - base_output).abs()
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()
        
        loss_diff = data['loss'] - base_loss
        
        print(f"\n{mode_name}:")
        print(f"  Loss: {data['loss']:.6f} (diff={loss_diff:+.6f})")
        print(f"  Output diff: max={max_diff:.6e}, mean={mean_diff:.6e}")
        
        if max_diff > 1e-3:
            print(f"  [WARNING] 输出差异较大！")
        elif max_diff > 1e-5:
            print(f"  [INFO] 输出有轻微差异（可能是数值精度）")
        else:
            print(f"  [OK] 输出一致")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    test_forward_output(args.model_path)


if __name__ == "__main__":
    main()

