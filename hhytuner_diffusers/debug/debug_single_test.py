"""
单次测试：验证 gradient checkpointing 是否导致梯度异常
"""

import os
import sys
import gc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_single_run(model_path: str, use_checkpointing: bool):
    """单次测试"""
    from flux2_modules import load_flux2_transformer_from_diffusers
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    mode = "有" if use_checkpointing else "无"
    print(f"\n>>> 测试 {mode} Gradient Checkpointing")
    
    cleanup()
    torch.manual_seed(42)
    
    print(">>> 加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device=device)
    transformer.requires_grad_(True)
    transformer.blocks_to_swap = 0
    
    if use_checkpointing:
        transformer.enable_gradient_checkpointing(activation_cpu_offloading=False)
    else:
        transformer.disable_gradient_checkpointing()
    
    print(f"  gradient_checkpointing: {transformer.gradient_checkpointing}")
    print(f"  activation_cpu_offloading: {transformer.activation_cpu_offloading}")
    
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    in_channels = transformer.config.get("in_channels", 128)
    
    batch_size = 1
    img_seq_len = 64
    txt_seq_len = 32
    
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype, device=device, requires_grad=True)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim, dtype=dtype, device=device)
    timestep = torch.tensor([0.5], dtype=dtype, device=device)
    
    height = int(img_seq_len ** 0.5)
    width = img_seq_len // height
    img_ids = torch.zeros(img_seq_len, 4, dtype=dtype, device=device)
    for i in range(img_seq_len):
        h = i // width
        w = i % width
        img_ids[i] = torch.tensor([0, 0, h, w], dtype=dtype)
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
    print(f"  Loss: {loss.item():.6f}")
    
    loss.backward()
    
    # 打印关键参数的梯度
    params_to_check = [
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "double_stream_modulation_img.linear.weight",
        "single_stream_modulation.linear.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
    ]
    
    print(f"  梯度 norm:")
    for name in params_to_check:
        param = dict(transformer.named_parameters())[name]
        if param.grad is not None:
            grad_norm = param.grad.float().norm().item()
            print(f"    {name.split('.')[-2]}: {grad_norm:.6e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpointing", action="store_true", help="Enable gradient checkpointing")
    args = parser.parse_args()
    
    test_single_run(args.model_path, args.checkpointing)


if __name__ == "__main__":
    main()

