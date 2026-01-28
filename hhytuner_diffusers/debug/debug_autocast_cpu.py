import torch
import sys
sys.path.insert(0, "D:/ai/hhytuner/hhytuner_diffusers")

def test_autocast_cpu_offload():
    from flux2_modules import load_flux2_transformer_from_diffusers
    from flux2_modules.flux2_model import create_cpu_offloading_wrapper
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    model_path = "F:/models/FLUX.2-klein-base-9B"
    
    print("加载模型...")
    transformer = load_flux2_transformer_from_diffusers(model_path, torch_dtype=dtype, device=device)
    transformer.eval()
    
    num_attention_heads = transformer.config.get("num_attention_heads", 48)
    attention_head_dim = transformer.config.get("attention_head_dim", 128)
    hidden_dim = num_attention_heads * attention_head_dim
    joint_attention_dim = transformer.config.get("joint_attention_dim", 15360)
    
    print(f"模型配置: hidden_dim={hidden_dim}, joint_attention_dim={joint_attention_dim}")
    
    batch_size = 1
    img_seq_len = 256
    txt_seq_len = 128
    
    in_channels = transformer.config.get("in_channels", 128)
    hidden_states = torch.randn(batch_size, img_seq_len, in_channels, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, txt_seq_len, joint_attention_dim, device=device, dtype=dtype)
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    
    img_ids = torch.zeros(img_seq_len, 4, device=device, dtype=dtype)
    for i in range(img_seq_len):
        h = i // 16
        w = i % 16
        img_ids[i] = torch.tensor([0, 0, h, w], dtype=dtype)
    
    txt_ids = torch.zeros(txt_seq_len, 4, device=device, dtype=dtype)
    
    print("\n" + "=" * 60)
    print("测试完整模型 forward + autocast")
    print("=" * 60)
    
    with torch.no_grad():
        print("\n--- 无 autocast ---")
        out_normal = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
        )
        print(f"输出: mean={out_normal['sample'].mean():.6f}, std={out_normal['sample'].std():.6f}")
        
        print("\n--- 有 autocast ---")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_autocast = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
            )
        print(f"输出: mean={out_autocast['sample'].mean():.6f}, std={out_autocast['sample'].std():.6f}")
        
        diff = (out_normal['sample'] - out_autocast['sample']).abs()
        print(f"差异: mean={diff.mean():.6f}, max={diff.max():.6f}")
    
    print("\n" + "=" * 60)
    print("测试 activation_cpu_offloading + autocast")
    print("=" * 60)
    
    print("\n--- 启用 gradient checkpointing (无 cpu offload) ---")
    transformer.enable_gradient_checkpointing(activation_cpu_offloading=False)
    hidden_states_grad = hidden_states.clone().requires_grad_(True)
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_ckpt = transformer(
            hidden_states=hidden_states_grad,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
        )
    print(f"输出: mean={out_ckpt['sample'].mean():.6f}, std={out_ckpt['sample'].std():.6f}")
    
    print("\n--- 启用 gradient checkpointing + cpu offload ---")
    import flux2_modules.flux2_model as fm
    fm._cpu_offload_debug = True
    transformer.enable_gradient_checkpointing(activation_cpu_offloading=True)
    hidden_states_grad2 = hidden_states.clone().requires_grad_(True)
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_ckpt_offload = transformer(
            hidden_states=hidden_states_grad2,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
        )
    print(f"输出: mean={out_ckpt_offload['sample'].mean():.6f}, std={out_ckpt_offload['sample'].std():.6f}")
    
    diff = (out_ckpt['sample'] - out_ckpt_offload['sample']).abs()
    print(f"\n差异: mean={diff.mean():.6f}, max={diff.max():.6f}")
    
    if diff.max() > 0.01:
        print("\n[ERROR] CPU Offload 导致输出不一致!")
        print(f"  无 offload std: {out_ckpt['sample'].std():.6f}")
        print(f"  有 offload std: {out_ckpt_offload['sample'].std():.6f}")
        print(f"  std 比例: {out_ckpt_offload['sample'].std() / out_ckpt['sample'].std():.4f}")
    else:
        print("\n[OK] 输出一致")
    
    del transformer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_autocast_cpu_offload()
