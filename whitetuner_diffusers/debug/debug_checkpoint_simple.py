"""
最简化测试：验证 checkpoint + autocast 是否导致梯度翻倍
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class SimpleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(self.linear(x) + x)


def test_checkpoint_with_autocast():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    dim = 64
    batch_size = 2
    seq_len = 8
    num_blocks = 4
    
    print("=" * 70)
    print("测试: Checkpoint + Autocast 梯度问题")
    print("=" * 70)
    
    # 创建模型
    blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(num_blocks)])
    blocks.to(device, dtype=dtype)
    
    # 保存初始状态
    init_state = {k: v.clone() for k, v in blocks.state_dict().items()}
    
    torch.manual_seed(42)
    x_base = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    
    results = {}
    
    for mode in ["直接调用", "Checkpoint", "Autocast", "Checkpoint+Autocast"]:
        print(f"\n>>> 测试: {mode}")
        blocks.load_state_dict(init_state)
        blocks.zero_grad()
        
        x = x_base.clone().requires_grad_(True)
        
        use_ckpt = "Checkpoint" in mode
        use_autocast = "Autocast" in mode
        
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for block in blocks:
                    if use_ckpt:
                        x = checkpoint(block, x, use_reentrant=False)
                    else:
                        x = block(x)
        else:
            for block in blocks:
                if use_ckpt:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        
        loss = x.float().mean()
        loss.backward()
        
        grad_norm = blocks[0].linear.weight.grad.float().norm().item()
        results[mode] = grad_norm
        
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Grad norm: {grad_norm:.6e}")
    
    # 比较
    print("\n" + "-" * 70)
    print("比较结果:")
    print("-" * 70)
    base = results["直接调用"]
    for mode, norm in results.items():
        ratio = norm / base if base > 0 else 0
        print(f"  {mode}: ratio={ratio:.4f}")


def test_checkpoint_reentrant():
    """测试 use_reentrant 参数的影响"""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    dim = 64
    batch_size = 2
    seq_len = 8
    
    print("\n" + "=" * 70)
    print("测试: use_reentrant 参数的影响")
    print("=" * 70)
    
    linear = nn.Linear(dim, dim).to(device, dtype=dtype)
    init_state = linear.state_dict()
    
    torch.manual_seed(42)
    x_base = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    
    for mode in ["直接", "ckpt(reentrant=True)", "ckpt(reentrant=False)"]:
        print(f"\n>>> 测试: {mode}")
        linear.load_state_dict(init_state)
        linear.zero_grad()
        
        x = x_base.clone().requires_grad_(True)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if mode == "直接":
                y = linear(x)
            elif mode == "ckpt(reentrant=True)":
                y = checkpoint(linear, x, use_reentrant=True)
            else:
                y = checkpoint(linear, x, use_reentrant=False)
        
        loss = y.float().mean()
        loss.backward()
        
        grad_norm = linear.weight.grad.float().norm().item()
        print(f"  Grad norm: {grad_norm:.6e}")


def test_multi_block_accumulation():
    """测试多个 block 时梯度是否异常累积"""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    dim = 64
    batch_size = 2
    seq_len = 8
    
    print("\n" + "=" * 70)
    print("测试: 多 block 梯度累积")
    print("=" * 70)
    
    for num_blocks in [1, 2, 4, 8]:
        blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(num_blocks)])
        blocks.to(device, dtype=dtype)
        
        torch.manual_seed(42)
        x_base = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
        
        results = {}
        init_state = {k: v.clone() for k, v in blocks.state_dict().items()}
        
        for use_ckpt in [False, True]:
            blocks.load_state_dict(init_state)
            blocks.zero_grad()
            
            x = x_base.clone().requires_grad_(True)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for block in blocks:
                    if use_ckpt:
                        x = checkpoint(block, x, use_reentrant=False)
                    else:
                        x = block(x)
            
            loss = x.float().mean()
            loss.backward()
            
            grad_norm = blocks[0].linear.weight.grad.float().norm().item()
            mode = "ckpt" if use_ckpt else "直接"
            results[mode] = grad_norm
        
        ratio = results["ckpt"] / results["直接"] if results["直接"] > 0 else 0
        print(f"  num_blocks={num_blocks}: 直接={results['直接']:.6e}, ckpt={results['ckpt']:.6e}, ratio={ratio:.4f}")


def test_modulation_params():
    """测试带有 modulation 参数的情况 - 模拟 FLUX2 的结构"""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    dim = 64
    batch_size = 2
    seq_len = 8
    num_blocks = 4
    
    print("\n" + "=" * 70)
    print("测试: 带 modulation 参数 (模拟 FLUX2 结构)")
    print("=" * 70)
    
    class ModulatedBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
        
        def forward(self, x, mod_params):
            # mod_params 是外部传入的，每个 block 都共享同一个
            scale = mod_params[:, :x.shape[1], :]
            return self.norm(self.linear(x) * scale + x)
    
    blocks = nn.ModuleList([ModulatedBlock(dim) for _ in range(num_blocks)])
    blocks.to(device, dtype=dtype)
    
    # 模拟 modulation 参数（所有 block 共享）
    mod_linear = nn.Linear(dim, dim * seq_len).to(device, dtype=dtype)
    
    init_state_blocks = {k: v.clone() for k, v in blocks.state_dict().items()}
    init_state_mod = {k: v.clone() for k, v in mod_linear.state_dict().items()}
    
    torch.manual_seed(42)
    x_base = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    temb = torch.randn(batch_size, dim, dtype=dtype, device=device)
    
    results = {}
    
    for use_ckpt in [False, True]:
        blocks.load_state_dict(init_state_blocks)
        mod_linear.load_state_dict(init_state_mod)
        blocks.zero_grad()
        mod_linear.zero_grad()
        
        x = x_base.clone().requires_grad_(True)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mod_params = mod_linear(temb).view(batch_size, seq_len, dim)
            
            for block in blocks:
                if use_ckpt:
                    x = checkpoint(block, x, mod_params, use_reentrant=False)
                else:
                    x = block(x, mod_params)
        
        loss = x.float().mean()
        loss.backward()
        
        block_grad_norm = blocks[0].linear.weight.grad.float().norm().item()
        mod_grad_norm = mod_linear.weight.grad.float().norm().item()
        
        mode = "ckpt" if use_ckpt else "直接"
        results[mode] = {
            'block': block_grad_norm,
            'mod': mod_grad_norm,
        }
        
        print(f"\n>>> {mode}:")
        print(f"  block[0].linear.weight grad: {block_grad_norm:.6e}")
        print(f"  mod_linear.weight grad: {mod_grad_norm:.6e}")
    
    # 比较
    print("\n比较:")
    block_ratio = results["ckpt"]["block"] / results["直接"]["block"]
    mod_ratio = results["ckpt"]["mod"] / results["直接"]["mod"]
    print(f"  block grad ratio: {block_ratio:.4f}")
    print(f"  mod grad ratio: {mod_ratio:.4f}")
    
    if abs(mod_ratio - num_blocks) < 0.1:
        print(f"\n  ⚠️ 发现问题: mod 参数梯度被累加了 {num_blocks} 次！")
        print(f"  原因: checkpoint 重新计算 forward 时，mod_params 的梯度被多次累加")


def test_flux2_like_structure():
    """精确模拟 FLUX2 结构来定位问题"""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    dim = 64
    batch_size = 2
    seq_len = 8
    num_double_blocks = 2
    num_single_blocks = 4
    
    print("\n" + "=" * 70)
    print("测试: FLUX2-like 结构 (double + single blocks)")
    print("=" * 70)
    
    class DoubleBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear_img = nn.Linear(dim, dim)
            self.linear_txt = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
        
        def forward(self, hidden_states, encoder_hidden_states, mod_img, mod_txt, rotary_emb):
            # 模拟 double stream block
            h = self.linear_img(hidden_states) * mod_img + hidden_states
            e = self.linear_txt(encoder_hidden_states) * mod_txt + encoder_hidden_states
            return self.norm(e), self.norm(h)
    
    class SingleBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
        
        def forward(self, hidden_states, encoder_hidden_states, mod, rotary_emb, joint_kwargs):
            # 模拟 single stream block，忽略 encoder_hidden_states
            return self.norm(self.linear(hidden_states) * mod + hidden_states)
    
    # 创建类似 FLUX2 的结构
    double_blocks = nn.ModuleList([DoubleBlock(dim) for _ in range(num_double_blocks)])
    single_blocks = nn.ModuleList([SingleBlock(dim) for _ in range(num_single_blocks)])
    mod_embed = nn.Linear(dim, dim * 3)  # 生成 mod_img, mod_txt, single_mod
    
    double_blocks.to(device, dtype=dtype)
    single_blocks.to(device, dtype=dtype)
    mod_embed.to(device, dtype=dtype)
    
    init_state_double = {k: v.clone() for k, v in double_blocks.state_dict().items()}
    init_state_single = {k: v.clone() for k, v in single_blocks.state_dict().items()}
    init_state_mod = {k: v.clone() for k, v in mod_embed.state_dict().items()}
    
    torch.manual_seed(42)
    hidden_states_base = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    encoder_hidden_states_base = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    temb = torch.randn(batch_size, dim, dtype=dtype, device=device)
    rotary_emb = torch.randn(seq_len, dim, dtype=dtype, device=device)  # 简化的 rotary emb
    
    results = {}
    
    for use_ckpt in [False, True]:
        mode = "ckpt" if use_ckpt else "直接"
        print(f"\n>>> 测试: {mode}")
        
        double_blocks.load_state_dict(init_state_double)
        single_blocks.load_state_dict(init_state_single)
        mod_embed.load_state_dict(init_state_mod)
        double_blocks.zero_grad()
        single_blocks.zero_grad()
        mod_embed.zero_grad()
        
        hidden_states = hidden_states_base.clone().requires_grad_(True)
        encoder_hidden_states = encoder_hidden_states_base.clone()
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 计算 modulation 参数（和 FLUX2 一样在 forward 开始时计算）
            mods = mod_embed(temb)
            mod_img = mods[:, :dim].unsqueeze(1).expand(-1, seq_len, -1)
            mod_txt = mods[:, dim:2*dim].unsqueeze(1).expand(-1, seq_len, -1)
            single_mod = mods[:, 2*dim:].unsqueeze(1).expand(-1, seq_len, -1)
            
            # Double blocks
            for block in double_blocks:
                if use_ckpt:
                    encoder_hidden_states, hidden_states = checkpoint(
                        block, hidden_states, encoder_hidden_states, 
                        mod_img, mod_txt, rotary_emb,
                        use_reentrant=False
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states, encoder_hidden_states, mod_img, mod_txt, rotary_emb
                    )
            
            # Concat (like FLUX2)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            
            # 扩展 single_mod 以匹配 concat 后的序列长度
            # 使用 repeat 来复制，因为 expand 不能增加已有维度的大小
            single_mod_expanded = torch.cat([single_mod, single_mod], dim=1)  # [batch, 16, dim]
            
            # Single blocks
            for block in single_blocks:
                if use_ckpt:
                    hidden_states = checkpoint(
                        block, hidden_states, None, single_mod_expanded, rotary_emb, None,
                        use_reentrant=False
                    )
                else:
                    hidden_states = block(hidden_states, None, single_mod_expanded, rotary_emb, None)
        
        loss = hidden_states.float().mean()
        loss.backward()
        
        results[mode] = {
            'double_block': double_blocks[0].linear_img.weight.grad.float().norm().item(),
            'single_block': single_blocks[0].linear.weight.grad.float().norm().item(),
            'mod_embed': mod_embed.weight.grad.float().norm().item(),
        }
        
        print(f"  double_block[0] grad: {results[mode]['double_block']:.6e}")
        print(f"  single_block[0] grad: {results[mode]['single_block']:.6e}")
        print(f"  mod_embed grad: {results[mode]['mod_embed']:.6e}")
    
    # 比较
    print("\n比较:")
    for key in ['double_block', 'single_block', 'mod_embed']:
        ratio = results["ckpt"][key] / results["直接"][key] if results["直接"][key] > 0 else 0
        print(f"  {key} ratio: {ratio:.4f}")
        if abs(ratio - 1.0) > 0.1:
            print(f"    ⚠️ 异常！")


def test_shared_tensor_gradient():
    """测试共享 tensor 的梯度累积"""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    dim = 64
    batch_size = 2
    seq_len = 8
    num_blocks = 4
    
    print("\n" + "=" * 70)
    print("测试: 共享 tensor 梯度累积 (关键测试)")
    print("=" * 70)
    
    class BlockWithSharedInput(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
        
        def forward(self, x, shared_param):
            # shared_param 被所有 block 共享，且参与计算
            return self.linear(x) + shared_param
    
    blocks = nn.ModuleList([BlockWithSharedInput(dim) for _ in range(num_blocks)])
    shared_linear = nn.Linear(dim, dim)  # 生成共享参数的层
    
    blocks.to(device, dtype=dtype)
    shared_linear.to(device, dtype=dtype)
    
    init_blocks = {k: v.clone() for k, v in blocks.state_dict().items()}
    init_shared = {k: v.clone() for k, v in shared_linear.state_dict().items()}
    
    torch.manual_seed(42)
    x_base = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    temb = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    
    for use_ckpt in [False, True]:
        mode = "ckpt" if use_ckpt else "直接"
        print(f"\n>>> {mode}:")
        
        blocks.load_state_dict(init_blocks)
        shared_linear.load_state_dict(init_shared)
        blocks.zero_grad()
        shared_linear.zero_grad()
        
        x = x_base.clone().requires_grad_(True)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 计算共享参数 - 这个 tensor 会被传递给所有 block
            shared_param = shared_linear(temb)
            
            for block in blocks:
                if use_ckpt:
                    x = checkpoint(block, x, shared_param, use_reentrant=False)
                else:
                    x = block(x, shared_param)
        
        loss = x.float().mean()
        loss.backward()
        
        block_grad = blocks[0].linear.weight.grad.float().norm().item()
        shared_grad = shared_linear.weight.grad.float().norm().item()
        
        print(f"  block[0].linear grad: {block_grad:.6e}")
        print(f"  shared_linear grad: {shared_grad:.6e}")
        
        # 预期：如果 checkpoint 导致问题，shared_linear 的梯度会被累加 num_blocks 次
        if use_ckpt:
            print(f"  (如果 shared_linear grad 是直接的 {num_blocks} 倍，说明梯度被累加)")


if __name__ == "__main__":
    test_checkpoint_with_autocast()
    test_checkpoint_reentrant()
    test_multi_block_accumulation()
    test_modulation_params()
    test_flux2_like_structure()
    test_shared_tensor_gradient()

