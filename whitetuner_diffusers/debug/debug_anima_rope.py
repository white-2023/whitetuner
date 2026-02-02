import torch
from einops import rearrange

def comfy_apply_rope(t, freqs):
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out

class ComfyRopeEmb(torch.nn.Module):
    def __init__(self, head_dim, len_h, len_w, len_t, 
                 h_extrapolation_ratio=1.0, w_extrapolation_ratio=1.0, t_extrapolation_ratio=1.0):
        super().__init__()
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        print(f"ComfyUI dim split: dim_h={dim_h}, dim_w={dim_w}, dim_t={dim_t}, total={dim_h+dim_w+dim_t}")
        
        self.register_buffer("dim_spatial_range", torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h, persistent=False)
        self.register_buffer("dim_temporal_range", torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t, persistent=False)
        
        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2)) if dim_h > 2 else 1.0
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2)) if dim_w > 2 else 1.0
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2)) if dim_t > 2 else 1.0
        
    def forward(self, x_B_T_H_W_C, fps=None, device=None):
        B, T, H, W, _ = x_B_T_H_W_C.shape
        if device is None:
            device = x_B_T_H_W_C.device
            
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor
        
        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))
        
        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
        half_emb_h = torch.outer(seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W], w_spatial_freqs)
        half_emb_t = torch.outer(seq[:T], temporal_freqs)
        
        print(f"  half_emb_h: {half_emb_h.shape}, half_emb_w: {half_emb_w.shape}, half_emb_t: {half_emb_t.shape}")
        
        half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
        half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
        half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)
        
        print(f"  After stack: half_emb_h: {half_emb_h.shape}")
        
        em_T_H_W_D = torch.cat([
            half_emb_t.view(T, 1, 1, -1, 4).expand(T, H, W, -1, 4),
            half_emb_h.view(1, H, 1, -1, 4).expand(T, H, W, -1, 4),
            half_emb_w.view(1, 1, W, -1, 4).expand(T, H, W, -1, 4),
        ], dim=-2)
        
        print(f"  em_T_H_W_D: {em_T_H_W_D.shape}")
        
        result = rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()
        print(f"  final rope_emb: {result.shape}")
        return result

print("=== ComfyUI RoPE Embedding Test ===")

head_dim = 128
n_heads = 16
resolution = 1024
patch_spatial = 2
latent_size = resolution // 8
patched_H = latent_size // patch_spatial
patched_W = latent_size // patch_spatial

B, T, H, W = 1, 1, patched_H, patched_W
print(f"Resolution: {resolution}, latent_size: {latent_size}, patched: {patched_H}x{patched_W}")
print(f"B={B}, T={T}, H={H}, W={W}")

comfy_rope = ComfyRopeEmb(
    head_dim=head_dim,
    len_h=120,
    len_w=120,
    len_t=128,
)

x_dummy = torch.randn(B, T, H, W, n_heads * head_dim)
rope_emb = comfy_rope(x_dummy, fps=None, device="cpu")
print(f"rope_emb shape: {rope_emb.shape}")

rope_emb_for_block = rope_emb.unsqueeze(1).unsqueeze(0)
print(f"After unsqueeze(1).unsqueeze(0): {rope_emb_for_block.shape}")

q = torch.randn(B, T*H*W, n_heads, head_dim)
print(f"\nq shape before rope: {q.shape}")

print("\n=== Step-by-step apply_rotary_pos_emb ===")
t = q
freqs = rope_emb_for_block
print(f"t shape: {t.shape}")
print(f"freqs shape: {freqs.shape}")

t_ = t.reshape(*t.shape[:-1], 2, -1)
print(f"After reshape(*t.shape[:-1], 2, -1): {t_.shape}")

t_ = t_.movedim(-2, -1)
print(f"After movedim(-2, -1): {t_.shape}")

t_ = t_.unsqueeze(-2)
print(f"After unsqueeze(-2): {t_.shape}")

t_ = t_.float()
print(f"t_ final shape: {t_.shape}")

print(f"\nfreqs[..., 0] shape: {freqs[..., 0].shape}")
print(f"freqs[..., 1] shape: {freqs[..., 1].shape}")
print(f"t_[..., 0] shape: {t_[..., 0].shape}")
print(f"t_[..., 1] shape: {t_[..., 1].shape}")

try:
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    print(f"t_out shape: {t_out.shape}")
except Exception as e:
    print(f"ERROR: {e}")
    print("\nDimension mismatch analysis:")
    print(f"  freqs[..., 0] needs to broadcast with t_[..., 0]")
    print(f"  freqs: {freqs.shape} -> [..., 0]: {freqs[..., 0].shape}")
    print(f"  t_: {t_.shape} -> [..., 0]: {t_[..., 0].shape}")

print("\n\n=== Test My Implementation ===")
import sys
sys.path.insert(0, r"D:\ai\whitetuner\whitetuner_diffusers")
from anima_modules.model import VideoRopePosition3DEmb as MyRopeEmb, apply_rotary_pos_emb_cosmos as my_apply_rope

my_rope = MyRopeEmb(
    model_channels=n_heads * head_dim,
    len_h=120,
    len_w=120,
    len_t=128,
    head_dim=head_dim,
)

x_dummy = torch.randn(B, T, H, W, n_heads * head_dim)
my_rope_emb = my_rope(x_dummy, fps=None, device="cpu")
print(f"My rope_emb shape: {my_rope_emb.shape}")

my_rope_emb_for_block = my_rope_emb.unsqueeze(1).unsqueeze(0)
print(f"After unsqueeze(1).unsqueeze(0): {my_rope_emb_for_block.shape}")

print("\n=== Step-by-step my apply_rotary_pos_emb ===")
t = q
my_freqs = my_rope_emb_for_block
print(f"t shape: {t.shape}")
print(f"my_freqs shape: {my_freqs.shape}")

t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
print(f"t_ final shape: {t_.shape}")

print(f"\nmy_freqs[..., 0] shape: {my_freqs[..., 0].shape}")
print(f"t_[..., 0] shape: {t_[..., 0].shape}")

try:
    my_t_out = my_freqs[..., 0] * t_[..., 0] + my_freqs[..., 1] * t_[..., 1]
    print(f"my_t_out shape: {my_t_out.shape}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR in my implementation: {e}")

print("\n\n=== Test Full Anima Forward ===")
from anima_modules.model import Anima, load_anima_model

model = Anima(
    in_channels=16,
    out_channels=16,
    model_channels=2048,
    num_blocks=2,
    num_heads=16,
    use_adaln_lora=True,
)
model.eval()

latent = torch.randn(1, 16, 1, latent_size, latent_size)
timesteps = torch.tensor([0.5])
context = torch.randn(1, 512, 1024)

print(f"latent shape: {latent.shape}")
print(f"timesteps shape: {timesteps.shape}")
print(f"context shape: {context.shape}")

try:
    with torch.no_grad():
        output = model(latent, timesteps, context)
    print(f"output shape: {output.shape}")
    print("FULL FORWARD SUCCESS!")
except Exception as e:
    print(f"ERROR in full forward: {e}")
    import traceback
    traceback.print_exc()

print("\n\n=== Test Load Real Model ===")
dit_path = r"F:\models\circlestone-labs-Anima\split_files\diffusion_models\anima-preview.safetensors"
try:
    real_model = load_anima_model(dit_path, device="cpu", dtype=torch.float32)
    print(f"Model loaded successfully")
    print(f"  model_channels: {real_model.model_channels}")
    print(f"  num_heads: {real_model.num_heads}")
    print(f"  num_blocks: {real_model.num_blocks}")
    print(f"  pos_embedder head_dim: {real_model.pos_embedder.head_dim}")
    
    real_model.eval()
    
    latent = torch.randn(1, 16, 1, latent_size, latent_size)
    timesteps = torch.tensor([0.5])
    context = torch.randn(1, 512, 1024)
    
    print(f"\nTrying forward pass...")
    with torch.no_grad():
        output = real_model(latent, timesteps, context)
    print(f"output shape: {output.shape}")
    print("REAL MODEL FORWARD SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n\n=== Check pos_embedder buffers ===")
print(f"dim_spatial_range shape: {real_model.pos_embedder.dim_spatial_range.shape}")
print(f"dim_temporal_range shape: {real_model.pos_embedder.dim_temporal_range.shape}")
print(f"Expected: dim_spatial_range=(21,), dim_temporal_range=(22,) for head_dim=128")

print("\n\n=== Test VAE Encoding ===")
from anima_modules.vae import load_anima_vae

vae_path = r"F:\models\circlestone-labs-Anima\split_files\vae\qwen_image_vae.safetensors"
try:
    vae = load_anima_vae(vae_path, device="cpu", dtype=torch.float32)
    
    test_img = torch.randn(3, 1, resolution, resolution)
    print(f"Test image shape: {test_img.shape}")
    
    latent = vae.encode([test_img])[0]
    print(f"VAE encoded latent shape: {latent.shape}")
    
    latents = torch.stack([latent], dim=0)
    print(f"After torch.stack (batch=1): {latents.shape}")
    
    print(f"\nExpected for model input: (B, C=16, T=1, H={latent_size}, W={latent_size})")
    print(f"Actual shape: {latents.shape}")
    
    if latents.shape == (1, 16, 1, latent_size, latent_size):
        print("Shape matches!")
    else:
        print(f"SHAPE MISMATCH! Need to fix VAE encoding")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
