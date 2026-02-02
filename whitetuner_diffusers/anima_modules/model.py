import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import logging
from typing import Optional, Tuple
import math
import torch.nn.functional as F

from safetensors.torch import load_file


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


def pad_to_patch_size(x, patch_size):
    T, H, W = patch_size
    _, _, t, h, w = x.shape
    pad_t = (T - t % T) % T
    pad_h = (H - h % H) % H
    pad_w = (W - w % W) % W
    if pad_t > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
    return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight.to(x.dtype)


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)

        return q, k, v

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        in_q_shape = q.shape
        in_k_shape = k.shape
        q = rearrange(q, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
        k = rearrange(k, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
        v = rearrange(v, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
        result = F.scaled_dot_product_attention(q, k, v)
        result = result.transpose(1, 2).reshape(in_q_shape[0], -1, self.n_heads * self.head_dim)
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False, device=None, dtype=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora, device=device, dtype=dtype)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False, device=device, dtype=dtype)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


class PatchEmbed(nn.Module):
    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
        device=None, dtype=None,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False, device=device, dtype=dtype
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device=None, dtype=None,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False, device=device, dtype=dtype
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False, device=device, dtype=dtype)
            )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale_B_T_1_1_D = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")

        x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


class Block(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads, device=device, dtype=dtype)

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = Attention(
            x_dim, context_dim, num_heads, x_dim // num_heads, device=device, dtype=dtype
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), device=device, dtype=dtype)

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype))

    def _forward_impl(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        normalized_x_B_T_H_W_D = self.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_self_attn_B_T_1_1_D) + shift_self_attn_B_T_1_1_D
        result_B_T_H_W_D = rearrange(
            self.self_attn(
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        normalized_x_B_T_H_W_D = self.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + scale_cross_attn_B_T_1_1_D) + shift_cross_attn_B_T_1_1_D
        result_B_T_H_W_D = rearrange(
            self.cross_attn(
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = self.layer_norm_mlp(x_B_T_H_W_D) * (1 + scale_mlp_B_T_1_1_D) + shift_mlp_B_T_1_1_D
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_per_block_pos_emb,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_per_block_pos_emb,
            )


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def forward(self, x_B_T_H_W_D, fps=None, device=None):
        B, T, H, W, _ = x_B_T_H_W_D.shape
        if device is None:
            device = x_B_T_H_W_D.device

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
        half_emb_h = torch.outer(seq[:H].to(device=device), h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W].to(device=device), w_spatial_freqs)

        if fps is None or self.enable_fps_modulation is False:
            half_emb_t = torch.outer(seq[:T].to(device=device), temporal_freqs)
        else:
            half_emb_t = torch.outer(seq[:T].to(device=device) / fps * self.base_fps, temporal_freqs)

        half_emb_h = torch.stack([torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1)
        half_emb_w = torch.stack([torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1)
        half_emb_t = torch.stack([torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )

        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_adapter(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class AdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim, device=None, dtype=None):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=1e-6, device=device, dtype=dtype)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)

        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb_adapter(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb_adapter(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class AdapterTransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False, device=None, dtype=None):
        super().__init__()
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm_self_attn = RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
            self.self_attn = AdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim//num_heads,
                device=device,
                dtype=dtype,
            )

        self.norm_cross_attn = RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = AdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim//num_heads,
            device=device,
            dtype=dtype,
        )

        self.norm_mlp = RMSNorm(model_dim, eps=1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim, device=device, dtype=dtype)
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(normed, mask=target_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, mask=source_attention_mask, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    def __init__(
            self,
            source_dim=1024,
            target_dim=1024,
            model_dim=1024,
            num_layers=6,
            num_heads=16,
            use_self_attn=True,
            device=None,
            dtype=None,
        ):
        super().__init__()

        self.embed = nn.Embedding(32128, target_dim, device=device, dtype=dtype)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim, device=device, dtype=dtype)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim//num_heads)
        self.blocks = nn.ModuleList([
            AdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn, device=device, dtype=dtype) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim, device=device, dtype=dtype)
        self.norm = RMSNorm(target_dim, eps=1e-6, device=device, dtype=dtype)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states.to(x.dtype)
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context, target_attention_mask=target_attention_mask, source_attention_mask=source_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


class Anima(nn.Module):
    def __init__(
        self,
        max_img_h: int = 240,
        max_img_w: int = 240,
        max_frames: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "rope3d",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        max_fps: int = 30,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.blocks_to_swap = None
        self.offloader = None
        self.gradient_checkpointing = False
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=model_channels // num_heads,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            base_fps=max_fps,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_t_extrapolation_ratio,
            device=device,
        )

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora, device=device, dtype=dtype),
        )

        in_ch = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_ch,
            out_channels=model_channels,
            device=device, dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    device=device, dtype=dtype,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            device=device, dtype=dtype,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6, device=device, dtype=dtype)
        
        self.llm_adapter = LLMAdapter(device=device, dtype=dtype)

    def preprocess_text_embeds(self, text_embeds, text_ids):
        if text_ids is not None:
            return self.llm_adapter(text_embeds, text_ids)
        else:
            return text_embeds

    def enable_block_swap(self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        from wan_modules.offloading_utils import ModelOffloader
        
        self.blocks_to_swap = blocks_to_swap
        num_blocks = len(self.blocks)

        assert (
            self.blocks_to_swap <= num_blocks - 1
        ), f"Cannot swap more than {num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."

        self.offloader = ModelOffloader(
            "anima_attn_block", self.blocks, num_blocks, self.blocks_to_swap, supports_backward, device, use_pinned_memory
        )
        print(
            f"Anima: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {num_blocks} blocks. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print(f"Anima: Block swap set to forward only")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print(f"Anima: Block swap set to forward and backward")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def cleanup_offloader(self):
        if self.offloader is not None:
            self.offloader.shutdown()
            self.offloader = None

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for block in self.blocks:
            block.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        for block in self.blocks:
            block.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not hasattr(self, '_debug_forward_count'):
            self._debug_forward_count = 0
        self._debug_forward_count += 1
        do_debug = self._debug_forward_count <= 2
        
        orig_shape = list(x.shape)
        x = pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
        
        if do_debug:
            print(f"[DEBUG Anima.forward] call={self._debug_forward_count}")
            print(f"  input x: shape={x.shape}, mean={x.float().mean().item():.4f}, std={x.float().std().item():.4f}")
            print(f"  timesteps: {timesteps.float().mean().item():.6f}")
            print(f"  context: shape={context.shape}, mean={context.float().mean().item():.4f}, std={context.float().std().item():.4f}")
        
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(x.shape[0], 1, x.shape[3], x.shape[4], dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding_mask.unsqueeze(1).repeat(1, 1, x.shape[2], 1, 1)], dim=1)

        x_B_T_H_W_D = self.x_embedder(x)
        
        if do_debug:
            print(f"  x_B_T_H_W_D after patch_embed: shape={x_B_T_H_W_D.shape}, mean={x_B_T_H_W_D.float().mean().item():.4f}, std={x_B_T_H_W_D.float().std().item():.4f}")
        
        rope_emb = self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x.device)
        
        if do_debug:
            print(f"  rope_emb: shape={rope_emb.shape}, mean={rope_emb.float().mean().item():.4f}, std={rope_emb.float().std().item():.4f}")

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_embedding, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype))
        t_embedding = self.t_embedding_norm(t_embedding)
        
        if do_debug:
            print(f"  t_embedding: shape={t_embedding.shape}, mean={t_embedding.float().mean().item():.4f}, std={t_embedding.float().std().item():.4f}")
            if adaln_lora is not None:
                print(f"  adaln_lora: shape={adaln_lora.shape}, mean={adaln_lora.float().mean().item():.4f}, std={adaln_lora.float().std().item():.4f}")

        rope_emb = rope_emb.unsqueeze(1).unsqueeze(0)
        
        if do_debug:
            print(f"  rope_emb after unsqueeze: shape={rope_emb.shape}")

        input_device = x_B_T_H_W_D.device
        for i, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(i)
            
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding,
                context,
                rope_emb_L_1_1_D=rope_emb,
                adaln_lora_B_T_3D=adaln_lora,
            )
            
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, i)
            
            if do_debug and i == 0:
                print(f"  after block 0: mean={x_B_T_H_W_D.float().mean().item():.4f}, std={x_B_T_H_W_D.float().std().item():.4f}")

        if self.blocks_to_swap and x_B_T_H_W_D.device != input_device:
            x_B_T_H_W_D = x_B_T_H_W_D.to(input_device)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding, adaln_lora_B_T_3D=adaln_lora)

        x_out = rearrange(
            x_B_T_H_W_O,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        
        if do_debug:
            print(f"  final output: shape={x_out.shape}, mean={x_out.float().mean().item():.4f}, std={x_out.float().std().item():.4f}")
        
        return x_out[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]


def load_anima_model(
    dit_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Anima:
    print(f"Loading Anima model from {dit_path}")
    
    state_dict = load_file(dit_path)
    
    if any(k.startswith('net.') for k in state_dict.keys()):
        state_dict = {k.replace('net.', '', 1): v for k, v in state_dict.items()}
        print("  - Removed 'net.' prefix from keys")
    
    in_channels = (state_dict['x_embedder.proj.1.weight'].shape[1] // 4) - 1
    model_channels = state_dict['x_embedder.proj.1.weight'].shape[0]
    num_blocks = sum(1 for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.self_attn.q_proj.weight'))
    
    if model_channels == 2048:
        num_heads = 16
    elif model_channels == 5120:
        num_heads = 40
    else:
        num_heads = model_channels // 128
    
    use_adaln_lora = 'blocks.0.adaln_modulation_self_attn.2.weight' in state_dict
    adaln_lora_dim = 256 if use_adaln_lora else 256
    
    print(f"  - in_channels: {in_channels}")
    print(f"  - model_channels: {model_channels}")
    print(f"  - num_blocks: {num_blocks}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - use_adaln_lora: {use_adaln_lora}")
    
    model = Anima(
        in_channels=in_channels,
        out_channels=16,
        model_channels=model_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        use_adaln_lora=use_adaln_lora,
        adaln_lora_dim=adaln_lora_dim,
        dtype=dtype,
        device=device,
    )
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  - Missing keys: {len(missing)}")
    if unexpected:
        print(f"  - Unexpected keys: {len(unexpected)}")
    
    model = model.to(device=device, dtype=dtype)
    print(f"Anima model loaded successfully")
    
    return model
