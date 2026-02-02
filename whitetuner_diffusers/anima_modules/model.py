"""
Anima Model - 基于 Cosmos Predict2 的文生图模型

简化的独立实现，用于训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import logging

from safetensors.torch import load_file


def apply_rotary_pos_emb_cosmos(t: Tensor, freqs: Tensor) -> Tensor:
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_adapter(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


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
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

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
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False):
        super().__init__()
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm_self_attn = RMSNorm(model_dim, eps=1e-6)
            self.self_attn = AdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim//num_heads,
            )

        self.norm_cross_attn = RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = AdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim//num_heads,
        )

        self.norm_mlp = RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim)
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
        ):
        super().__init__()

        self.embed = nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim//num_heads)
        self.blocks = nn.ModuleList([
            AdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = RMSNorm(target_dim, eps=1e-6)

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
        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context, target_attention_mask=target_attention_mask, source_attention_mask=source_attention_mask, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class CosmosAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
    ):
        super().__init__()
        self.is_selfattn = context_dim is None

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def compute_qkv(self, x: Tensor, context: Optional[Tensor] = None, rope_emb: Optional[Tensor] = None):
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
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb_cosmos(q, rope_emb)
            k = apply_rotary_pos_emb_cosmos(k, rope_emb)
        return q, k, v

    def compute_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = rearrange(q, "b ... h d -> b h ... d")
        k = rearrange(k, "b ... h d -> b h ... d")
        v = rearrange(v, "b ... h d -> b h ... d")
        
        in_q_shape = q.shape
        q = q.view(in_q_shape[0], in_q_shape[1], -1, in_q_shape[-1])
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[-1])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[-1])
        
        result = F.scaled_dot_product_attention(q, k, v)
        result = result.transpose(1, 2).reshape(in_q_shape[0], -1, self.n_heads * self.head_dim)
        return self.output_proj(result)

    def forward(self, x: Tensor, context: Optional[Tensor] = None, rope_emb: Optional[Tensor] = None) -> Tensor:
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: Tensor) -> Tensor:
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
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

    def forward(self, sample: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
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
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
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
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

    def forward(self, x_B_T_H_W_D: Tensor, emb_B_T_D: Tensor, adaln_lora_B_T_3D: Optional[Tensor] = None):
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


class CosmosBlock(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = CosmosAttention(x_dim, None, num_heads, x_dim // num_heads)

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CosmosAttention(x_dim, context_dim, num_heads, x_dim // num_heads)

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def forward(
        self,
        x_B_T_H_W_D: Tensor,
        emb_B_T_D: Tensor,
        crossattn_emb: Tensor,
        rope_emb_L_1_1_D: Optional[Tensor] = None,
        adaln_lora_B_T_3D: Optional[Tensor] = None,
        extra_per_block_pos_emb: Optional[Tensor] = None,
    ) -> Tensor:
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
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
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

        normalized_x = self.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_self_attn_B_T_1_1_D) + shift_self_attn_B_T_1_1_D
        result = rearrange(
            self.self_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result

        normalized_x = self.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + scale_cross_attn_B_T_1_1_D) + shift_cross_attn_B_T_1_1_D
        result = rearrange(
            self.cross_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T, h=H, w=W,
        )
        x_B_T_H_W_D = result * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x = self.layer_norm_mlp(x_B_T_H_W_D) * (1 + scale_mlp_B_T_1_1_D) + shift_mlp_B_T_1_1_D
        result = self.mlp(normalized_x)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result

        return x_B_T_H_W_D


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        max_fps: int = 30,
        min_fps: int = 1,
        is_learnable: bool = False,
        head_dim: int = 64,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        interpolation: str = "crop",
        enable_fps_modulation: bool = True,
        device=None,
    ):
        super().__init__()
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t
        self.head_dim = head_dim
        self.h_extrapolation_ratio = h_extrapolation_ratio
        self.w_extrapolation_ratio = w_extrapolation_ratio
        self.t_extrapolation_ratio = t_extrapolation_ratio
        self.base_fps = max_fps
        self.interpolation = interpolation
        self.enable_fps_modulation = enable_fps_modulation

        h_dim = head_dim // 4
        w_dim = head_dim // 4
        t_dim = head_dim - h_dim - w_dim

        freqs_h = self._precompute_freqs(h_dim, len_h, h_extrapolation_ratio, device)
        freqs_w = self._precompute_freqs(w_dim, len_w, w_extrapolation_ratio, device)
        freqs_t = self._precompute_freqs(t_dim, len_t, t_extrapolation_ratio, device)

        self.register_buffer("freqs_h", freqs_h, persistent=False)
        self.register_buffer("freqs_w", freqs_w, persistent=False)
        self.register_buffer("freqs_t", freqs_t, persistent=False)

    def _precompute_freqs(self, dim: int, length: int, extrapolation_ratio: float, device=None) -> Tensor:
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs = freqs / extrapolation_ratio
        t = torch.arange(length, device=device).float()
        freqs = torch.outer(t, freqs)
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    def forward(self, x: Tensor, fps: Optional[Tensor] = None, device=None) -> Tensor:
        B, T, H, W, D = x.shape
        
        freqs_h = self.freqs_h[:H].to(device)
        freqs_w = self.freqs_w[:W].to(device)
        freqs_t = self.freqs_t[:T].to(device)

        freqs_h = freqs_h.view(1, 1, H, 1, -1, 2).expand(1, T, H, W, -1, 2)
        freqs_w = freqs_w.view(1, 1, 1, W, -1, 2).expand(1, T, H, W, -1, 2)
        freqs_t = freqs_t.view(1, T, 1, 1, -1, 2).expand(1, T, H, W, -1, 2)

        freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-2)
        return freqs


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
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 32,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "rope3d",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.dtype = dtype
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
        self.use_adaln_lora = use_adaln_lora

        self.pos_embedder = VideoRopePosition3DEmb(
            model_channels=model_channels,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            head_dim=model_channels // num_heads,
            device=device,
        )

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        in_ch = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_ch,
            out_channels=model_channels,
        )

        self.blocks = nn.ModuleList([
            CosmosBlock(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_adaln_lora=use_adaln_lora,
                adaln_lora_dim=adaln_lora_dim,
            )
            for _ in range(num_blocks)
        ])

        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
        
        self.llm_adapter = LLMAdapter()

    def preprocess_text_embeds(self, text_embeds, text_ids):
        if text_ids is not None:
            return self.llm_adapter(text_embeds, text_ids)
        else:
            return text_embeds

    def _pad_to_patch_size(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape
        pad_t = (self.patch_temporal - T % self.patch_temporal) % self.patch_temporal
        pad_h = (self.patch_spatial - H % self.patch_spatial) % self.patch_spatial
        pad_w = (self.patch_spatial - W % self.patch_spatial) % self.patch_spatial
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        return x

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        fps: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        orig_shape = list(x.shape)
        x = self._pad_to_patch_size(x)
        
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(x.shape[0], 1, x.shape[3], x.shape[4], dtype=x.dtype, device=x.device)
            padding_mask = padding_mask.unsqueeze(1).repeat(1, 1, x.shape[2], 1, 1)
            x = torch.cat([x, padding_mask], dim=1)

        x_B_T_H_W_D = self.x_embedder(x)
        
        B, T, H, W, D = x_B_T_H_W_D.shape
        rope_emb = self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x.device)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_embedding, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype))
        t_embedding = self.t_embedding_norm(t_embedding)

        rope_emb = rope_emb.unsqueeze(1).unsqueeze(0)

        for block in self.blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding,
                context,
                rope_emb_L_1_1_D=rope_emb,
                adaln_lora_B_T_3D=adaln_lora,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding, adaln_lora_B_T_3D=adaln_lora)

        x_out = rearrange(
            x_B_T_H_W_O,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_out[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]

    def enable_gradient_checkpointing(self):
        for block in self.blocks:
            block._gradient_checkpointing = True


def load_anima_model(
    dit_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Anima:
    print(f"Loading Anima model from {dit_path}")
    
    state_dict = load_file(dit_path)
    
    in_channels = (state_dict['x_embedder.proj.1.weight'].shape[1] // 4) - 1
    model_channels = state_dict['x_embedder.proj.1.weight'].shape[0]
    num_blocks = sum(1 for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.self_attn.q_proj.weight'))
    
    print(f"  - in_channels: {in_channels}")
    print(f"  - model_channels: {model_channels}")
    print(f"  - num_blocks: {num_blocks}")
    
    model = Anima(
        in_channels=in_channels,
        out_channels=16,
        model_channels=model_channels,
        num_blocks=num_blocks,
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
