"""
ZImage-Lite: 修复版 ZImage Transformer

修复内容:
1. 位置编码解耦: 图像从(0,0,0)开始，文本放到(512+pos, 511, 511)角落
2. 真正的 Padding: 零向量 padding + 正确的 attention mask
3. 文本常量 KV: 每层覆盖文本回 txt0，防止文本隐变量漂移
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import zimage_config
from .zimage_config import (
    ADALN_EMBED_DIM,
    FREQUENCY_EMBEDDING_SIZE,
    MAX_PERIOD,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
    TEXT_POS_OFFSET,
    TEXT_POS_H,
    TEXT_POS_W,
    SEQ_MULTI_OF,
)

logger = logging.getLogger(__name__)


def ceil_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=FREQUENCY_EMBEDDING_SIZE):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=MAX_PERIOD):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        w_f = self.weight.float()
        out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * w_f).to(x.dtype)


def clamp_fp16(x):
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x, apply_fp16_downscale=False):
        x3 = self.w3(x)
        if x.dtype == torch.float16 and apply_fp16_downscale:
            x3.div_(32)
        return self.w2(clamp_fp16(F.silu(self.w1(x)) * x3))

    def forward(self, x, apply_fp16_downscale=False):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, apply_fp16_downscale, use_reentrant=False)
        else:
            return self._forward(x, apply_fp16_downscale)


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True, eps: float = 1e-5, use_16bit: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.use_16bit = use_16bit

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_kv_heads, -1))
        value = value.unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype if not self.use_16bit else value.dtype
        query, key = query.to(dtype), key.to(dtype)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.n_kv_heads != self.n_heads:
            key = key.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            value = value.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask[:, None, None, :]

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).flatten(2)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)
        if self.use_16bit and output.dtype == torch.float16:
            output.div_(4)
        return output

    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, hidden_states, freqs_cis, attn_mask, use_reentrant=False)
        else:
            return self._forward(hidden_states, freqs_cis, attn_mask)


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        use_16bit: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps, use_16bit=use_16bit)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        self.feed_forward.enable_gradient_checkpointing()
        self.attention.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.feed_forward.disable_gradient_checkpointing()
        self.attention.disable_gradient_checkpointing()

    def _forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            del adaln_input
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs_cis=freqs_cis, attn_mask=attn_mask)
            del scale_msa
            x = x + gate_msa * self.attention_norm2(clamp_fp16(attn_out))
            del gate_msa
            x = x + gate_mlp * self.ffn_norm2(
                clamp_fp16(self.feed_forward(self.ffn_norm1(x) * scale_mlp, apply_fp16_downscale=True))
            )
            del scale_mlp, gate_mlp
        else:
            attn_out = self.attention(self.attention_norm1(x), freqs_cis=freqs_cis, attn_mask=attn_mask)
            x = x + self.attention_norm2(clamp_fp16(attn_out))
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, freqs_cis, adaln_input, attn_mask, use_reentrant=False)
        else:
            return self._forward(x, freqs_cis, adaln_input, attn_mask)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(self, theta: float, axes_dims: List[int], axes_lens: List[int]):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self._freqs_cis: Optional[torch.Tensor] = None

    def _build_freqs_cis(self, device: torch.device):
        all_freqs = []
        for ax_idx, (ax_dim, ax_len) in enumerate(zip(self.axes_dims, self.axes_lens)):
            half_dim = ax_dim // 2
            freqs = 1.0 / (self.theta ** (torch.arange(0, half_dim, device=device).float() / half_dim))
            indices = torch.arange(ax_len, device=device).float()
            freqs = torch.outer(indices, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            all_freqs.append(freqs_cis)
        return all_freqs

    def __call__(self, pos_ids: torch.Tensor) -> torch.Tensor:
        device = pos_ids.device
        if self._freqs_cis is None:
            self._freqs_cis = self._build_freqs_cis(device)
        
        all_freqs = self._freqs_cis
        if all_freqs[0].device != device:
            all_freqs = [f.to(device) for f in all_freqs]
            self._freqs_cis = all_freqs

        pos_ids = pos_ids.long()
        freqs_list = []
        for ax_idx in range(len(self.axes_dims)):
            ax_indices = pos_ids[..., ax_idx]
            ax_indices = ax_indices.clamp(0, self.axes_lens[ax_idx] - 1)
            freqs_list.append(all_freqs[ax_idx][ax_indices])
        
        freqs_cis = torch.cat(freqs_list, dim=-1)
        return freqs_cis


class ZImageLiteTransformer2DModel(nn.Module):
    """
    ZImage-Lite: 修复版 ZImage Transformer
    
    修复:
    1. 位置编码解耦 - 图像从(0,0,0)，文本放(512+pos, 511, 511)
    2. 零向量 padding + 正确 attention mask
    3. 文本常量 KV - 每层覆盖文本回 txt0
    """
    
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=ROPE_THETA,
        t_scale=1000.0,
        axes_dims=ROPE_AXES_DIMS,
        axes_lens=ROPE_AXES_LENS,
        use_16bit_for_attention: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.cap_feat_dim = cap_feat_dim
        self.rope_theta = rope_theta
        self.t_scale = t_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder
            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True, use_16bit=use_16bit_for_attention
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False, use_16bit=use_16bit_for_attention
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, use_16bit=use_16bit_for_attention)
                for layer_id in range(n_layers)
            ]
        )

        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = n_layers

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = cpu_offload
        for block in self.noise_refiner + self.context_refiner + self.layers:
            block.enable_gradient_checkpointing(activation_cpu_offloading=cpu_offload)
        print(f"ZImage-Lite: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        for block in self.noise_refiner + self.context_refiner + self.layers:
            block.disable_gradient_checkpointing()

    def cleanup_offloader(self):
        if self.offloader is not None:
            self.offloader.shutdown()
            self.offloader = None

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def create_image_position_ids(self, F_tokens: int, H_tokens: int, W_tokens: int, device: torch.device) -> torch.Tensor:
        """
        [修复1] 图像位置从 (0,0,0) 开始，与文本长度完全解耦
        """
        return self.create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens),
            start=(0, 0, 0),
            device=device
        ).flatten(0, 2)

    def create_caption_position_ids(self, txt_pad_len: int, device: torch.device) -> torch.Tensor:
        """
        [修复1] 文本放到角落: (TEXT_POS_OFFSET + pos, 511, 511)
        与图像位置不重叠，与文本真实长度无关
        """
        pos_ids = torch.zeros(txt_pad_len, 3, dtype=torch.int32, device=device)
        pos_ids[:, 0] = TEXT_POS_OFFSET + torch.arange(txt_pad_len, device=device)
        pos_ids[:, 1] = TEXT_POS_H
        pos_ids[:, 2] = TEXT_POS_W
        return pos_ids

    def patchify(self, x: torch.Tensor, patch_size: int, f_patch_size: int) -> torch.Tensor:
        pH = pW = patch_size
        pF = f_patch_size
        B, C, F_size, H_size, W_size = x.shape
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW

        x = x.view(B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
        x = x.reshape(B, F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return x

    def unpatchify(self, x: torch.Tensor, size: Tuple[int, int, int], patch_size: int, f_patch_size: int) -> torch.Tensor:
        pH = pW = patch_size
        pF = f_patch_size
        F_size, H_size, W_size = size
        B = x.shape[0]
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW
        ori_len = F_tokens * H_tokens * W_tokens

        x = x[:, :ori_len]
        x = x.view(B, F_tokens, H_tokens, W_tokens, pF, pH, pW, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, self.out_channels, F_size, H_size, W_size)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_ori_lens: Optional[List[int]] = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> torch.Tensor:
        """
        """
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        B, C, F_size, H_size, W_size = x.shape
        device = x.device

        t = t * self.t_scale
        adaln_input = self.t_embedder(t)

        pH = pW = patch_size
        pF = f_patch_size
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW
        img_seq_len = F_tokens * H_tokens * W_tokens
        img_pad_len = ceil_to_multiple(img_seq_len, SEQ_MULTI_OF)

        x = self.patchify(x, patch_size, f_patch_size)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        if img_pad_len > img_seq_len:
            img_padding = torch.zeros(B, img_pad_len - img_seq_len, self.dim, device=device, dtype=x.dtype)
            x = torch.cat([x, img_padding], dim=1)

        adaln_input = adaln_input.type_as(x)

        img_pos_ids = self.create_image_position_ids(F_tokens, H_tokens, W_tokens, device)
        if img_pad_len > img_seq_len:
            pad_pos = torch.zeros(img_pad_len - img_seq_len, 3, dtype=torch.int32, device=device)
            img_pos_ids = torch.cat([img_pos_ids, pad_pos], dim=0)
        img_freqs_cis = self.rope_embedder(img_pos_ids)
        img_freqs_cis = img_freqs_cis.unsqueeze(0).expand(B, -1, -1)

        img_mask = torch.zeros(B, img_pad_len, dtype=torch.bool, device=device)
        for i in range(B):
            img_mask[i, :img_seq_len] = True
        img_key_padding_mask = img_mask

        for layer in self.noise_refiner:
            x = layer(x, img_freqs_cis, adaln_input, attn_mask=img_key_padding_mask)

        if cap_ori_lens is None:
            nonzero = (cap_feats.float().abs().sum(dim=-1) > 0)
            inferred_lens = []
            for i in range(B):
                nz = nonzero[i].nonzero(as_tuple=False)
                inferred_lens.append(int(nz[-1].item() + 1) if nz.numel() > 0 else 0)
            cap_ori_lens = inferred_lens

        max_txt_supported = int(self.axes_lens[0] - TEXT_POS_OFFSET)
        capped_lens = [min(int(l), int(cap_feats.shape[1]), max_txt_supported) for l in cap_ori_lens]

        txt_pad_len = ceil_to_multiple(max(capped_lens) if capped_lens else 0, SEQ_MULTI_OF)
        txt_pad_len = min(txt_pad_len, max_txt_supported)
        
        cap_feats_padded = torch.zeros(B, txt_pad_len, cap_feats.shape[-1], device=device, dtype=cap_feats.dtype)
        for i in range(B):
            actual_len = min(capped_lens[i], txt_pad_len)
            cap_feats_padded[i, :actual_len] = cap_feats[i, :actual_len]

        txt0 = self.cap_embedder(cap_feats_padded)

        txt_mask = torch.zeros(B, txt_pad_len, dtype=torch.bool, device=device)
        for i, ori_len in enumerate(capped_lens):
            if ori_len > 0:
                txt_mask[i, :ori_len] = True
        txt_key_padding_mask = txt_mask

        txt_pos_ids = self.create_caption_position_ids(txt_pad_len, device)
        txt_freqs_cis = self.rope_embedder(txt_pos_ids)
        txt_freqs_cis = txt_freqs_cis.unsqueeze(0).expand(B, -1, -1)

        for layer in self.context_refiner:
            txt0 = layer(txt0, txt_freqs_cis, attn_mask=txt_key_padding_mask)

        unified = torch.cat([x, txt0], dim=1)
        unified_freqs_cis = torch.cat([img_freqs_cis, txt_freqs_cis], dim=1)
        
        unified_key_padding_mask = torch.cat([img_key_padding_mask, txt_key_padding_mask], dim=1)

        for layer in self.layers:
            unified = layer(unified, unified_freqs_cis, adaln_input, attn_mask=unified_key_padding_mask)
            
            img_out = unified[:, :img_pad_len]
            unified = torch.cat([img_out, txt0], dim=1)

        img_out = unified[:, :img_pad_len]

        img_out = self.all_final_layer[f"{patch_size}-{f_patch_size}"](img_out, adaln_input)

        output = self.unpatchify(img_out, (F_size, H_size, W_size), patch_size, f_patch_size)

        return output

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        import os
        import json
        from safetensors.torch import save_file
        
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            "all_patch_size": list(self.all_patch_size),
            "all_f_patch_size": list(self.all_f_patch_size),
            "in_channels": self.in_channels,
            "dim": self.dim,
            "n_layers": len(self.layers),
            "n_refiner_layers": len(self.noise_refiner),
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "norm_eps": self.norm_eps,
            "qk_norm": self.qk_norm,
            "cap_feat_dim": self.cap_feat_dim,
            "rope_theta": self.rope_theta,
            "t_scale": self.t_scale,
            "axes_dims": self.axes_dims,
            "axes_lens": self.axes_lens,
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        state_dict = self.state_dict()
        if safe_serialization:
            save_file(state_dict, os.path.join(save_directory, "model.safetensors"))
        else:
            torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))


def load_zimage_transformer(
    transformer_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> ZImageLiteTransformer2DModel:
    import os
    import json
    from safetensors.torch import load_file
    
    config_path = os.path.join(transformer_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    model = ZImageLiteTransformer2DModel(
        all_patch_size=tuple(config.get("all_patch_size", zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE)),
        all_f_patch_size=tuple(config.get("all_f_patch_size", zimage_config.DEFAULT_TRANSFORMER_F_PATCH_SIZE)),
        in_channels=config.get("in_channels", zimage_config.DEFAULT_TRANSFORMER_IN_CHANNELS),
        dim=config.get("dim", zimage_config.DEFAULT_TRANSFORMER_DIM),
        n_layers=config.get("n_layers", zimage_config.DEFAULT_TRANSFORMER_N_LAYERS),
        n_refiner_layers=config.get("n_refiner_layers", zimage_config.DEFAULT_TRANSFORMER_N_REFINER_LAYERS),
        n_heads=config.get("n_heads", zimage_config.DEFAULT_TRANSFORMER_N_HEADS),
        n_kv_heads=config.get("n_kv_heads", zimage_config.DEFAULT_TRANSFORMER_N_KV_HEADS),
        norm_eps=config.get("norm_eps", zimage_config.DEFAULT_TRANSFORMER_NORM_EPS),
        qk_norm=config.get("qk_norm", zimage_config.DEFAULT_TRANSFORMER_QK_NORM),
        cap_feat_dim=config.get("cap_feat_dim", zimage_config.DEFAULT_TRANSFORMER_CAP_FEAT_DIM),
        rope_theta=config.get("rope_theta", zimage_config.ROPE_THETA),
        t_scale=config.get("t_scale", zimage_config.DEFAULT_TRANSFORMER_T_SCALE),
        axes_dims=config.get("axes_dims", zimage_config.ROPE_AXES_DIMS),
        axes_lens=config.get("axes_lens", zimage_config.ROPE_AXES_LENS),
    )
    
    index_file = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = set(weight_map.values())
        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(transformer_path, shard_file)
            if os.path.exists(shard_path):
                shard_dict = load_file(shard_path, device="cpu")
                state_dict.update(shard_dict)
        print(f"[ZImage-Lite] 从 {len(shard_files)} 个分片文件加载权重")
    else:
        candidate_files = [
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
        ]
        state_dict = None
        for fname in candidate_files:
            p = os.path.join(transformer_path, fname)
            if os.path.exists(p):
                if p.endswith(".safetensors"):
                    state_dict = load_file(p, device="cpu")
                else:
                    state_dict = torch.load(p, map_location="cpu")
                break
        if state_dict is None:
            raise FileNotFoundError(f"未找到权重文件: {transformer_path}")
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device=device, dtype=dtype)
    
    return model


def load_zimage_lite_from_original(
    original_model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> ZImageLiteTransformer2DModel:
    return load_zimage_transformer(original_model_path, device=device, dtype=dtype)

