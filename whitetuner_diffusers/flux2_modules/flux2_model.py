import os
import math
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file

from wan_modules.offloading_utils import ModelOffloader
from wan_modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, use_real=False, repeat_interleave_real=True, freqs_dtype=torch.float32):
    
    if isinstance(pos, int):
        pos = torch.arange(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    freqs = torch.outer(pos.float(), freqs)
    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
        return freqs_cos, freqs_sin
    elif use_real:
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def apply_rotary_emb(x, freqs, sequence_dim=1):
    cos, sin = freqs
    if sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    cos, sin = cos.to(x.device), sin.to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


def to_device(obj, device, non_blocking=False):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, (tuple, list)):
        return type(obj)(to_device(o, device, non_blocking) for o in obj)
    elif isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
    return obj


def to_cpu(obj):
    return to_device(obj, torch.device("cpu"))


_cpu_offload_debug = False

def create_cpu_offloading_wrapper(func, device):
    def wrapper(orig_func):
        def custom_forward(*inputs):
            if _cpu_offload_debug:
                print(f"[CPU_OFFLOAD] inputs[0] device={inputs[0].device}, dtype={inputs[0].dtype}")
                print(f"[CPU_OFFLOAD] autocast enabled={torch.is_autocast_enabled()}")
            cuda_inputs = to_device(inputs, device, non_blocking=True)
            if _cpu_offload_debug:
                print(f"[CPU_OFFLOAD] cuda_inputs[0] device={cuda_inputs[0].device}, dtype={cuda_inputs[0].dtype}")
            outputs = orig_func(*cuda_inputs)
            if _cpu_offload_debug:
                if isinstance(outputs, tuple):
                    print(f"[CPU_OFFLOAD] outputs[0] device={outputs[0].device}, dtype={outputs[0].dtype}, std={outputs[0].std():.4f}")
                else:
                    print(f"[CPU_OFFLOAD] outputs device={outputs.device}, dtype={outputs.dtype}, std={outputs.std():.4f}")
            cpu_outputs = to_cpu(outputs)
            if _cpu_offload_debug:
                if isinstance(cpu_outputs, tuple):
                    print(f"[CPU_OFFLOAD] cpu_outputs[0] device={cpu_outputs[0].device}, dtype={cpu_outputs[0].dtype}, std={cpu_outputs[0].std():.4f}")
                else:
                    print(f"[CPU_OFFLOAD] cpu_outputs device={cpu_outputs.device}, dtype={cpu_outputs.dtype}, std={cpu_outputs.std():.4f}")
            return cpu_outputs
        return custom_forward
    return wrapper(func)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, sample_proj_bias: bool = True):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=sample_proj_bias)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, elementwise_affine: bool = True, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = emb.chunk(2, dim=-1)
        if scale.ndim == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class Flux2SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x = self.gate_fn(x1) * x2
        return x


class Flux2FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: float = 3.0, inner_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class Flux2AttnProcessor:
    """
    Attention processor for Flux2, aligned with official diffusers implementation.
    """
    def __init__(self):
        pass

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 保存输入 dtype 用于后续恢复
        input_dtype = hidden_states.dtype
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # QK Norm - 官方实现直接调用，不做额外 dtype 转换
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Transpose for attention: (B, seq, heads, head_dim) -> (B, heads, seq, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)
        hidden_states = hidden_states.transpose(1, 2).flatten(-2)
        
        # 官方实现: hidden_states = hidden_states.to(query.dtype)
        # 转换为输入的 dtype，确保与后续层兼容
        hidden_states = hidden_states.to(input_dtype)

        if attn.added_kv_proj_dim is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states[:, :encoder_seq_len], hidden_states[:, encoder_seq_len:]
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Flux2Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.use_bias = bias
        self.dropout = dropout
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        self.processor = processor if processor is not None else Flux2AttnProcessor()

    def set_processor(self, processor):
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Flux2ParallelSelfAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Parallel in (QKV + MLP in) projection
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # QK Norm - 官方实现直接调用
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Transpose for attention: (B, seq, heads, head_dim) -> (B, heads, seq, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        
        # 官方实现: hidden_states = hidden_states.to(query.dtype)
        attn_output = attn_output.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = attn.act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


class Flux2ParallelSelfAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: int = None,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        mlp_ratio: float = 3.0,
        mlp_mult_factor: int = 2,
        processor=None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        self.to_qkv_mlp_proj = nn.Linear(query_dim, 3 * self.inner_dim + self.mlp_hidden_dim * mlp_mult_factor, bias=bias)
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)
        self.act_fn = Flux2SwiGLU()
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias)

        self.processor = processor if processor is not None else Flux2ParallelSelfAttnProcessor()

    def set_processor(self, processor):
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            processor=Flux2ParallelSelfAttnProcessor(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        else:
            return hidden_states


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            processor=Flux2AttnProcessor(),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_params_img: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        temb_mod_params_txt: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ):
        super().__init__()
        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
        )
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
            )
        else:
            self.guidance_embedder = None

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))
        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))
            time_guidance_emb = timesteps_emb + guidance_emb
            return time_guidance_emb
        else:
            return timesteps_emb


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class Flux2Transformer2DModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
    ):
        super().__init__()
        self.config = {
            "patch_size": patch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_layers": num_layers,
            "num_single_layers": num_single_layers,
            "attention_head_dim": attention_head_dim,
            "num_attention_heads": num_attention_heads,
            "joint_attention_dim": joint_attention_dim,
            "timestep_guidance_channels": timestep_guidance_channels,
            "mlp_ratio": mlp_ratio,
            "axes_dims_rope": axes_dims_rope,
            "rope_theta": rope_theta,
            "eps": eps,
            "guidance_embeds": guidance_embeds,
        }
        
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList([
            Flux2TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
                bias=False,
            )
            for _ in range(num_layers)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            Flux2SingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                eps=eps,
                bias=False,
            )
            for _ in range(num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = 0
        self.offloader = None

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    @property
    def num_blocks(self):
        return len(self.transformer_blocks) + len(self.single_transformer_blocks)

    @property
    def num_double_blocks(self):
        return len(self.transformer_blocks)

    @property
    def num_single_blocks(self):
        return len(self.single_transformer_blocks)

    _enable_block_swap_call_count = 0
    _prepare_call_count = 0
    _forward_call_count = 0

    def enable_block_swap(
        self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False
    ):
        Flux2Transformer2DModel._enable_block_swap_call_count += 1
        call_count = Flux2Transformer2DModel._enable_block_swap_call_count
        
        import traceback
        print(f"\n[DEBUG] enable_block_swap 被调用 (第 {call_count} 次)")
        print(f"[DEBUG] 调用栈:\n{''.join(traceback.format_stack()[-5:-1])}")
        
        if hasattr(self, 'offloader') and self.offloader is not None:
            print(f"[WARNING] offloader 已存在，将被覆盖！正在清理旧 offloader...")
            self.cleanup_offloader()
        
        self.blocks_to_swap = blocks_to_swap
        num_single = self.num_single_blocks

        assert self.blocks_to_swap <= num_single - 1, (
            f"最多只能交换 {num_single - 1} 个 single blocks，请求了 {self.blocks_to_swap} 个"
        )

        self.offloader = ModelOffloader(
            "flux2-klein-single-block",
            list(self.single_transformer_blocks),
            num_single,
            self.blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        print(
            f"Flux2Model: Block swap 已启用，交换 {self.blocks_to_swap}/{num_single} 个 single blocks, "
            f"支持反向传播: {supports_backward}, 固定内存: {use_pinned_memory}"
        )
        print(f"[DEBUG] offloader id: {id(self.offloader)}")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_single_blocks = self.single_transformer_blocks
            self.single_transformer_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.single_transformer_blocks = save_single_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        Flux2Transformer2DModel._prepare_call_count += 1
        if Flux2Transformer2DModel._prepare_call_count <= 5:
            print(f"[DEBUG] prepare_block_swap_before_forward 被调用 (第 {Flux2Transformer2DModel._prepare_call_count} 次), offloader id: {id(self.offloader)}")
        self.offloader.prepare_block_devices_before_forward(list(self.single_transformer_blocks))

    def cleanup_offloader(self):
        if self.offloader is not None:
            self.offloader.shutdown()
            self.offloader = None

    def _gradient_checkpointing_func(self, block, *args):
        if self.activation_cpu_offloading:
            block = create_cpu_offloading_wrapper(block, self.x_embedder.weight.device)
        return torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        input_device = hidden_states.device

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_params_img=double_stream_mod_img,
                    temb_mod_params_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        single_blocks = list(self.single_transformer_blocks)
        for idx, block in enumerate(single_blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(idx)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod_params=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(single_blocks, idx)

        if input_device != hidden_states.device:
            hidden_states = hidden_states.to(input_device)

        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return {"sample": output}

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """保存模型到指定目录"""
        import os
        import json
        from safetensors.torch import save_file as st_save_file
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存配置
        config = {
            "_class_name": "Flux2Transformer2DModel",
            "patch_size": self.config.get("patch_size", 1),
            "in_channels": self.config.get("in_channels", 128),
            "out_channels": self.config.get("out_channels", None),
            "num_layers": self.config.get("num_layers", 8),
            "num_single_layers": self.config.get("num_single_layers", 48),
            "attention_head_dim": self.config.get("attention_head_dim", 128),
            "num_attention_heads": self.config.get("num_attention_heads", 48),
            "joint_attention_dim": self.config.get("joint_attention_dim", 15360),
            "timestep_guidance_channels": self.config.get("timestep_guidance_channels", 256),
            "mlp_ratio": self.config.get("mlp_ratio", 3.0),
            "axes_dims_rope": list(self.config.get("axes_dims_rope", [32, 32, 32, 32])),
            "rope_theta": self.config.get("rope_theta", 2000),
            "eps": self.config.get("eps", 1e-6),
            "guidance_embeds": self.config.get("guidance_embeds", True),
        }
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # 收集 state_dict
        state_dict = {}
        for name, param in self.named_parameters():
            state_dict[name] = param.cpu().clone()
        for name, buf in self.named_buffers():
            state_dict[name] = buf.cpu().clone()
        
        # 保存权重
        if safe_serialization:
            weights_path = os.path.join(save_directory, "diffusion_pytorch_model.safetensors")
            st_save_file(state_dict, weights_path)
        else:
            weights_path = os.path.join(save_directory, "diffusion_pytorch_model.bin")
            torch.save(state_dict, weights_path)
        
        print(f"Model saved to {save_directory}")


def load_flux2_transformer_from_diffusers(
    model_id: str,
    subfolder: str = "transformer",
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    from pathlib import Path
    import json
    
    model_path = Path(model_id) / subfolder
    config_path = model_path / "config.json"
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    with init_empty_weights():
        model = Flux2Transformer2DModel(
            patch_size=config.get("patch_size", 1),
            in_channels=config.get("in_channels", 128),
            out_channels=config.get("out_channels", None),
            num_layers=config.get("num_layers", 8),
            num_single_layers=config.get("num_single_layers", 48),
            attention_head_dim=config.get("attention_head_dim", 128),
            num_attention_heads=config.get("num_attention_heads", 48),
            joint_attention_dim=config.get("joint_attention_dim", 15360),
            timestep_guidance_channels=config.get("timestep_guidance_channels", 256),
            mlp_ratio=config.get("mlp_ratio", 3.0),
            axes_dims_rope=tuple(config.get("axes_dims_rope", [32, 32, 32, 32])),
            rope_theta=config.get("rope_theta", 2000),
            eps=config.get("eps", 1e-6),
            guidance_embeds=config.get("guidance_embeds", True),
        )
    
    safetensors_files = list(model_path.glob("*.safetensors"))
    state_dict = {}
    for sf_file in safetensors_files:
        partial_state_dict = load_file(sf_file)
        state_dict.update(partial_state_dict)
    
    model.load_state_dict(state_dict, strict=True, assign=True)
    model = model.to(dtype=torch_dtype, device=device)
    
    return model
