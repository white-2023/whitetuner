"""
Anima Text Encoder - Qwen3 0.6B

从 ComfyUI 的实现中提取的独立版本
"""

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


@dataclass
class Qwen3_06BConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    head_dim: int = 128


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(xq, xk, cos, sin):
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3_06BConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.inner_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, cos, sin)

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask)
        output = output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        return self.o_proj(output)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3_06BConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3_06BConfig):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask=attention_mask, cos=cos, sin=sin)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3_06BConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._init_rope()

    def _init_rope(self):
        head_dim = self.config.head_dim
        theta = self.config.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device)
        freqs = torch.outer(position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids)
        
        cos, sin = self._compute_rope(seq_length, device, hidden_states.dtype)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, cos=cos, sin=sin)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3TextEncoder(nn.Module):
    def __init__(self, config: Qwen3_06BConfig = None):
        super().__init__()
        if config is None:
            config = Qwen3_06BConfig()
        self.config = config
        self.model = Qwen3Model(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        return self.model(input_ids, attention_mask)


def load_qwen3_text_encoder(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3TextEncoder:
    print(f"Loading Qwen3 0.6B from {model_path}")
    
    state_dict = load_file(model_path)
    
    config = Qwen3_06BConfig()
    model = Qwen3TextEncoder(config)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("model.layers."):
            parts = k.split(".")
            layer_idx = int(parts[2])
            rest = ".".join(parts[3:])
            
            if "self_attn" in rest:
                rest = rest.replace("self_attn.", "self_attn.")
            
            new_key = f"model.layers.{layer_idx}.{rest}"
        
        new_state_dict[new_key] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  - Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  - Unexpected keys: {len(unexpected)}")
    
    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"Qwen3 0.6B loaded successfully")
    
    return model


class AnimaTokenizer:
    def __init__(self, qwen_tokenizer_path: str = None):
        from transformers import AutoTokenizer
        
        if qwen_tokenizer_path is None:
            qwen_tokenizer_path = "Qwen/Qwen2.5-0.5B"
        
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_tokenizer_path,
            trust_remote_code=True,
        )
        
        self.pad_token_id = 151643

    def __call__(
        self,
        text,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ):
        if isinstance(text, str):
            text = [text]
        
        qwen_encoding = self.qwen_tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        
        return {
            "input_ids": qwen_encoding["input_ids"],
            "attention_mask": qwen_encoding["attention_mask"],
        }
