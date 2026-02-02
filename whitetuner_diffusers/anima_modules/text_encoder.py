import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from safetensors.torch import load_file


def rms_norm(x, weight, eps=1e-6):
    x_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * weight.float()
    return x.to(x_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add=False, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.add = add

    def forward(self, x: torch.Tensor):
        w = self.weight
        if self.add:
            w = w + 1.0
        return rms_norm(x, w, self.eps)


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
    rms_norm_add: bool = False
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    rope_dims = None
    q_norm: str = "gemma3"
    k_norm: str = "gemma3"
    rope_scale = None
    final_norm: bool = True


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_freqs_cis(head_dim, position_ids, theta, rope_scale=None, rope_dims=None, device=None):
    if not isinstance(theta, list):
        theta = [theta]

    out = []
    for index, t in enumerate(theta):
        theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
        inv_freq = 1.0 / (t ** (theta_numerator / head_dim))

        if rope_scale is not None:
            if isinstance(rope_scale, list):
                inv_freq /= rope_scale[index]
            else:
                inv_freq /= rope_scale

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        if rope_dims is not None and position_ids.shape[0] > 1:
            mrope_section = rope_dims * 2
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        out.append((cos, sin))

    if len(out) == 1:
        return out[0]

    return out


def apply_rope(xq, xk, freqs_cis):
    org_dtype = xq.dtype
    cos = freqs_cis[0]
    sin = freqs_cis[1]
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed.to(org_dtype), k_embed.to(org_dtype)


class Attention(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = None
        self.k_norm = None

        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        mask = attention_mask
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        attn_output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=0.0, is_causal=False)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.inner_size)
        
        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        if config.mlp_activation == "silu":
            self.activation = F.silu
        elif config.mlp_activation == "gelu_pytorch_tanh":
            self.activation = lambda a: F.gelu(a, approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype)
        self.mlp = MLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Qwen3TextEncoder(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(config, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ])

        if config.final_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        else:
            self.norm = None

    _debug_call = 0
    
    def forward(self, input_ids, attention_mask=None):
        Qwen3TextEncoder._debug_call += 1
        call_num = Qwen3TextEncoder._debug_call
        
        x = self.embed_tokens(input_ids)
        if call_num <= 2:
            print(f"[DEBUG TE] call={call_num} after embed_tokens: mean={x.float().mean().item():.6f}, std={x.float().std().item():.6f}")

        seq_len = x.shape[1]
        position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        freqs_cis = precompute_freqs_cis(
            self.config.head_dim,
            position_ids,
            self.config.rope_theta,
            self.config.rope_scale,
            self.config.rope_dims,
            device=x.device
        )
        if call_num <= 2:
            print(f"[DEBUG TE] call={call_num} freqs_cis[0]: mean={freqs_cis[0].float().mean().item():.6f}, std={freqs_cis[0].float().std().item():.6f}")

        mask = None
        if seq_len > 1:
            causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
            mask = causal_mask

        for i, layer in enumerate(self.layers):
            x = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis)
            if call_num <= 2 and i in [0, 5, 10, 15, 20, 25, 27]:
                print(f"[DEBUG TE] call={call_num} after layer {i}: mean={x.float().mean().item():.6f}, std={x.float().std().item():.6f}")

        if self.norm is not None:
            x = self.norm(x)

        if call_num <= 2:
            print(f"[DEBUG TE] call={call_num} final output: mean={x.float().mean().item():.6f}, std={x.float().std().item():.6f}")

        return x


def load_qwen3_text_encoder(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    print(f"Loading Qwen3 0.6B from {model_path}")
    
    config = Qwen3_06BConfig()
    
    model = Qwen3TextEncoder(config, device="cpu", dtype=dtype)
    
    state_dict = load_file(model_path)
    
    new_state_dict = {}
    prefix_map = {
        "model.embed_tokens": "embed_tokens",
        "model.layers": "layers",
        "model.norm": "norm",
    }
    
    for k, v in state_dict.items():
        new_key = k
        for old_prefix, new_prefix in prefix_map.items():
            if k.startswith(old_prefix):
                new_key = new_prefix + k[len(old_prefix):]
                break
        
        new_key = new_key.replace(".self_attn.q_norm.", ".self_attn.q_norm.")
        new_key = new_key.replace(".self_attn.k_norm.", ".self_attn.k_norm.")
        
        new_state_dict[new_key] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  - Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  - Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")
    
    embed_weight = model.embed_tokens.weight
    print(f"  - embed_tokens: mean={embed_weight.float().mean().item():.6f}, std={embed_weight.float().std().item():.6f}")
    layer0_ln = model.layers[0].input_layernorm.weight
    print(f"  - layers.0.input_layernorm: mean={layer0_ln.float().mean().item():.6f}, std={layer0_ln.float().std().item():.6f}")
    
    q_norm_weight = model.layers[0].self_attn.q_norm.weight if model.layers[0].self_attn.q_norm else None
    k_norm_weight = model.layers[0].self_attn.k_norm.weight if model.layers[0].self_attn.k_norm else None
    if q_norm_weight is not None:
        print(f"  - layers.0.q_norm: mean={q_norm_weight.float().mean().item():.6f}, std={q_norm_weight.float().std().item():.6f}")
    else:
        print(f"  - layers.0.q_norm: None")
    if k_norm_weight is not None:
        print(f"  - layers.0.k_norm: mean={k_norm_weight.float().mean().item():.6f}, std={k_norm_weight.float().std().item():.6f}")
    else:
        print(f"  - layers.0.k_norm: None")
    
    print(f"  - Keys in state_dict containing 'norm' (first 10):")
    norm_keys = [k for k in state_dict.keys() if 'norm' in k.lower()]
    for k in norm_keys[:10]:
        print(f"    - {k}")
    
    model = model.to(device=device)
    model.eval()
    print(f"Qwen3 0.6B loaded successfully")
    
    return model


class AnimaTokenizer:
    QWEN_HF_MODEL_ID = "Qwen/Qwen2.5-0.5B"
    T5_HF_MODEL_ID = "google-t5/t5-base"
    
    def __init__(self, qwen_tokenizer_path: str = None, t5_tokenizer_path: str = None):
        from transformers import Qwen2Tokenizer, T5TokenizerFast
        
        if qwen_tokenizer_path is None:
            qwen_tokenizer_path = self.QWEN_HF_MODEL_ID
        
        if t5_tokenizer_path is None:
            t5_tokenizer_path = self.T5_HF_MODEL_ID
        
        print(f"Loading Qwen2Tokenizer from: {qwen_tokenizer_path}")
        self.qwen_tokenizer = Qwen2Tokenizer.from_pretrained(qwen_tokenizer_path)
        
        print(f"Loading T5TokenizerFast from: {t5_tokenizer_path}")
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_tokenizer_path)
        
        self.pad_token_id = 151643

    def encode_qwen(self, text: str, device: str = "cuda"):
        if not text or text.strip() == "":
            text = " "
        
        inputs = self.qwen_tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
    
    def encode_t5(self, text: str, device: str = "cuda"):
        if not text or text.strip() == "":
            text = " "
        
        inputs = self.t5_tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return inputs["input_ids"].to(device).squeeze(0)

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt", device="cuda"):
        if isinstance(texts, str):
            texts = [texts]
        
        qwen_ids_list = []
        t5_ids_list = []
        max_qwen_len = 0
        max_t5_len = 0
        
        for text in texts:
            if not text or text.strip() == "":
                text = " "
            
            qwen_inputs = self.qwen_tokenizer(
                text,
                padding=False,
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            qwen_ids = qwen_inputs["input_ids"].squeeze(0)
            qwen_ids_list.append(qwen_ids)
            max_qwen_len = max(max_qwen_len, qwen_ids.shape[0])
            
            t5_inputs = self.t5_tokenizer(
                text,
                padding=False,
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt",
            )
            t5_ids = t5_inputs["input_ids"].squeeze(0)
            t5_ids_list.append(t5_ids)
            max_t5_len = max(max_t5_len, t5_ids.shape[0])
        
        if padding:
            padded_qwen = []
            padded_t5 = []
            attention_masks = []
            
            for qwen_ids, t5_ids in zip(qwen_ids_list, t5_ids_list):
                qwen_pad_len = max_qwen_len - qwen_ids.shape[0]
                t5_pad_len = max_t5_len - t5_ids.shape[0]
                
                attention_mask = torch.ones(max_qwen_len, dtype=torch.long)
                if qwen_pad_len > 0:
                    qwen_ids = torch.cat([qwen_ids, torch.full((qwen_pad_len,), self.pad_token_id, dtype=torch.long)])
                    attention_mask[max_qwen_len - qwen_pad_len:] = 0
                
                if t5_pad_len > 0:
                    t5_ids = torch.cat([t5_ids, torch.zeros(t5_pad_len, dtype=torch.long)])
                
                padded_qwen.append(qwen_ids)
                padded_t5.append(t5_ids)
                attention_masks.append(attention_mask)
            
            qwen_ids = torch.stack(padded_qwen)
            t5_ids = torch.stack(padded_t5)
            attention_mask = torch.stack(attention_masks)
        else:
            qwen_ids = qwen_ids_list[0].unsqueeze(0)
            t5_ids = t5_ids_list[0].unsqueeze(0)
            attention_mask = torch.ones_like(qwen_ids)
        
        class TokenizerOutput:
            def __init__(self, input_ids, attention_mask, t5_ids):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.t5_ids = t5_ids
            
            def __getitem__(self, key):
                return getattr(self, key)
        
        return TokenizerOutput(
            input_ids=qwen_ids.to(device) if device else qwen_ids,
            attention_mask=attention_mask.to(device) if device else attention_mask,
            t5_ids=t5_ids.to(device) if device else t5_ids,
        )
