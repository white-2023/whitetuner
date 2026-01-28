from typing import Optional
import torch

__all__ = ["flash_attention"]


def flash_attention(
    qkv,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    attn_mode: Optional[str] = "torch",
    split_attn: bool = False,
):
    q, k, v = qkv
    qkv.clear()

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if attn_mode == "torch" or attn_mode == "sdpa":
        if q_scale is not None:
            q = q * q_scale
        q = half(q.transpose(1, 2))
        k = half(k.transpose(1, 2))
        v = half(v.transpose(1, 2))

        if not split_attn:
            q = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale
            )
            x = q
        else:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = torch.nn.functional.scaled_dot_product_attention(
                    q[i : i + 1], k[i : i + 1], v[i : i + 1], is_causal=causal, dropout_p=dropout_p, scale=softmax_scale
                )

        del q, k, v
        x = x.transpose(1, 2).contiguous()
        return x.type(out_dtype)

    raise ValueError(f"Unknown attention mode: {attn_mode}, only 'torch' is supported in whitetuner")

