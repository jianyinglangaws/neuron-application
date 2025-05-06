import functools
import os

import einops
import torch
import torch.nn.functional as F
from torch import nn
import neuronxcc.nki.language as nl

from esm.layers.rotary import (
    RotaryEmbedding,
    TritonRotaryEmbedding,
)

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func  # type:ignore
except ImportError:
    flash_attn_varlen_func = None

from neuronxcc.nki.kernels.attention import flash_fwd
from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_nki_call = nki_jit()(flash_fwd)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )

        if seq_id is not None:
            # Where True, enable participation in attention.
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, mask_BHLL
            )
        else:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD
            )

        context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")

        return self.out_proj(context_BLD)

def _flash_attn_forward(q, k, v, causal, mixed_precision, seed, dropout_p, softmax_scale):
    bs, num_heads, head_dim, seq = q.shape
    attn_output = torch.zeros(size=(bs, num_heads, seq, head_dim), dtype=q.dtype, device=q.device)
    if mixed_precision:
        if os.environ.get("XLA_DOWNCAST_BF16"):
            lse_dtype = torch.float64
        else:
            lse_dtype = torch.float32
    else:
        lse_dtype = q.dtype
    lse = torch.empty(
        size=(bs, num_heads, nl.tile_size.pmax, seq // nl.tile_size.pmax),
        dtype=lse_dtype, device=q.device,
    )
    _flash_fwd_nki_call[bs, num_heads](
        q,
        k,
        v,
        seed,
        attn_output,
        lse,
        use_causal_mask=causal,
        mixed_precision=mixed_precision,
        dropout_p=dropout_p,
    )
    return attn_output, lse

class NeuronMultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        v = v.unflatten(-1, (self.n_heads, self.d_head))
        q, k, v = self.rotary(q, k, v)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        v = v.flatten(-2, -1)
        return q, k, v

    def forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        print("query_BHLD shape after chunk: ", query_BLD.shape)
        print("value_BHLD shape after chunk: ", value_BLD.shape)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        print("query_BHLD shape after q_ln: ", query_BLD.shape)
        print("value_BHLD shape after q_ln: ", value_BLD.shape)
        query_BLD, key_BLD, value_BLD = self._apply_rotary(query_BLD, key_BLD, value_BLD)
        print("query_BHLD shape after rotary: ", query_BLD.shape)
        print("value_BHLD shape after rotary: ", value_BLD.shape)
        reshaper = functools.partial(
            einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads
        )
 
        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper, (query_BLD, key_BLD, value_BLD)
        )
   
        print("query_BHLD shape: ", query_BHLD.shape)
        print("value_BHLD shape: ", value_BHLD.shape)
        query_BHLD = query_BHLD.permute(0, 1, 3, 2)
        key_BHLD = key_BHLD.permute(0, 1, 3, 2)
        #value_BHLD = value_BHLD.permute(0, 1, 3, 2)

        attn_output, lse = _flash_attn_forward(
            query_BHLD,
            key_BHLD,
            value_BHLD,
            causal=True,
            mixed_precision=True,
            seed=None,
            dropout_p=0.0,
            softmax_scale=self.d_head**-0.5,
        )

  #      if seq_id is not None:
  #          # Where True, enable participation in attention.
  #          mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
  #          mask_BHLL = mask_BLL.unsqueeze(1)

  #          context_BHLD = F.scaled_dot_product_attention(
  #              query_BHLD, key_BHLD, value_BHLD, mask_BHLL
  #          )
  #      else:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
  #          context_BHLD = F.scaled_dot_product_attention(
  #              query_BHLD, key_BHLD, value_BHLD
  #          )

        context_BLD = einops.rearrange(attn_output, "b h s d -> b s (h d)")

        return self.out_proj(context_BLD)


class FlashMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__(
            d_model=d_model, n_heads=n_heads, bias=bias, qk_layernorm=qk_layernorm
        )

        # Flash attention rotary.
        self.rotary = TritonRotaryEmbedding(d_model // n_heads)

    def forward(self, x, seq_id):
        assert seq_id.dtype == torch.bool

        seqlens = seq_id.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = seqlens.max().item()

        qkv_ND3 = self.layernorm_qkv(x)

        query_ND, key_ND, value_ND = torch.chunk(qkv_ND3, 3, dim=-1)
        query_ND, key_ND = (
            self.q_ln(query_ND).to(query_ND.dtype),
            self.k_ln(key_ND).to(query_ND.dtype),
        )

        qkv_N3D = torch.stack([query_ND, key_ND, value_ND], dim=1)
        qkv_N3HD = einops.rearrange(
            qkv_N3D, pattern="n a (h d) -> n a h d", h=self.n_heads
        )
        qkv_N3HD = self.rotary(qkv_N3HD, cu_seqlens, max_seqlen)

        context_NHD = flash_attn_varlen_qkvpacked_func(
            qkv_N3HD, cu_seqlens, max_seqlen, softmax_scale=self.d_head**-0.5
        )
        context_ND = einops.rearrange(context_NHD, "n h d -> n (h d)")

        return self.out_proj(context_ND)
