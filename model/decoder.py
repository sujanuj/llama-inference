"""A single Llama decoder layer: the pre-norm residual block that gets
stacked 16 times to build the full Llama-3.2-1B model.

    residual = x
    x = rms_norm(x, input_layernorm)
    x = attention(x, ...)
    x = residual + x                    <- residual connection 1

    residual = x
    x = rms_norm(x, post_attention_layernorm)
    x = swiglu_mlp(x, ...)
    x = residual + x                    <- residual connection 2

This is "pre-norm": each sub-block (attention, MLP) normalizes its INPUT
before doing any work, rather than normalizing its output. The reason
this matters beyond style: the residual connections let the identity
function flow through unchanged at every layer (x = residual + 0 is
possible if a sub-block contributes nothing), which is what makes very
deep transformers (16 layers here, 100+ in larger models) trainable at
all — gradients have a path that doesn't have to flow through every
single nonlinearity in the network. Post-norm architectures (normalize
the OUTPUT instead) don't have this property as cleanly, which is part
of why every modern large transformer, Llama included, uses pre-norm.
"""

import torch

from model.attention import attention, attention_with_cache
from model.mlp import swiglu_mlp
from model.rmsnorm import rms_norm
from model.weights import DecoderLayerWeights


def decoder_layer(
    hidden_states: torch.Tensor,
    weights: DecoderLayerWeights,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    causal: bool = True,
) -> torch.Tensor:
    """Args:
        hidden_states: (batch, seq_len, hidden_size)
        weights: this layer's RMSNorm + attention + MLP weights.
        cos, sin: RoPE tables, already sliced to the correct absolute
            positions for this forward pass (see attention.py).

    Returns:
        (batch, seq_len, hidden_size) — same shape as input.
    """
    residual = hidden_states
    normed = rms_norm(hidden_states, weights.input_layernorm, rms_norm_eps)
    attn_out = attention(
        normed,
        weights.attention.q_proj,
        weights.attention.k_proj,
        weights.attention.v_proj,
        weights.attention.o_proj,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
    )
    hidden_states = residual + attn_out

    residual = hidden_states
    normed = rms_norm(hidden_states, weights.post_attention_layernorm, rms_norm_eps)
    mlp_out = swiglu_mlp(
        normed, weights.mlp.gate_proj, weights.mlp.up_proj, weights.mlp.down_proj
    )
    hidden_states = residual + mlp_out

    return hidden_states


def decoder_layer_with_cache(
    hidden_states: torch.Tensor,
    weights: DecoderLayerWeights,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    cache,
    layer_idx: int,
) -> torch.Tensor:
    """Cache-aware version of decoder_layer, for incremental generation.
    Identical pre-norm residual structure (see module docstring) --
    the only difference from decoder_layer is calling
    attention_with_cache instead of attention. The MLP sub-block needs
    no cache awareness at all: it operates purely on the current
    hidden_states with no notion of sequence history, so it's identical
    in both the cached and uncached paths.

    Args:
        hidden_states: (batch, new_seq_len, hidden_size) -- only the
            NEW tokens for this generation step.
        cos, sin: RoPE tables sliced to the new tokens' absolute
            positions (see attention_with_cache for why this matters).
        cache, layer_idx: see attention_with_cache.

    Returns:
        (batch, new_seq_len, hidden_size)
    """
    residual = hidden_states
    normed = rms_norm(hidden_states, weights.input_layernorm, rms_norm_eps)
    attn_out = attention_with_cache(
        normed,
        weights.attention.q_proj,
        weights.attention.k_proj,
        weights.attention.v_proj,
        weights.attention.o_proj,
        cos,
        sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache=cache,
        layer_idx=layer_idx,
    )
    hidden_states = residual + attn_out

    residual = hidden_states
    normed = rms_norm(hidden_states, weights.post_attention_layernorm, rms_norm_eps)
    mlp_out = swiglu_mlp(
        normed, weights.mlp.gate_proj, weights.mlp.up_proj, weights.mlp.down_proj
    )
    hidden_states = residual + mlp_out

    return hidden_states
