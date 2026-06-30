"""Grouped-query attention (GQA) — Llama-3.2-1B uses 32 query heads but
only 8 key/value heads, with every 4 query heads sharing one KV head.

This is purely an INFERENCE-memory optimization: the KV-cache (built in
the next phase) stores one K/V vector per cached token per KV head per
layer, so fewer KV heads means a proportionally smaller cache — 4x
smaller here, for identical sequence length, compared to what full
multi-head attention (32 KV heads) would need.

To actually compute attention scores, each KV head's K/V vectors must be
REPEATED so every query head has a same-shaped K/V partner to dot
against. Getting the repeat axis and block-vs-interleave ordering wrong
is the main correctness risk in this file — it would silently pair some
query heads with the wrong KV head's cache entries, producing a model
that runs, produces plausible-looking numbers, and is simply wrong. This
is exactly the kind of bug the explicit indexing test below
(test_gqa.py) is built to catch.
"""

import math

import torch

from model.rope import apply_rope


def repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Expand (batch, num_kv_heads, seq_len, head_dim) to
    (batch, num_kv_heads * num_groups, seq_len, head_dim) by repeating
    each KV head num_groups times, BLOCK-style (not interleaved):

        kv heads: [A, B]  with num_groups=4  ->  [A, A, A, A, B, B, B, B]

    not interleaved ([A, B, A, B, A, B, A, B]). Block-style is what
    actually matches how query heads are grouped in the real model —
    query heads [0,1,2,3] attend via KV head 0, query heads [4,5,6,7]
    via KV head 1, and so on, which only lines up correctly with a
    block repeat. This function is split out and unit-tested on its own
    specifically because getting the repeat pattern backwards would be
    silently wrong rather than loudly broken.
    """
    batch, num_kv_heads, seq_len, head_dim = x.shape
    # repeat_interleave with dim=1 produces exactly the block pattern
    # described above: each of the num_kv_heads slices along dim 1 gets
    # duplicated num_groups times IN PLACE before moving to the next
    # slice — this is the block pattern, not the interleaved one,
    # despite the "interleave" in the function's name (torch's naming
    # refers to interleaving repeats of the SAME element, not
    # interleaving DIFFERENT elements with each other).
    return x.repeat_interleave(num_groups, dim=1)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Standard attention: softmax(QK^T / sqrt(d)) V, with an optional
    causal mask so position i can only attend to positions <= i.

    Args:
        query: (batch, num_heads, q_len, head_dim)
        key, value: (batch, num_heads, kv_len, head_dim) — already
            repeat_kv'd to match query's head count.
        causal: if True, mask out attention to future positions.

    Returns:
        (batch, num_heads, q_len, head_dim) — same dtype as the inputs.
    """
    input_dtype = query.dtype
    head_dim = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    # scores: (batch, num_heads, q_len, kv_len)

    if causal:
        q_len, kv_len = scores.shape[-2], scores.shape[-1]
        # kv_len can exceed q_len when there's cached context: the
        # query only covers the NEWEST tokens, but keys/values cover
        # cached history too. Position i in the query block corresponds
        # to absolute position (kv_len - q_len + i) — it may attend to
        # everything up to and including itself, nothing after.
        offset = kv_len - q_len
        row_positions = torch.arange(q_len).unsqueeze(1) + offset  # (q_len, 1)
        col_positions = torch.arange(kv_len).unsqueeze(0)  # (1, kv_len)
        mask = col_positions > row_positions  # True where attention should be blocked
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax computed in float32 regardless of the input's dtype, then
    # cast back before the final matmul with value. This matters for two
    # real reasons, not just style:
    #   1. Numerical stability — softmax involves exponentials, and bf16
    #      has very few mantissa bits, so accumulating a sum of
    #      exponentials in bf16 loses real precision. Real Llama
    #      checkpoints (loaded as bf16, see model/load_weights.py) are
    #      trained with reference implementations that do this same
    #      upcast for exactly this reason.
    #   2. A real, observed bug: without an explicit, consistent dtype
    #      discipline here, torch's type-promotion rules for masked_fill
    #      + softmax on a bf16 tensor produced a torch.matmul dtype
    #      mismatch (float32 weights vs. bf16 value) when this was first
    #      run against the real Llama-3.2-1B bf16 checkpoint — caught
    #      immediately as a loud RuntimeError, not a silent wrong
    #      number, which is the failure mode this project's testing
    #      philosophy is built around catching either way.
    weights = torch.softmax(scores.to(torch.float32), dim=-1).to(input_dtype)
    return torch.matmul(weights, value)


def attention(
    hidden_states: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool = True,
) -> torch.Tensor:
    """Full GQA attention block: project to Q/K/V, apply RoPE, repeat
    KV heads, compute attention, project back out.

    Args:
        hidden_states: (batch, seq_len, hidden_size)
        q_proj: (hidden_size, num_heads * head_dim)
        k_proj, v_proj: (hidden_size, num_kv_heads * head_dim)
        o_proj: (num_heads * head_dim, hidden_size)
        cos, sin: (seq_len, head_dim) — RoPE tables sliced to this
            sequence length, at the correct ABSOLUTE positions (the
            caller is responsible for slicing these correctly when
            there's cached context — see kvcache phase).

    Returns:
        (batch, seq_len, hidden_size)
    """
    batch, seq_len, hidden_size = hidden_states.shape

    q = hidden_states @ q_proj  # (batch, seq_len, num_heads * head_dim)
    k = hidden_states @ k_proj  # (batch, seq_len, num_kv_heads * head_dim)
    v = hidden_states @ v_proj

    # Reshape to (batch, heads, seq_len, head_dim) — heads moved before
    # seq_len so RoPE and attention can broadcast/matmul correctly
    # along the last two dims without per-head loops.
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    num_groups = num_heads // num_kv_heads
    k = repeat_kv(k, num_groups)
    v = repeat_kv(v, num_groups)

    out = scaled_dot_product_attention(q, k, v, causal=causal)
    # (batch, num_heads, seq_len, head_dim)

    out = out.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
    return out @ o_proj


def attention_with_cache(
    hidden_states: torch.Tensor,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cache,
    layer_idx: int,
) -> torch.Tensor:
    """Cache-aware GQA attention for incremental generation: computes
    Q/K/V for only the NEW tokens in hidden_states, appends the new K/V
    into the cache (at the KV-head dimension -- see kvcache/naive_cache.py
    for why that's where the real GQA memory savings live), then attends
    the new queries against the FULL cached history (old + new).

    This deliberately reuses scaled_dot_product_attention and repeat_kv
    unchanged from the no-cache path above -- the only new logic here is
    the cache read/append, not a second implementation of attention
    math that could drift out of sync with the original.

    Args:
        hidden_states: (batch, new_seq_len, hidden_size) -- ONLY the new
            tokens for this generation step, not the full sequence so far.
        cos, sin: RoPE tables already sliced to the ABSOLUTE positions of
            the new tokens specifically (e.g. if 10 tokens are already
            cached and 1 new token is being generated, this should be
            cos[10:11], sin[10:11] -- not cos[0:1]). Getting this offset
            wrong would rotate the new token as if it were always at
            position 0, which is exactly the kind of silent, no-crash
            bug this project's testing philosophy is built to catch.
        cache: a kvcache.naive_cache.KVCache instance.
        layer_idx: which layer's slot in the cache to read/append.

    Returns:
        (batch, new_seq_len, hidden_size)
    """
    batch, new_seq_len, hidden_size = hidden_states.shape

    q = hidden_states @ q_proj
    k_new = hidden_states @ k_proj
    v_new = hidden_states @ v_proj

    q = q.view(batch, new_seq_len, num_heads, head_dim).transpose(1, 2)
    k_new = k_new.view(batch, new_seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v_new = v_new.view(batch, new_seq_len, num_kv_heads, head_dim).transpose(1, 2)

    q = apply_rope(q, cos, sin)
    k_new = apply_rope(k_new, cos, sin)

    # Append into the cache and get back the FULL key/value (cached
    # history concatenated with the new tokens) -- this, not k_new/v_new
    # alone, is what the new queries need to attend against.
    k_full, v_full = cache.append(layer_idx, k_new, v_new)

    num_groups = num_heads // num_kv_heads
    k_full = repeat_kv(k_full, num_groups)
    v_full = repeat_kv(v_full, num_groups)

    # causal=True still applies correctly here even though q's length
    # (new_seq_len) and k/v's length (full cached length) differ --
    # this is exactly the kv_len > q_len case scaled_dot_product_attention
    # was already built and tested for back in Phase 2
    # (test_attention_with_kv_offset_for_cached_context), specifically
    # anticipating this use.
    out = scaled_dot_product_attention(q, k_full, v_full, causal=True)

    out = out.transpose(1, 2).contiguous().view(batch, new_seq_len, num_heads * head_dim)
    return out @ o_proj
