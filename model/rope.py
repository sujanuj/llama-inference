"""Rotary Position Embeddings (RoPE) — how Llama encodes token position.

Unlike GPT-2's learned absolute position embeddings (a lookup table
added to the input), RoPE encodes position by ROTATING each query/key
vector by an angle proportional to its position in the sequence, before
the attention dot product. The key property that makes this work: the
dot product of two rotated vectors depends only on their RELATIVE
rotation (i.e. their relative position), not their absolute positions —
so attention scores naturally become a function of relative distance
between tokens, which is exactly what positional encoding is supposed to
provide.

This file is the single highest-bug-risk part of the whole model. RoPE
has at least two materially different but superficially similar
implementations in circulation (the original RoPE paper's interleaved
pairing vs. the "rotate_half" convention HuggingFace's Llama code
actually uses), and getting the wrong one produces a model that runs
without any error, produces plausible-looking-but-wrong attention
patterns, and degrades output quality in a way that's easy to miss
without an explicit numerical check against a reference. This
implementation follows HuggingFace's rotate_half convention specifically
because that's what real Llama checkpoints are trained against — using
the "more elegant" interleaved version from the original paper would be
architecturally well-formed but numerically WRONG for this model.
"""

import torch


def compute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin rotation tables for every position up to
    max_seq_len.

    Returns:
        cos, sin: each (max_seq_len, head_dim), ready to be sliced down
        to whatever sequence length is actually needed and broadcast
        against query/key tensors.
    """
    # One frequency per PAIR of dimensions, not per dimension — head_dim
    # is split into head_dim/2 rotation planes, each rotating at a
    # different frequency. Lower dimension-pairs rotate fast (capture
    # fine-grained, local position differences); higher pairs rotate
    # slow (capture coarse, long-range position differences) — the same
    # multi-frequency idea as sinusoidal position embeddings, just
    # applied as a rotation instead of an additive signal.
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )  # shape: (head_dim/2,)

    positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim/2)

    # Each frequency is used for BOTH halves of its rotation pair (see
    # rotate_half below), so the table is duplicated to width head_dim
    # rather than left at head_dim/2 — this matches what apply_rope
    # expects to multiply against directly, with no further reshaping.
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dimension in half and swap-and-negate the halves:
    [a, b] -> [-b, a]. This, combined with the duplicated cos/sin table
    above, is what actually performs the 2D rotation in each frequency
    pair — NOT by interleaving adjacent dimensions (the original RoPE
    paper's presentation), but by treating the first and second HALVES
    of head_dim as the two members of each rotation pair. This is the
    specific convention real Llama checkpoints are trained with; this
    function existing separately and being unit-tested on its own is
    deliberate, since a sign or ordering mistake here is exactly the
    silent-wrongness failure mode described in the module docstring.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to a query or key tensor.

    Args:
        x: (batch, num_heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim) — already sliced to match x's
            actual sequence length and broadcastable against it.

    Returns:
        Same shape AND dtype as x.
    """
    # compute_rope_frequencies always returns cos/sin in float32 (see
    # that function's docstring) for the same numerical-stability reason
    # RMSNorm and the attention softmax upcast internally. But x itself
    # may be bf16 (real Llama-3.2-1B checkpoints are stored in bf16 —
    # see model/load_weights.py). Without casting cos/sin to x's dtype
    # FIRST, `x * cos` would silently upcast to float32 via torch's type
    # promotion rules, and the rotated result would end up float32 even
    # though x came in as bf16 — not an error, just a quiet dtype change
    # that could cause a mismatch further down the pipeline (exactly the
    # class of bug that surfaced as a real RuntimeError in
    # scaled_dot_product_attention when this was first run against the
    # actual bf16 checkpoint). Casting explicitly here keeps the
    # function's contract ("same dtype as x") true regardless of x's
    # actual dtype.
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    # Standard 2D rotation formula applied per frequency pair:
    #   rotated = x * cos + rotate_half(x) * sin
    # This is algebraically the rotation matrix
    #   [cos  -sin] [x1]
    #   [sin   cos] [x2]
    # applied independently to each of the head_dim/2 pairs, with
    # rotate_half providing the "-x2, x1" term and the duplicated cos/sin
    # table making the elementwise multiply line up correctly across
    # both halves without an explicit reshape into pairs.
    return x * cos + _rotate_half(x) * sin
