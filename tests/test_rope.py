"""Tests for Rotary Position Embeddings.

This is the highest-bug-risk file in the whole model (see the module
docstring in model/rope.py for why), so the testing strategy here is
deliberately layered:

1. A hand-computed single-pair rotation, checked by hand against the
   2D rotation matrix formula directly — no torch involved in computing
   the expected values.
2. The actual mathematical property RoPE exists to provide: the
   attention dot product between a rotated query and a rotated key
   depends ONLY on their relative position, not their absolute
   positions. This is the property that would catch a subtly-wrong
   implementation that still "looks reasonable" (right shapes, no
   NaNs, plausible-looking numbers) but doesn't actually encode
   position correctly.
3. A no-op check: rotating by position 0 should change nothing, since
   the rotation angle at position 0 is 0 for every frequency.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.rope import apply_rope, compute_rope_frequencies


def test_rope_position_zero_is_identity():
    # cos(0) = 1, sin(0) = 0 for every frequency, so position 0 must be
    # a complete no-op — this is the simplest possible sanity check and
    # would catch a table-indexing-off-by-one bug immediately.
    head_dim = 8
    cos, sin = compute_rope_frequencies(head_dim, max_seq_len=4, theta=10000.0)

    x = torch.randn(1, 1, 1, head_dim)
    rotated = apply_rope(x, cos[0:1], sin[0:1])

    assert torch.allclose(rotated, x, atol=1e-6)


def test_rope_single_pair_matches_hand_computed_2d_rotation():
    # head_dim=2 means there's exactly ONE rotation pair, no duplication
    # confusion from rotate_half — the simplest possible non-trivial
    # case, checked against the textbook 2D rotation matrix computed by
    # hand in plain Python.
    head_dim = 2
    theta = 10000.0
    max_seq_len = 3
    cos, sin = compute_rope_frequencies(head_dim, max_seq_len, theta)

    x = torch.tensor([[[[1.0, 0.0]]]])  # batch=1, heads=1, seq=1, head_dim=2

    for position in range(max_seq_len):
        rotated = apply_rope(x, cos[position : position + 1], sin[position : position + 1])

        # inv_freq for the only pair (dim index 0) is theta^0 = 1, so
        # the rotation angle at this position is simply `position`
        # radians. Hand-computing the rotation of (1, 0) by that angle:
        # (cos(angle), sin(angle)).
        angle = position * (1.0 / (theta ** (0 / head_dim)))
        expected_x1 = math.cos(angle)
        expected_x2 = math.sin(angle)

        got = rotated[0, 0, 0].tolist()
        assert abs(got[0] - expected_x1) < 1e-4, f"position {position}: x1 got {got[0]}, want {expected_x1}"
        assert abs(got[1] - expected_x2) < 1e-4, f"position {position}: x2 got {got[1]}, want {expected_x2}"


def test_rope_preserves_vector_norm():
    # Rotation is norm-preserving by definition — if apply_rope changes
    # the magnitude of a vector, something is wrong with the rotation
    # math (e.g. a missing or extra factor, a sign error that turns the
    # operation into a shear rather than a rotation).
    torch.manual_seed(0)
    head_dim = 64
    cos, sin = compute_rope_frequencies(head_dim, max_seq_len=16, theta=500000.0)

    x = torch.randn(2, 4, 16, head_dim)
    rotated = apply_rope(x, cos, sin)

    norm_before = x.norm(dim=-1)
    norm_after = rotated.norm(dim=-1)

    assert torch.allclose(norm_before, norm_after, atol=1e-4), (
        f"max diff: {(norm_before - norm_after).abs().max()}"
    )


def test_rope_attention_score_depends_only_on_relative_position():
    # This is the defining property RoPE is built to provide, and the
    # single most important test in this file: for a fixed relative
    # offset between a query position and a key position, the resulting
    # dot product should be the SAME regardless of where that pair sits
    # in absolute terms. A subtly wrong rotation convention (e.g. an
    # interleaved-pairing bug, or a sign error in rotate_half) can pass
    # the simpler tests above while still failing this one, because this
    # is the actual mathematical guarantee the simpler checks don't
    # fully exercise.
    torch.manual_seed(0)
    head_dim = 64
    theta = 500000.0
    max_seq_len = 32
    cos, sin = compute_rope_frequencies(head_dim, max_seq_len, theta)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    relative_offset = 5
    dot_products = []
    for query_pos in [0, 3, 10, 20]:
        key_pos = query_pos + relative_offset
        if key_pos >= max_seq_len:
            continue

        q_rot = apply_rope(q, cos[query_pos : query_pos + 1], sin[query_pos : query_pos + 1])
        k_rot = apply_rope(k, cos[key_pos : key_pos + 1], sin[key_pos : key_pos + 1])

        dot = (q_rot[0, 0, 0] * k_rot[0, 0, 0]).sum().item()
        dot_products.append(dot)

    assert len(dot_products) >= 3, "test setup error: not enough valid (query_pos, key_pos) pairs"
    for d in dot_products[1:]:
        assert abs(d - dot_products[0]) < 1e-3, (
            f"dot products at the same relative offset should match: {dot_products}"
        )


def test_rope_shapes_match_input():
    head_dim = 64
    cos, sin = compute_rope_frequencies(head_dim, max_seq_len=128, theta=500000.0)
    x = torch.randn(3, 8, 20, head_dim)
    rotated = apply_rope(x, cos[:20], sin[:20])
    assert rotated.shape == x.shape


def test_rope_frequency_table_shape():
    head_dim = 64
    max_seq_len = 128
    cos, sin = compute_rope_frequencies(head_dim, max_seq_len, theta=500000.0)
    assert cos.shape == (max_seq_len, head_dim)
    assert sin.shape == (max_seq_len, head_dim)
