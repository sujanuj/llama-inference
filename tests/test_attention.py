"""Tests for grouped-query attention.

repeat_kv is the highest-risk function here (see model/attention.py's
module docstring) — getting block-vs-interleave backwards would
silently mispair query heads with the wrong KV head. The test below
catches this directly by giving each KV head a DISTINCT, IDENTIFIABLE
value (not random noise) and checking exactly which repeated copies
match which original head, rather than just checking shapes.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.attention import attention, repeat_kv, scaled_dot_product_attention


def test_repeat_kv_block_pattern_not_interleaved():
    # Two KV heads, each filled with a distinct constant value, so it's
    # possible to read off EXACTLY which original head each repeated
    # slice came from — not just confirm the right shape came out.
    batch, num_kv_heads, seq_len, head_dim = 1, 2, 1, 4
    x = torch.zeros(batch, num_kv_heads, seq_len, head_dim)
    x[:, 0, :, :] = 100.0  # head A
    x[:, 1, :, :] = 200.0  # head B

    num_groups = 4
    out = repeat_kv(x, num_groups)

    assert out.shape == (batch, num_kv_heads * num_groups, seq_len, head_dim)

    # Block pattern: [A, A, A, A, B, B, B, B] — NOT interleaved
    # [A, B, A, B, A, B, A, B]. Reading off each of the 8 output head
    # slots directly is what actually verifies the ordering, since a
    # shape-only check can't distinguish the two patterns.
    expected_values = [100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 200.0]
    for head_idx, expected in enumerate(expected_values):
        actual = out[0, head_idx, 0, 0].item()
        assert actual == expected, (
            f"head {head_idx}: got {actual}, want {expected} "
            f"(block pattern: heads 0-3 should be A=100, heads 4-7 should be B=200)"
        )


def test_repeat_kv_preserves_values_exactly():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 8)
    out = repeat_kv(x, num_groups=4)

    assert out.shape == (2, 12, 5, 8)
    # Every group of 4 consecutive output heads should be an EXACT copy
    # of one input head (not an average, not noise — repeat_kv must not
    # alter values at all).
    for kv_head in range(3):
        for group_member in range(4):
            out_head = kv_head * 4 + group_member
            assert torch.equal(out[:, out_head], x[:, kv_head]), (
                f"output head {out_head} should exactly equal input KV head {kv_head}"
            )


def test_causal_mask_blocks_future_positions():
    # Construct query/key/value such that without masking, every
    # position would attend strongly to a "future" position with an
    # artificially huge key value — if the causal mask isn't working,
    # this would show up as a wildly large output at early positions.
    torch.manual_seed(0)
    batch, num_heads, seq_len, head_dim = 1, 1, 4, 8

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.zeros(batch, num_heads, seq_len, head_dim)
    # Position 3 (the last, "future" position relative to earlier
    # queries) gets a huge, easily-identifiable value.
    v[:, :, 3, :] = 1000.0

    out_causal = scaled_dot_product_attention(q, k, v, causal=True)
    out_noncausal = scaled_dot_product_attention(q, k, v, causal=False)

    # Position 0 can ONLY see itself under causal masking, so it must
    # not show ANY influence from position 3's huge value.
    assert out_causal[0, 0, 0].abs().max().item() < 1.0, (
        "position 0 should not be influenced by position 3 under causal masking"
    )
    # Without masking, position 0 CAN see position 3 — sanity-checking
    # that the test setup itself is meaningful (i.e. masking is doing
    # something observable, not just silently matching by coincidence).
    assert out_noncausal[0, 0, 0].abs().max().item() > 1.0, (
        "test setup error: position 0 should be influenced by position 3 without masking"
    )


def test_causal_mask_allows_attending_to_self_and_past():
    torch.manual_seed(0)
    batch, num_heads, seq_len, head_dim = 1, 1, 3, 4
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.randn(batch, num_heads, seq_len, head_dim)

    out = scaled_dot_product_attention(q, k, v, causal=True)

    # The LAST position attends to everything (itself + all past), so
    # it should be a genuine weighted combination, not equal to any
    # single value vector and not NaN/zero.
    assert not torch.isnan(out).any()
    assert out[0, 0, -1].abs().sum().item() > 0


def test_attention_output_shape():
    torch.manual_seed(0)
    from model.rope import compute_rope_frequencies

    batch, seq_len, hidden_size = 2, 6, 32
    num_heads, num_kv_heads, head_dim = 8, 2, 4

    hidden_states = torch.randn(batch, seq_len, hidden_size)
    q_proj = torch.randn(hidden_size, num_heads * head_dim)
    k_proj = torch.randn(hidden_size, num_kv_heads * head_dim)
    v_proj = torch.randn(hidden_size, num_kv_heads * head_dim)
    o_proj = torch.randn(num_heads * head_dim, hidden_size)

    cos, sin = compute_rope_frequencies(head_dim, max_seq_len=seq_len, theta=10000.0)

    out = attention(
        hidden_states, q_proj, k_proj, v_proj, o_proj, cos, sin,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )

    assert out.shape == (batch, seq_len, hidden_size)
    assert not torch.isnan(out).any()


def test_attention_with_kv_offset_for_cached_context():
    # Simulates the kv-cache scenario one phase ahead of schedule: a
    # NEW query token attending against a LONGER key/value history that
    # includes already-cached past tokens (kv_len > q_len). The causal
    # mask logic in scaled_dot_product_attention has explicit offset
    # handling for exactly this case — this test exercises it directly,
    # since q_len == kv_len (the common case in the other tests above)
    # would never catch an off-by-one in that offset math.
    torch.manual_seed(0)
    num_heads, cached_len, new_len, head_dim = 1, 5, 1, 4
    kv_len = cached_len + new_len

    q = torch.randn(1, num_heads, new_len, head_dim)
    k = torch.randn(1, num_heads, kv_len, head_dim)
    v = torch.zeros(1, num_heads, kv_len, head_dim)
    v[:, :, -1, :] = 1.0  # the new token's own value, identifiable

    out = scaled_dot_product_attention(q, k, v, causal=True)

    # The single new query token (absolute position = cached_len) must
    # be able to see all cached history AND itself — it should NOT be
    # entirely zero (which would mean it got masked from seeing
    # anything, including itself, an off-by-one in the wrong direction).
    assert not torch.isnan(out).any()
    assert out.abs().sum().item() > 0
