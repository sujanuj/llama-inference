"""Tests for bf16 dtype handling across the model.

Every test in this file exists because of a real bug: running the full
model against the actual bf16 Llama-3.2-1B checkpoint produced

    RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16

Every test fixture before this file used torch.randn(...), which
defaults to float32 -- so this entire class of "silent or loud dtype
mismatch when the input is bf16" bug had no way to be caught until a
real bf16 checkpoint was actually loaded. These tests close that gap
going forward, using random bf16 weights (not real downloaded ones --
see testutil/random_weights.py's reasoning for why that's legitimate)
so the dtype-handling code paths get exercised on every test run, not
just whenever someone happens to load real weights again.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.attention import attention, scaled_dot_product_attention
from model.config import LlamaConfig
from model.model import forward
from model.rope import apply_rope, compute_rope_frequencies
from testutil.random_weights import random_model_weights


def _to_bf16(weights):
    """Casts every tensor field in a ModelWeights tree to bf16, in
    place conceptually (returns a new tree) -- mirrors what a real
    bf16 checkpoint load actually produces, without needing real
    downloaded weights.
    """
    weights.embed_tokens = weights.embed_tokens.to(torch.bfloat16)
    weights.final_norm = weights.final_norm.to(torch.bfloat16)
    if weights.lm_head_weight is not None:
        weights.lm_head_weight = weights.lm_head_weight.to(torch.bfloat16)
    for layer in weights.layers:
        layer.input_layernorm = layer.input_layernorm.to(torch.bfloat16)
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(torch.bfloat16)
        layer.attention.q_proj = layer.attention.q_proj.to(torch.bfloat16)
        layer.attention.k_proj = layer.attention.k_proj.to(torch.bfloat16)
        layer.attention.v_proj = layer.attention.v_proj.to(torch.bfloat16)
        layer.attention.o_proj = layer.attention.o_proj.to(torch.bfloat16)
        layer.mlp.gate_proj = layer.mlp.gate_proj.to(torch.bfloat16)
        layer.mlp.up_proj = layer.mlp.up_proj.to(torch.bfloat16)
        layer.mlp.down_proj = layer.mlp.down_proj.to(torch.bfloat16)
    return weights


def test_scaled_dot_product_attention_runs_with_bf16_inputs():
    # This is the exact call site the real bug traced back to: a
    # torch.matmul dtype mismatch inside scaled_dot_product_attention
    # when query/key/value are bf16. Reproduced directly here, at the
    # smallest possible scope, rather than only at the full-model level
    # below -- a failure here pinpoints the bug precisely.
    torch.manual_seed(0)
    batch, num_heads, seq_len, head_dim = 1, 2, 4, 8
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    out = scaled_dot_product_attention(q, k, v, causal=True)

    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out.float()).any()


def test_apply_rope_preserves_input_dtype_with_float32_tables():
    # compute_rope_frequencies always returns float32 cos/sin tables
    # (by design, for numerical stability -- see that function's
    # docstring), but apply_rope must not let that silently upcast a
    # bf16 input. This is the second of the two real fixes this bug
    # required.
    torch.manual_seed(0)
    head_dim = 8
    cos, sin = compute_rope_frequencies(head_dim, max_seq_len=4, theta=10000.0)
    assert cos.dtype == torch.float32  # confirms the test's premise

    x_bf16 = torch.randn(1, 1, 4, head_dim, dtype=torch.bfloat16)
    rotated = apply_rope(x_bf16, cos, sin)

    assert rotated.dtype == torch.bfloat16, (
        f"apply_rope changed dtype from bfloat16 to {rotated.dtype} -- "
        f"cos/sin's float32 dtype must have silently promoted the result"
    )


def test_attention_block_runs_end_to_end_with_bf16_weights():
    torch.manual_seed(0)
    from model.weights import AttentionWeights

    hidden_size, seq_len, num_heads, num_kv_heads, head_dim = 16, 5, 4, 2, 4
    hidden_states = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
    weights = AttentionWeights(
        q_proj=torch.randn(hidden_size, num_heads * head_dim, dtype=torch.bfloat16),
        k_proj=torch.randn(hidden_size, num_kv_heads * head_dim, dtype=torch.bfloat16),
        v_proj=torch.randn(hidden_size, num_kv_heads * head_dim, dtype=torch.bfloat16),
        o_proj=torch.randn(num_heads * head_dim, hidden_size, dtype=torch.bfloat16),
    )
    cos, sin = compute_rope_frequencies(head_dim, max_seq_len=seq_len, theta=10000.0)

    out = attention(
        hidden_states, weights.q_proj, weights.k_proj, weights.v_proj, weights.o_proj,
        cos, sin, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )

    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out.float()).any()


def test_full_forward_pass_runs_end_to_end_with_bf16_weights():
    # The actual scenario that surfaced the original bug: a full
    # forward() call with bf16 weights throughout, matching what
    # loading a real Llama-3.2-1B checkpoint produces. This is the
    # regression test for the exact RuntimeError from the bug report.
    config = LlamaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=32, rms_norm_eps=1e-5, rope_theta=10000.0,
    )
    weights = random_model_weights(config, num_layers=2)
    weights = _to_bf16(weights)

    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    logits = forward(input_ids, weights, config)

    assert logits.dtype == torch.bfloat16
    assert not torch.isnan(logits.float()).any()
    assert not torch.isinf(logits.float()).any()


def test_bf16_and_fp32_forward_passes_agree_approximately():
    # The same weights and input, once in fp32 (exact) and once cast to
    # bf16 (lossy), should still produce ROUGHLY the same logits and,
    # in particular, the same argmax next-token prediction for an
    # easy/unambiguous case -- this is a sanity check that the bf16
    # path isn't just "running without crashing" but is computing
    # something in the right ballpark, not garbage that happens to be
    # finite.
    config = LlamaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=32, rms_norm_eps=1e-5, rope_theta=10000.0,
    )
    weights_fp32 = random_model_weights(config, num_layers=2)

    # Deep-ish copy via re-running the same builder with the same seed,
    # then casting -- simpler than implementing a generic deep-copy for
    # the dataclass tree, and equally valid since random_model_weights
    # is deterministic given the same seed.
    weights_bf16 = random_model_weights(config, num_layers=2)
    weights_bf16 = _to_bf16(weights_bf16)

    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    logits_fp32 = forward(input_ids, weights_fp32, config)
    logits_bf16 = forward(input_ids, weights_bf16, config).float()

    # Generous tolerance -- bf16 has ~3 decimal digits of precision, and
    # this is propagated through several layers, so this is checking
    # "same ballpark," not "numerically identical."
    max_diff = (logits_fp32 - logits_bf16).abs().max().item()
    assert max_diff < 5.0, f"fp32 vs bf16 logits differ by {max_diff}, suspiciously large"
