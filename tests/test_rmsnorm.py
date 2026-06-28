"""Tests for RMSNorm.

The reference values here are computed independently with plain Python
math (no torch), so a bug in the torch implementation can't accidentally
agree with a buggy "reference" written using the same library.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.rmsnorm import rms_norm


def test_rms_norm_matches_hand_computed_reference():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weight = torch.ones(4)
    eps = 1e-5

    # Hand-computed in plain Python, not torch, so this can't share a
    # bug with the implementation under test.
    values = [1.0, 2.0, 3.0, 4.0]
    mean_sq = sum(v * v for v in values) / len(values)
    denom = math.sqrt(mean_sq + eps)
    expected = [v / denom for v in values]

    got = rms_norm(x, weight, eps)[0].tolist()
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-5, f"got {g}, want {e}"


def test_rms_norm_weight_scales_output_elementwise():
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    weight = torch.tensor([2.0, 0.5, 1.0, 3.0])
    eps = 1e-5

    out = rms_norm(x, weight, eps)[0]

    # With all-equal inputs, the normalized (pre-weight) values are all
    # equal too, so the output should be EXACTLY proportional to weight.
    ratios = (out / weight).tolist()
    for r in ratios[1:]:
        assert abs(r - ratios[0]) < 1e-5, f"ratios should all match: {ratios}"


def test_rms_norm_output_rms_is_approximately_one_before_weighting():
    # Defining property of RMSNorm: after normalizing (before applying
    # the learned weight), the root-mean-square of the output should be
    # very close to 1 — that's the entire point of the operation.
    torch.manual_seed(0)
    x = torch.randn(8, 16) * 5.0  # arbitrary scale, shouldn't matter
    weight = torch.ones(16)
    eps = 1e-8

    out = rms_norm(x, weight, eps)
    rms = out.pow(2).mean(dim=-1).sqrt()

    assert torch.allclose(rms, torch.ones(8), atol=1e-3), rms


def test_rms_norm_handles_zero_input_without_nan():
    x = torch.zeros(1, 4)
    weight = torch.ones(4)
    eps = 1e-5

    out = rms_norm(x, weight, eps)

    assert not torch.isnan(out).any()
    assert torch.allclose(out, torch.zeros(1, 4))


def test_rms_norm_preserves_shape_for_batched_multidim_input():
    x = torch.randn(2, 3, 5, 16)
    weight = torch.randn(16)
    out = rms_norm(x, weight, 1e-5)
    assert out.shape == x.shape


def test_rms_norm_upcasts_bf16_internally_for_stability():
    # The implementation upcasts to fp32 internally regardless of input
    # dtype. This test doesn't check the upcast directly (that's an
    # implementation detail) but checks the OBSERVABLE consequence: a
    # bf16 input with values that would lose significant precision if
    # squared/averaged in bf16 itself should still produce a sane,
    # non-degenerate result.
    x = torch.full((1, 8), 1000.0, dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=torch.bfloat16)
    out = rms_norm(x, weight, 1e-5)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    # All-equal inputs should normalize to all-equal outputs near 1.0.
    out_f32 = out.to(torch.float32)
    assert torch.allclose(out_f32, torch.ones_like(out_f32), atol=0.05)
