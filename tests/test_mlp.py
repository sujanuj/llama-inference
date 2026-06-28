"""Tests for the SwiGLU MLP block."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.mlp import swiglu_mlp


def test_swiglu_matches_hand_computed_small_case():
    # hidden_size=2, intermediate_size=2, identity-like projections so
    # the math is traceable by hand: gate_proj and up_proj both = I,
    # down_proj = I, so gate = SiLU(x), up = x, output = SiLU(x) * x,
    # passed through identity.
    x = torch.tensor([[1.0, 2.0]])
    identity = torch.eye(2)

    out = swiglu_mlp(x, identity, identity, identity)

    def silu(v):
        return v / (1 + math.exp(-v))

    expected = [silu(1.0) * 1.0, silu(2.0) * 2.0]
    got = out[0].tolist()
    for g, e in zip(got, expected):
        assert abs(g - e) < 1e-5, f"got {g}, want {e}"


def test_swiglu_gate_actually_gates_not_just_passes_through():
    # If gate_proj produces a large NEGATIVE pre-activation, SiLU(x) for
    # very negative x approaches 0 (SiLU(-10) ~ -0.00045, near zero) —
    # so the gated output should be driven toward zero even though
    # up_proj's output alone would not be. This distinguishes real
    # gating from a bug where the multiplication is accidentally
    # skipped or replaced with addition.
    hidden_size, intermediate_size = 4, 4
    x = torch.ones(1, hidden_size)

    # gate_proj maps x to a large negative value in every intermediate
    # channel; up_proj maps x to a large positive value.
    gate_proj = torch.full((hidden_size, intermediate_size), -10.0 / hidden_size)
    up_proj = torch.full((hidden_size, intermediate_size), 100.0 / hidden_size)
    down_proj = torch.eye(intermediate_size)

    out = swiglu_mlp(x, gate_proj, up_proj, down_proj)

    # up alone (no gating) would be 100; with the strongly negative
    # gate suppressing it via SiLU, output should be much smaller.
    assert out.abs().max().item() < 5.0, (
        f"gating should suppress the large up_proj output, got max {out.abs().max().item()}"
    )


def test_swiglu_zero_input_produces_zero_output():
    x = torch.zeros(1, 8)
    gate_proj = torch.randn(8, 16)
    up_proj = torch.randn(8, 16)
    down_proj = torch.randn(16, 8)

    out = swiglu_mlp(x, gate_proj, up_proj, down_proj)

    assert torch.allclose(out, torch.zeros(1, 8), atol=1e-6)


def test_swiglu_shapes_for_batched_multidim_input():
    batch, seq_len, hidden_size, intermediate_size = 2, 5, 16, 32
    x = torch.randn(batch, seq_len, hidden_size)
    gate_proj = torch.randn(hidden_size, intermediate_size)
    up_proj = torch.randn(hidden_size, intermediate_size)
    down_proj = torch.randn(intermediate_size, hidden_size)

    out = swiglu_mlp(x, gate_proj, up_proj, down_proj)

    assert out.shape == (batch, seq_len, hidden_size)
    assert not torch.isnan(out).any()


def test_swiglu_matches_llama_3_2_1b_dimensions():
    # Sanity check against the actual config dimensions, so a future
    # change to model/config.py that breaks this wiring would be caught
    # here rather than only at real-weight-loading time.
    from model.config import LLAMA_3_2_1B

    batch, seq_len = 1, 3
    x = torch.randn(batch, seq_len, LLAMA_3_2_1B.hidden_size)
    gate_proj = torch.randn(LLAMA_3_2_1B.hidden_size, LLAMA_3_2_1B.intermediate_size)
    up_proj = torch.randn(LLAMA_3_2_1B.hidden_size, LLAMA_3_2_1B.intermediate_size)
    down_proj = torch.randn(LLAMA_3_2_1B.intermediate_size, LLAMA_3_2_1B.hidden_size)

    out = swiglu_mlp(x, gate_proj, up_proj, down_proj)

    assert out.shape == (batch, seq_len, LLAMA_3_2_1B.hidden_size)
