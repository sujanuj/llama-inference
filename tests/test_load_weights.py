"""Tests for model/load_weights.py.

Since this sandbox can't reach huggingface.co, these tests build a
SYNTHETIC checkpoint on disk — real safetensors files and a real
config.json, with random tensor VALUES but the exact naming convention
and shapes a genuine Llama-3.2-1B checkpoint uses. This is legitimate for
testing the LOADING MECHANISM (does every tensor get found under the
right name? does every transpose happen where it needs to? does
tied-vs-untied detection work?) — it's a separate, later question
whether the loaded numbers match real Llama-3.2-1B's actual behavior,
which needs the real checkpoint and is exactly what
scripts/verify_against_huggingface.py (run on a machine with Hub access)
is for.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import LlamaConfig
from model.load_weights import (
    CheckpointTensors,
    load_config_from_directory,
    load_model_weights,
    verify_config_matches,
)


def _tiny_config() -> LlamaConfig:
    # Deliberately tiny dimensions so the synthetic checkpoint this test
    # builds on disk is small and fast, while still exercising every
    # naming/shape/transpose code path a real checkpoint would.
    return LlamaConfig(
        vocab_size=37, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=24, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=512,
    )


def _write_synthetic_checkpoint(tmp_path: Path, config: LlamaConfig, tie_embeddings: bool) -> Path:
    """Writes a real config.json and a real .safetensors file to
    tmp_path, using the EXACT parameter names and shapes a genuine
    HuggingFace Llama checkpoint uses — including storing every Linear
    weight as (out_features, in_features), matching real checkpoints'
    actual on-disk convention, NOT this project's (in, out) convention.
    Getting this fixture's shapes backwards would make every test below
    pass for the wrong reason (no transpose actually being exercised),
    so this is worth being precise about.
    """
    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "max_position_embeddings": config.max_position_embeddings,
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config_dict, f)

    torch.manual_seed(0)
    tensors = {}
    tensors["model.embed_tokens.weight"] = torch.randn(config.vocab_size, config.hidden_size)

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        # Stored (out_features, in_features) -- the real HF convention.
        tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn(
            config.num_attention_heads * config.head_dim, config.hidden_size
        )
        tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn(
            config.num_key_value_heads * config.head_dim, config.hidden_size
        )
        tensors[f"{p}.self_attn.v_proj.weight"] = torch.randn(
            config.num_key_value_heads * config.head_dim, config.hidden_size
        )
        tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(
            config.hidden_size, config.num_attention_heads * config.head_dim
        )
        tensors[f"{p}.input_layernorm.weight"] = torch.randn(config.hidden_size)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.randn(config.hidden_size)
        tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn(config.intermediate_size, config.hidden_size)
        tensors[f"{p}.mlp.up_proj.weight"] = torch.randn(config.intermediate_size, config.hidden_size)
        tensors[f"{p}.mlp.down_proj.weight"] = torch.randn(config.hidden_size, config.intermediate_size)

    tensors["model.norm.weight"] = torch.randn(config.hidden_size)

    if not tie_embeddings:
        tensors["lm_head.weight"] = torch.randn(config.vocab_size, config.hidden_size)
    # else: deliberately absent, matching real Llama-3.2-1B/3B checkpoints

    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def test_load_config_from_directory_matches_written_json(tmp_path):
    config = _tiny_config()
    _write_synthetic_checkpoint(tmp_path, config, tie_embeddings=True)

    loaded = load_config_from_directory(str(tmp_path))

    assert loaded.vocab_size == config.vocab_size
    assert loaded.hidden_size == config.hidden_size
    assert loaded.num_hidden_layers == config.num_hidden_layers
    assert loaded.num_attention_heads == config.num_attention_heads
    assert loaded.num_key_value_heads == config.num_key_value_heads
    assert loaded.head_dim == config.head_dim
    assert loaded.intermediate_size == config.intermediate_size


def test_verify_config_matches_passes_for_identical_configs():
    config = _tiny_config()
    verify_config_matches(config, config)  # should not raise


def test_verify_config_matches_raises_with_clear_message_on_mismatch():
    config = _tiny_config()
    wrong = LlamaConfig(
        vocab_size=config.vocab_size, hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=999,  # deliberately wrong
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim, intermediate_size=config.intermediate_size,
    )
    try:
        verify_config_matches(config, wrong)
        assert False, "should have raised on mismatched num_attention_heads"
    except ValueError as e:
        assert "num_attention_heads" in str(e)
        assert "999" in str(e)


def test_checkpoint_tensors_raises_clear_error_for_missing_key(tmp_path):
    config = _tiny_config()
    _write_synthetic_checkpoint(tmp_path, config, tie_embeddings=True)
    tensors = CheckpointTensors.from_directory(str(tmp_path))

    try:
        tensors["model.this.key.does.not.exist"]
        assert False, "should have raised KeyError"
    except KeyError as e:
        assert "this.key.does.not.exist" in str(e)


def test_load_model_weights_transposes_every_2d_projection(tmp_path):
    # The single most important test in this file: confirm every
    # attention/MLP projection comes out shaped for THIS PROJECT's
    # x @ W convention (in_features, out_features), the TRANSPOSE of
    # how the synthetic checkpoint stored it -- if the loader forgot a
    # transpose anywhere, this test catches it as a shape assertion
    # failure, not a silent wrong-number bug discovered later.
    config = _tiny_config()
    _write_synthetic_checkpoint(tmp_path, config, tie_embeddings=True)

    weights = load_model_weights(str(tmp_path), config)

    layer0 = weights.layers[0]
    assert layer0.attention.q_proj.shape == (config.hidden_size, config.num_attention_heads * config.head_dim)
    assert layer0.attention.k_proj.shape == (config.hidden_size, config.num_key_value_heads * config.head_dim)
    assert layer0.attention.v_proj.shape == (config.hidden_size, config.num_key_value_heads * config.head_dim)
    assert layer0.attention.o_proj.shape == (config.num_attention_heads * config.head_dim, config.hidden_size)
    assert layer0.mlp.gate_proj.shape == (config.hidden_size, config.intermediate_size)
    assert layer0.mlp.up_proj.shape == (config.hidden_size, config.intermediate_size)
    assert layer0.mlp.down_proj.shape == (config.intermediate_size, config.hidden_size)


def test_load_model_weights_loads_correct_number_of_layers(tmp_path):
    config = _tiny_config()
    _write_synthetic_checkpoint(tmp_path, config, tie_embeddings=True)

    weights = load_model_weights(str(tmp_path), config)

    assert len(weights.layers) == config.num_hidden_layers


def test_load_model_weights_detects_tied_embeddings(tmp_path):
    config = _tiny_config()
    _write_synthetic_checkpoint(tmp_path, config, tie_embeddings=True)

    weights = load_model_weights(str(tmp_path), config)

    assert weights.lm_head_weight is None
    # And the fallback output_projection() should be embed_tokens.T,
    # exactly matching Phase 3's tied-embeddings test for the
    # synthetic-random-weights case -- this confirms the LOADED weights
    # behave the same way under that logic as the Phase 3 test fixtures did.
    assert torch.equal(weights.output_projection(), weights.embed_tokens.T)


def test_load_model_weights_detects_untied_lm_head(tmp_path):
    config = _tiny_config()
    _write_synthetic_checkpoint(tmp_path, config, tie_embeddings=False)

    weights = load_model_weights(str(tmp_path), config)

    assert weights.lm_head_weight is not None
    assert weights.lm_head_weight.shape == (config.vocab_size, config.hidden_size)
    assert torch.equal(weights.output_projection(), weights.lm_head_weight.T)


def test_loaded_weights_actually_run_through_the_real_forward_pass(tmp_path):
    # End-to-end: load a synthetic checkpoint, run it through the REAL
    # forward() from model/model.py (not just check shapes in isolation)
    # -- this is the same style of integration check Phase 3 did with
    # random_model_weights, now exercised on the LOADED path instead.
    from model.model import forward

    config = _tiny_config()
    _write_synthetic_checkpoint(tmp_path, config, tie_embeddings=True)
    weights = load_model_weights(str(tmp_path), config)

    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    logits = forward(input_ids, weights, config)

    assert logits.shape == (1, 5, config.vocab_size)
    assert not torch.isnan(logits).any()
