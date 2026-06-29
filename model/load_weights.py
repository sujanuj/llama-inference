"""Loads a real Llama-3.2-1B checkpoint (HuggingFace safetensors format)
into this project's ModelWeights/DecoderLayerWeights dataclasses.

The single most important, most easily-gotten-wrong detail in this file:
HuggingFace's nn.Linear stores weights as (out_features, in_features) and
computes y = x @ W.T + b. Every model/*.py forward-pass function in this
project computes y = x @ W directly (no transpose) — a deliberate choice
made back in Phase 1-3 to keep the raw tensor math simple and readable.
That means EVERY weight loaded from a real checkpoint must be
TRANSPOSED before going into this project's dataclasses, or the shapes
will either fail loudly (good) or, for any square weight matrix, silently
multiply with completely wrong semantics (bad — exactly the failure mode
this whole project's testing philosophy exists to catch). Every load_*
helper below transposes explicitly and comments on it, rather than
relying on a single get-it-right-once transpose buried in a shared
helper, specifically so a future edit to one weight's loading code can't
accidentally skip the transpose for another.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from model.config import LlamaConfig
from model.weights import AttentionWeights, DecoderLayerWeights, MLPWeights, ModelWeights


class CheckpointTensors:
    """Thin wrapper over one or more safetensors files, indexed by the
    HuggingFace parameter name, so the rest of this module can do
    `tensors["model.layers.0.self_attn.q_proj.weight"]` without caring
    whether the checkpoint is split across multiple shard files (real
    checkpoints over a certain size are sharded; Llama-3.2-1B itself is
    small enough to usually ship as a single file, but handling the
    sharded case costs little and avoids a surprise on a different model
    size later).
    """

    def __init__(self, tensors: dict):
        self._tensors = tensors

    @classmethod
    def from_directory(cls, checkpoint_dir: str) -> "CheckpointTensors":
        directory = Path(checkpoint_dir)
        safetensor_files = sorted(directory.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {checkpoint_dir} — "
                f"did the download actually complete?"
            )
        tensors = {}
        for f in safetensor_files:
            tensors.update(load_file(str(f)))
        return cls(tensors)

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._tensors:
            raise KeyError(
                f"Expected parameter {name!r} not found in checkpoint. "
                f"This usually means either the config's layer/head counts "
                f"don't match the actual checkpoint, or the naming "
                f"convention assumed by this loader doesn't match this "
                f"particular checkpoint's format."
            )
        return self._tensors[name]

    def has(self, name: str) -> bool:
        return name in self._tensors


def load_config_from_directory(checkpoint_dir: str) -> LlamaConfig:
    """Reads config.json from the checkpoint directory and builds a
    LlamaConfig from it, rather than trusting the hardcoded
    model/config.py constants to still be accurate. If the checkpoint's
    actual config disagrees with model/config.py's LLAMA_3_2_1B in any
    field that matters, that disagreement should surface explicitly
    (see verify_config_matches below), not be silently overridden in one
    direction or the other.
    """
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path) as f:
        raw = json.load(f)

    head_dim = raw.get("head_dim")
    if head_dim is None:
        head_dim = raw["hidden_size"] // raw["num_attention_heads"]

    return LlamaConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=head_dim,
        intermediate_size=raw["intermediate_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        max_position_embeddings=raw["max_position_embeddings"],
    )


def verify_config_matches(checkpoint_config: LlamaConfig, expected: LlamaConfig) -> None:
    """Raises a clear, specific error if any architecturally-significant
    field disagrees, rather than letting a mismatch manifest later as a
    confusing shape error deep inside a matmul. Called once, right after
    loading, so a wrong assumption anywhere in model/config.py is caught
    immediately and at the source.
    """
    fields_to_check = [
        "vocab_size", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "intermediate_size",
    ]
    mismatches = []
    for field_name in fields_to_check:
        actual = getattr(checkpoint_config, field_name)
        wanted = getattr(expected, field_name)
        if actual != wanted:
            mismatches.append(f"{field_name}: checkpoint has {actual}, model/config.py assumed {wanted}")
    if mismatches:
        raise ValueError(
            "Checkpoint config disagrees with model/config.py's assumptions:\n  "
            + "\n  ".join(mismatches)
        )


def _load_attention_weights(tensors: CheckpointTensors, layer_idx: int) -> AttentionWeights:
    prefix = f"model.layers.{layer_idx}.self_attn"
    # .T on every projection: see module docstring for why this is
    # mandatory, not optional, for every single one of these four.
    return AttentionWeights(
        q_proj=tensors[f"{prefix}.q_proj.weight"].T,
        k_proj=tensors[f"{prefix}.k_proj.weight"].T,
        v_proj=tensors[f"{prefix}.v_proj.weight"].T,
        o_proj=tensors[f"{prefix}.o_proj.weight"].T,
    )


def _load_mlp_weights(tensors: CheckpointTensors, layer_idx: int) -> MLPWeights:
    prefix = f"model.layers.{layer_idx}.mlp"
    return MLPWeights(
        gate_proj=tensors[f"{prefix}.gate_proj.weight"].T,
        up_proj=tensors[f"{prefix}.up_proj.weight"].T,
        down_proj=tensors[f"{prefix}.down_proj.weight"].T,
    )


def _load_decoder_layer_weights(tensors: CheckpointTensors, layer_idx: int) -> DecoderLayerWeights:
    prefix = f"model.layers.{layer_idx}"
    return DecoderLayerWeights(
        # RMSNorm weights are 1-D (hidden_size,) — no transpose
        # applicable or needed, unlike every 2-D projection above.
        input_layernorm=tensors[f"{prefix}.input_layernorm.weight"],
        attention=_load_attention_weights(tensors, layer_idx),
        post_attention_layernorm=tensors[f"{prefix}.post_attention_layernorm.weight"],
        mlp=_load_mlp_weights(tensors, layer_idx),
    )


def load_model_weights(checkpoint_dir: str, config: LlamaConfig) -> ModelWeights:
    """Loads a full ModelWeights from a HuggingFace-format checkpoint
    directory (containing config.json and one or more .safetensors
    files).
    """
    tensors = CheckpointTensors.from_directory(checkpoint_dir)

    # (vocab_size, hidden_size) -- already the shape this project's
    # embedding lookup expects (weights.embed_tokens[input_ids] indexes
    # the FIRST dimension by token id), so no transpose here -- this is
    # an embedding TABLE, not a Linear projection, and HuggingFace
    # stores it in exactly the layout this project needs.
    embed_tokens = tensors["model.embed_tokens.weight"]

    layers = [
        _load_decoder_layer_weights(tensors, i) for i in range(config.num_hidden_layers)
    ]

    final_norm = tensors["model.norm.weight"]

    # Tied-embeddings detection: real Llama-3.2-1B/3B checkpoints simply
    # do not have an "lm_head.weight" tensor at all (see model/weights.py
    # for the full reasoning) -- checking tensors.has(...) rather than
    # assuming based on model size keeps this correct even if a future
    # checkpoint variant changes that convention.
    lm_head_weight = None
    if tensors.has("lm_head.weight"):
        # NOT transposed here -- see ModelWeights.output_projection(),
        # which itself calls .T on this field when using it. Storing it
        # un-transposed keeps this field's convention symmetric with
        # embed_tokens (both stored as (vocab_size, hidden_size)),
        # rather than this loader pre-transposing some fields and not
        # others.
        lm_head_weight = tensors["lm_head.weight"]

    return ModelWeights(
        embed_tokens=embed_tokens,
        layers=layers,
        final_norm=final_norm,
        lm_head_weight=lm_head_weight,
    )
