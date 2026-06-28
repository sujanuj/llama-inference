"""Test-only helper: builds a ModelWeights instance with random tensors
of the CORRECT shapes for a given config, so the full forward pass can
be exercised end-to-end without needing real downloaded weights.

This is legitimate, not a workaround: architecture correctness (do the
shapes line up? does the data flow through every layer without crashing
or producing NaN? does causal masking behave correctly across the whole
stack?) doesn't depend on which specific numbers are in the weight
tensors. The question "does this match REAL Llama-3.2-1B's actual
outputs" is a separate, later verification step that needs real
downloaded weights — this helper exists for everything that comes
before that point.
"""

import torch

from model.config import LlamaConfig
from model.weights import AttentionWeights, DecoderLayerWeights, MLPWeights, ModelWeights


def random_model_weights(
    config: LlamaConfig, num_layers=None, seed: int = 0
) -> ModelWeights:
    """Build a ModelWeights with random tensors matching config's shapes.

    num_layers can override config.num_hidden_layers to build a smaller
    model for fast tests — the per-layer weight SHAPES still come from
    config, only the layer COUNT is overridden, so this still exercises
    the real per-layer dimensions.
    """
    torch.manual_seed(seed)
    n_layers = num_layers if num_layers is not None else config.num_hidden_layers

    embed_tokens = torch.randn(config.vocab_size, config.hidden_size) * 0.02

    layers = []
    for _ in range(n_layers):
        attn = AttentionWeights(
            q_proj=torch.randn(config.hidden_size, config.num_attention_heads * config.head_dim) * 0.02,
            k_proj=torch.randn(config.hidden_size, config.num_key_value_heads * config.head_dim) * 0.02,
            v_proj=torch.randn(config.hidden_size, config.num_key_value_heads * config.head_dim) * 0.02,
            o_proj=torch.randn(config.num_attention_heads * config.head_dim, config.hidden_size) * 0.02,
        )
        mlp = MLPWeights(
            gate_proj=torch.randn(config.hidden_size, config.intermediate_size) * 0.02,
            up_proj=torch.randn(config.hidden_size, config.intermediate_size) * 0.02,
            down_proj=torch.randn(config.intermediate_size, config.hidden_size) * 0.02,
        )
        layers.append(
            DecoderLayerWeights(
                input_layernorm=torch.ones(config.hidden_size),
                attention=attn,
                post_attention_layernorm=torch.ones(config.hidden_size),
                mlp=mlp,
            )
        )

    return ModelWeights(
        embed_tokens=embed_tokens,
        layers=layers,
        final_norm=torch.ones(config.hidden_size),
        lm_head_weight=None,  # tied embeddings, matching real Llama-3.2-1B/3B
    )
