"""Weight containers for the full model.

Plain dataclasses, not nn.Module/nn.Parameter — these just hold tensors
in a structured, named way so the forward-pass functions (decoder.py,
model.py) and the future weight-loading code both have a clear, typed
place to read from and write to, rather than threading raw dicts or
positional lists through everything.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class AttentionWeights:
    q_proj: torch.Tensor  # (hidden_size, num_heads * head_dim)
    k_proj: torch.Tensor  # (hidden_size, num_kv_heads * head_dim)
    v_proj: torch.Tensor  # (hidden_size, num_kv_heads * head_dim)
    o_proj: torch.Tensor  # (num_heads * head_dim, hidden_size)


@dataclass
class MLPWeights:
    gate_proj: torch.Tensor  # (hidden_size, intermediate_size)
    up_proj: torch.Tensor  # (hidden_size, intermediate_size)
    down_proj: torch.Tensor  # (intermediate_size, hidden_size)


@dataclass
class DecoderLayerWeights:
    input_layernorm: torch.Tensor  # (hidden_size,) — pre-attention RMSNorm
    attention: AttentionWeights
    post_attention_layernorm: torch.Tensor  # (hidden_size,) — pre-MLP RMSNorm
    mlp: MLPWeights


@dataclass
class ModelWeights:
    embed_tokens: torch.Tensor  # (vocab_size, hidden_size)
    layers: list = field(default_factory=list)  # list[DecoderLayerWeights]
    final_norm: torch.Tensor = None  # (hidden_size,)

    # Llama-3.2-1B and -3B use TIED embeddings: the output projection
    # (lm_head) shares the same weights as embed_tokens, transposed,
    # rather than having its own separate (vocab_size, hidden_size)
    # matrix. This is a real memory-saving choice in the actual
    # checkpoint, not a simplification made here — loading real weights
    # later will find no separate "lm_head.weight" tensor in the
    # checkpoint for these model sizes, and lm_head_weight stays None
    # to reflect that; the output projection always falls back to
    # embed_tokens.T in that case. Larger Llama variants (e.g. 8B+) do
    # NOT tie embeddings and would populate this field explicitly.
    lm_head_weight: torch.Tensor = None

    def output_projection(self) -> torch.Tensor:
        """Returns the (hidden_size, vocab_size) matrix to project final
        hidden states into logits — the untied lm_head if present,
        otherwise the transposed input embedding (tied-embedding case).
        """
        if self.lm_head_weight is not None:
            return self.lm_head_weight.T
        return self.embed_tokens.T
