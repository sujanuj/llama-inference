"""The full Llama-3.2-1B forward pass: token embedding -> 16 decoder
layers -> final RMSNorm -> output projection to vocabulary logits.

This is the first point in the project where a complete, runnable
forward pass exists — everything before this phase tested individual
pieces (RMSNorm, RoPE, attention, MLP) in isolation; this assembles them
into something that actually takes token IDs in and produces next-token
logits out.
"""

import torch

from model.config import LlamaConfig
from model.decoder import decoder_layer
from model.rmsnorm import rms_norm
from model.rope import compute_rope_frequencies
from model.weights import ModelWeights


def forward(
    input_ids: torch.Tensor,
    weights: ModelWeights,
    config: LlamaConfig,
    causal: bool = True,
) -> torch.Tensor:
    """Args:
        input_ids: (batch, seq_len) — integer token IDs.
        weights: full model weights.
        config: architecture configuration (must match the weights'
            actual dimensions, or this will fail with a shape error
            rather than silently produce wrong output).

    Returns:
        logits: (batch, seq_len, vocab_size)
    """
    batch, seq_len = input_ids.shape

    hidden_states = weights.embed_tokens[input_ids]  # (batch, seq_len, hidden_size)

    cos_table, sin_table = compute_rope_frequencies(
        config.head_dim, max_seq_len=seq_len, theta=config.rope_theta
    )
    # Sliced to exactly seq_len here since this is a fresh forward pass
    # with no cached context — every position's absolute index equals
    # its index within input_ids. Once the KV-cache phase introduces
    # incremental decoding (new tokens appended after already-cached
    # ones), the caller will need to slice this table starting at the
    # correct ABSOLUTE position offset, not always from 0 — that's a
    # real thing to get right later, not yet a concern here.
    cos = cos_table[:seq_len]
    sin = sin_table[:seq_len]

    for layer_weights in weights.layers:
        hidden_states = decoder_layer(
            hidden_states,
            layer_weights,
            cos,
            sin,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            causal=causal,
        )

    hidden_states = rms_norm(hidden_states, weights.final_norm, config.rms_norm_eps)

    logits = hidden_states @ weights.output_projection()  # (batch, seq_len, vocab_size)
    return logits


def next_token_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy decoding: pick the highest-logit token at the LAST
    position of each sequence in the batch.

    Args:
        logits: (batch, seq_len, vocab_size)

    Returns:
        (batch,) — next token ID for each sequence.
    """
    last_position_logits = logits[:, -1, :]  # (batch, vocab_size)
    return last_position_logits.argmax(dim=-1)
