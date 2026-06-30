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


def forward_with_cache(
    input_ids: torch.Tensor,
    weights: ModelWeights,
    config: LlamaConfig,
    cache,
    position_offset: int,
) -> torch.Tensor:
    """Cache-aware forward pass for incremental generation: input_ids
    contains only the NEW tokens for this step (the whole prompt on the
    first "prefill" call, then exactly one new token per step
    afterward), and position_offset is the absolute position of the
    first new token -- i.e. how many tokens are already cached.

    Getting position_offset right matters for the same reason flagged in
    forward()'s docstring: RoPE needs each token's TRUE absolute
    position, not its position within just this call's input_ids. A
    generation loop is responsible for tracking and passing this
    correctly across calls -- see generate() below for the reference
    implementation of that bookkeeping.

    Args:
        input_ids: (batch, new_seq_len) -- new tokens only.
        cache: a kvcache.naive_cache.KVCache, already containing
            position_offset tokens' worth of cached K/V per layer.
        position_offset: absolute position of input_ids[:, 0].

    Returns:
        logits: (batch, new_seq_len, vocab_size)
    """
    from model.decoder import decoder_layer_with_cache

    batch, new_seq_len = input_ids.shape

    hidden_states = weights.embed_tokens[input_ids]

    cos_table, sin_table = compute_rope_frequencies(
        config.head_dim,
        max_seq_len=position_offset + new_seq_len,
        theta=config.rope_theta,
    )
    # Sliced starting at position_offset, NOT 0 -- this is the one-line
    # difference from forward() that makes incremental decoding
    # correct. A new token generated when 50 tokens are already cached
    # is at absolute position 50, and must be rotated accordingly, not
    # treated as if it were the first token in the sequence.
    cos = cos_table[position_offset : position_offset + new_seq_len]
    sin = sin_table[position_offset : position_offset + new_seq_len]

    for layer_idx, layer_weights in enumerate(weights.layers):
        hidden_states = decoder_layer_with_cache(
            hidden_states,
            layer_weights,
            cos,
            sin,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            cache=cache,
            layer_idx=layer_idx,
        )

    hidden_states = rms_norm(hidden_states, weights.final_norm, config.rms_norm_eps)
    logits = hidden_states @ weights.output_projection()
    return logits


def generate(
    input_ids: torch.Tensor,
    weights: ModelWeights,
    config: LlamaConfig,
    max_new_tokens: int,
    cache_factory=None,
):
    """Generates max_new_tokens new tokens autoregressively, using a
    KVCache so each step only computes the new token's Q/K/V rather
    than recomputing attention over the whole sequence so far.

    Two distinct phases, both using forward_with_cache:
      - "Prefill": the first call processes the ENTIRE input prompt at
        once (new_seq_len = len(prompt)), populating the cache with
        every prompt token's K/V in one pass.
      - "Decode": every call after that processes exactly ONE new token
        (the just-generated one), with position_offset advancing by 1
        each step.

    Args:
        input_ids: (batch, prompt_len) -- the prompt to continue from.
        max_new_tokens: how many tokens to generate after the prompt.

    Returns:
        (batch, prompt_len + max_new_tokens) -- the full sequence,
        prompt followed by generated tokens.
    """
    if cache_factory is None:
        from kvcache.naive_cache import KVCache
        cache = KVCache(num_layers=config.num_hidden_layers)
    else:
        cache = cache_factory(config)
    all_ids = input_ids

    # Prefill: process the whole prompt in one forward_with_cache call,
    # at position_offset=0 since nothing is cached yet.
    logits = forward_with_cache(input_ids, weights, config, cache, position_offset=0)
    next_token = next_token_greedy(logits).unsqueeze(1)  # (batch, 1)
    all_ids = torch.cat([all_ids, next_token], dim=1)

    # Decode: one new token per step, each time passing ONLY that one
    # token as input_ids (not the whole sequence so far -- the cache
    # already holds everything needed from before), with position_offset
    # equal to however many tokens are cached at the start of this step.
    for _ in range(max_new_tokens - 1):
        position_offset = cache.seq_len()
        logits = forward_with_cache(next_token, weights, config, cache, position_offset=position_offset)
        next_token = next_token_greedy(logits).unsqueeze(1)
        all_ids = torch.cat([all_ids, next_token], dim=1)

    return all_ids
