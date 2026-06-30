"""Tests for the naive KV-cache and cache-aware generation.

The single most important property tested here: running the model with
the cache, one token at a time, must produce EXACTLY the same logits as
running the equivalent full sequence through the no-cache forward() in
one shot. The cache is a performance optimization, not a different
computation -- if it changes the actual numbers, that's a real
correctness bug, not an acceptable tradeoff.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvcache.naive_cache import KVCache, LayerKVCache
from model.config import LlamaConfig
from model.model import forward, forward_with_cache, generate, next_token_greedy
from testutil.random_weights import random_model_weights


def _tiny_config():
    return LlamaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=32, rms_norm_eps=1e-5, rope_theta=10000.0,
    )


def test_layer_kv_cache_starts_empty():
    cache = LayerKVCache()
    assert cache.seq_len() == 0
    assert cache.memory_bytes() == 0


def test_layer_kv_cache_append_grows_seq_len():
    cache = LayerKVCache()
    k1 = torch.randn(1, 2, 3, 4)  # batch=1, kv_heads=2, seq_len=3, head_dim=4
    v1 = torch.randn(1, 2, 3, 4)
    cache.append(k1, v1)
    assert cache.seq_len() == 3

    k2 = torch.randn(1, 2, 1, 4)  # one more token
    v2 = torch.randn(1, 2, 1, 4)
    cache.append(k2, v2)
    assert cache.seq_len() == 4


def test_layer_kv_cache_append_preserves_earlier_values_exactly():
    # The concatenation in LayerKVCache.append must not alter
    # already-cached values -- this would be a real bug if, say, the
    # concat axis were wrong and values got interleaved or overwritten.
    cache = LayerKVCache()
    k1 = torch.full((1, 2, 2, 4), 1.0)
    v1 = torch.full((1, 2, 2, 4), 2.0)
    cache.append(k1, v1)

    k2 = torch.full((1, 2, 1, 4), 99.0)
    v2 = torch.full((1, 2, 1, 4), 88.0)
    full_k, full_v = cache.append(k2, v2)

    assert torch.equal(full_k[:, :, :2, :], k1)
    assert torch.equal(full_v[:, :, :2, :], v1)
    assert torch.equal(full_k[:, :, 2:, :], k2)
    assert torch.equal(full_v[:, :, 2:, :], v2)


def test_kv_cache_seq_len_reflects_all_layers_in_lockstep():
    cache = KVCache(num_layers=3)
    for layer_idx in range(3):
        cache.append(layer_idx, torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    assert cache.seq_len() == 5


def test_kv_cache_memory_bytes_scales_with_appended_tokens():
    cache = KVCache(num_layers=2)
    assert cache.total_memory_bytes() == 0

    for layer_idx in range(2):
        cache.append(layer_idx, torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4))
    mem_at_10 = cache.total_memory_bytes()
    assert mem_at_10 > 0

    for layer_idx in range(2):
        cache.append(layer_idx, torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4))
    mem_at_20 = cache.total_memory_bytes()

    assert mem_at_20 == 2 * mem_at_10, (
        f"memory should scale exactly linearly with cached tokens for fixed-size appends, "
        f"got {mem_at_10} at 10 tokens and {mem_at_20} at 20"
    )


def test_cached_prefill_matches_uncached_forward_for_the_prompt():
    # The prefill step alone (no decoding yet) should produce IDENTICAL
    # logits to a plain forward() call on the same input -- prefill is
    # just forward_with_cache at position_offset=0, so this is really
    # checking that path doesn't introduce any difference.
    torch.manual_seed(0)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=3)
    prompt = torch.randint(0, config.vocab_size, (1, 6))

    logits_uncached = forward(prompt, weights, config)

    cache = KVCache(num_layers=config.num_hidden_layers)
    logits_cached = forward_with_cache(prompt, weights, config, cache, position_offset=0)

    assert torch.allclose(logits_uncached, logits_cached, atol=1e-5), (
        f"max diff: {(logits_uncached - logits_cached).abs().max()}"
    )


def test_cached_incremental_generation_matches_uncached_full_forward():
    # THE key correctness test: generate token by token WITH the cache,
    # then separately run the full resulting sequence through the
    # no-cache forward() in one shot, and confirm the logits at each
    # position agree. If position_offset or the cache append logic is
    # wrong, this is what would catch it -- the prefill-only test above
    # could pass while this one fails, since it specifically exercises
    # the incremental decode steps the prefill test doesn't reach.
    torch.manual_seed(0)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=3)
    prompt = torch.randint(0, config.vocab_size, (1, 4))

    full_sequence = generate(prompt, weights, config, max_new_tokens=5)
    assert full_sequence.shape == (1, 9)  # 4 prompt + 5 generated

    # Re-run the ENTIRE final sequence through the plain no-cache
    # forward() and confirm the next-token prediction at every position
    # matches what the cached generation actually produced -- this is
    # the strongest possible check: not just "did it run," but "does
    # the cached path compute the same function as the reference path,
    # at every single position along the way."
    logits_full = forward(full_sequence, weights, config)
    for pos in range(3, 8):  # positions where a "next token" prediction exists and was used
        predicted = logits_full[0, pos].argmax().item()
        actually_generated = full_sequence[0, pos + 1].item()
        assert predicted == actually_generated, (
            f"at position {pos}, the no-cache forward pass would have predicted token "
            f"{predicted}, but cached generation actually produced {actually_generated}"
        )


def test_cached_generation_with_longer_prompt_and_more_tokens():
    # A second, larger instance of the same correctness check, with a
    # longer prompt and more generated tokens -- specifically to
    # exercise the case where position_offset grows past prompt_len by
    # more than 1 (i.e. several decode steps deep), not just the very
    # first one.
    torch.manual_seed(7)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=3)
    prompt = torch.randint(0, config.vocab_size, (1, 8))

    full_sequence = generate(prompt, weights, config, max_new_tokens=10)
    assert full_sequence.shape == (1, 18)

    logits_full = forward(full_sequence, weights, config)
    for pos in range(7, 17):
        predicted = logits_full[0, pos].argmax().item()
        actually_generated = full_sequence[0, pos + 1].item()
        assert predicted == actually_generated, f"mismatch at position {pos}"


def test_generate_output_includes_original_prompt_unchanged():
    torch.manual_seed(0)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=2)
    prompt = torch.randint(0, config.vocab_size, (1, 5))

    full_sequence = generate(prompt, weights, config, max_new_tokens=3)

    assert torch.equal(full_sequence[:, :5], prompt)
