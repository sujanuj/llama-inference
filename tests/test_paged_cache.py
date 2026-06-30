"""Tests for the paged KV-cache.

Three things verified here:
  1. Structural correctness: block allocation, seq_len tracking, gather
     output shape -- the paged cache's own internal mechanics.
  2. Output equivalence with the naive cache: every value the paged
     cache returns from append() must be numerically identical to what
     the naive cache would return for the same sequence of inputs. The
     paged cache is a performance optimization, not a different
     computation.
  3. Model-level equivalence: generating tokens with the paged cache
     must produce the same sequence as generating with the naive cache.
     If the gather step or block-table logic is wrong, this is what
     catches it -- the structural tests above could pass while the
     model's actual predictions diverge.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvcache.naive_cache import KVCache, LayerKVCache
from kvcache.paged_cache import PagedKVCache, PagedLayerKVCache
from model.config import LlamaConfig
from model.model import forward_with_cache, generate
from testutil.random_weights import random_model_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config():
    return LlamaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=32, rms_norm_eps=1e-5, rope_theta=10000.0,
    )


def _make_paged(config, max_blocks=64, block_size=4):
    return PagedKVCache(
        num_layers=config.num_hidden_layers,
        max_blocks=max_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )


def _make_naive(config):
    return KVCache(num_layers=config.num_hidden_layers)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

def test_paged_layer_starts_empty():
    layer = PagedLayerKVCache(max_blocks=8, block_size=4, num_kv_heads=2, head_dim=4)
    assert layer.seq_len() == 0
    assert layer.memory_bytes() == 0


def test_paged_layer_seq_len_increments_correctly():
    layer = PagedLayerKVCache(max_blocks=8, block_size=4, num_kv_heads=2, head_dim=4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    layer.append(k, v)
    assert layer.seq_len() == 3

    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    layer.append(k2, v2)
    assert layer.seq_len() == 4


def test_paged_layer_gather_output_shape():
    layer = PagedLayerKVCache(max_blocks=8, block_size=4, num_kv_heads=2, head_dim=4)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    full_k, full_v = layer.append(k, v)
    # Should be (batch=1, kv_heads=2, seq_len=5, head_dim=4)
    assert full_k.shape == (1, 2, 5, 4), f"got {full_k.shape}"
    assert full_v.shape == (1, 2, 5, 4), f"got {full_v.shape}"


def test_paged_layer_allocates_new_block_when_tail_fills():
    block_size = 4
    layer = PagedLayerKVCache(max_blocks=8, block_size=block_size, num_kv_heads=2, head_dim=4)
    # Fill exactly one block.
    k = torch.randn(1, 2, block_size, 4)
    v = torch.randn(1, 2, block_size, 4)
    layer.append(k, v)
    assert len(layer.block_table) == 1

    # One more token: must spill into a second block.
    layer.append(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
    assert len(layer.block_table) == 2


def test_paged_cache_seq_len_matches_tokens_appended():
    config = _tiny_config()
    cache = _make_paged(config)
    for layer_idx in range(config.num_hidden_layers):
        cache.append(layer_idx, torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4))
    assert cache.seq_len() == 7


def test_paged_cache_memory_bytes_nonzero_after_append():
    config = _tiny_config()
    cache = _make_paged(config)
    assert cache.total_memory_bytes() == 0
    for layer_idx in range(config.num_hidden_layers):
        cache.append(layer_idx, torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    assert cache.total_memory_bytes() > 0


# ---------------------------------------------------------------------------
# Output equivalence with naive cache
# ---------------------------------------------------------------------------

def test_paged_layer_gather_matches_naive_single_append():
    # Same single-token append through both caches; the full K/V tensors
    # returned must be identical -- paging must not change any values.
    torch.manual_seed(42)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)

    naive = LayerKVCache()
    full_k_naive, full_v_naive = naive.append(k.clone(), v.clone())

    paged = PagedLayerKVCache(max_blocks=16, block_size=4, num_kv_heads=2, head_dim=4)
    full_k_paged, full_v_paged = paged.append(k.clone(), v.clone())

    assert torch.allclose(full_k_naive, full_k_paged, atol=1e-6), (
        f"K mismatch: max diff {(full_k_naive - full_k_paged).abs().max()}"
    )
    assert torch.allclose(full_v_naive, full_v_paged, atol=1e-6), (
        f"V mismatch: max diff {(full_v_naive - full_v_paged).abs().max()}"
    )


def test_paged_layer_gather_matches_naive_multi_append():
    # Multiple incremental appends (simulating decode steps). After each
    # step, the paged cache's gathered output must match the naive cache's
    # growing tensor exactly. This specifically exercises the cross-block-
    # boundary case: with block_size=4, appending 5 tokens spans two blocks.
    torch.manual_seed(7)
    naive = LayerKVCache()
    paged = PagedLayerKVCache(max_blocks=16, block_size=4, num_kv_heads=2, head_dim=4)

    for step_tokens in [3, 1, 1, 2]:  # 3 -> fills partially, 1 -> fills block, 1 -> new block, 2
        k = torch.randn(1, 2, step_tokens, 4)
        v = torch.randn(1, 2, step_tokens, 4)
        nk, nv = naive.append(k.clone(), v.clone())
        pk, pv = paged.append(k.clone(), v.clone())

        assert torch.allclose(nk, pk, atol=1e-6), (
            f"K mismatch after appending {step_tokens} tokens: "
            f"max diff {(nk - pk).abs().max()}"
        )
        assert torch.allclose(nv, pv, atol=1e-6), (
            f"V mismatch after appending {step_tokens} tokens: "
            f"max diff {(nv - pv).abs().max()}"
        )


# ---------------------------------------------------------------------------
# Model-level equivalence
# ---------------------------------------------------------------------------

def test_paged_cache_generation_matches_naive_cache_generation():
    # THE key test: generate the same prompt through both caches and
    # confirm the output sequences are token-for-token identical. If
    # the block-table logic or the gather step is wrong in a way that
    # only surfaces through full attention computation (not just raw
    # tensor equality), this is what catches it.
    torch.manual_seed(0)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)
    prompt = torch.randint(0, config.vocab_size, (1, 6))

    seq_naive = generate(prompt, weights, config, max_new_tokens=8, cache_factory=_make_naive)
    seq_paged = generate(prompt, weights, config, max_new_tokens=8, cache_factory=_make_paged)

    assert torch.equal(seq_naive, seq_paged), (
        f"Paged and naive generation produced different tokens.\n"
        f"Naive:  {seq_naive.tolist()}\n"
        f"Paged:  {seq_paged.tolist()}"
    )


def test_paged_cache_generation_matches_naive_cache_generation_longer():
    # Second instance with a longer prompt and more decode steps --
    # specifically to exercise >1 full block being allocated and gathered
    # during actual model computation, not just raw tensor ops.
    torch.manual_seed(13)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    seq_naive = generate(prompt, weights, config, max_new_tokens=12, cache_factory=_make_naive)
    seq_paged = generate(prompt, weights, config, max_new_tokens=12, cache_factory=_make_paged)

    assert torch.equal(seq_naive, seq_paged), (
        f"Paged and naive generation diverged on longer sequence.\n"
        f"Naive:  {seq_naive.tolist()}\n"
        f"Paged:  {seq_paged.tolist()}"
    )
