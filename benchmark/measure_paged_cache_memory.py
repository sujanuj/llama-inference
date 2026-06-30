"""Measures the paged KV-cache's memory footprint at real Llama-3.2-1B
dimensions and compares it directly against the naive cache's numbers.

The naive cache's baseline was measured in benchmark/measure_naive_cache_memory.py.
This script produces the "after" numbers using the same sequence lengths
and batch sizes so the comparison is apples-to-apples. The key difference
is fragmentation: the naive cache allocates one tensor per request sized
for exactly the tokens received, while a real serving engine would size
its naive cache for worst-case max_seq_len to avoid repeated reallocation.
The paged cache's block-granularity allocation bounds that fragmentation
to at most (block_size - 1) tokens per sequence -- one partial tail block.

Run with: python benchmark/measure_paged_cache_memory.py
No real weights or network access needed -- memory depends only on tensor
shapes and dtype, not on trained weight values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kvcache.naive_cache import KVCache as NaiveKVCache
from kvcache.paged_cache import PagedKVCache
from model.config import LLAMA_3_2_1B


BLOCK_SIZE = 16  # tokens per block. 16 is a common choice in real systems
                  # (vLLM defaults to 16); small enough to limit tail-block
                  # fragmentation to <=15 tokens, large enough that the
                  # block-table metadata overhead is negligible.


def measure_naive(seq_len: int, batch_size: int, dtype=torch.bfloat16) -> int:
    config = LLAMA_3_2_1B
    cache = NaiveKVCache(num_layers=config.num_hidden_layers)
    for layer_idx in range(config.num_hidden_layers):
        k = torch.zeros(batch_size, config.num_key_value_heads, seq_len, config.head_dim, dtype=dtype)
        v = torch.zeros(batch_size, config.num_key_value_heads, seq_len, config.head_dim, dtype=dtype)
        cache.append(layer_idx, k, v)
    return cache.total_memory_bytes()


def measure_paged(seq_len: int, batch_size: int, dtype=torch.bfloat16) -> int:
    """Measures paged cache memory for batch_size independent sequences,
    each of length seq_len. Since PagedKVCache is per-sequence, we
    instantiate one cache per sequence and sum their footprints -- this
    mirrors the naive cache's per-request model exactly.
    """
    config = LLAMA_3_2_1B
    # Each sequence needs enough blocks to hold seq_len tokens.
    # ceil(seq_len / BLOCK_SIZE) blocks per layer.
    blocks_per_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Add a small headroom so the pool isn't exactly full at max seq_len.
    max_blocks = blocks_per_seq + 4

    total = 0
    for _ in range(batch_size):
        cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            max_blocks=max_blocks,
            block_size=BLOCK_SIZE,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
        )
        for layer_idx in range(config.num_hidden_layers):
            k = torch.zeros(1, config.num_key_value_heads, seq_len, config.head_dim, dtype=dtype)
            v = torch.zeros(1, config.num_key_value_heads, seq_len, config.head_dim, dtype=dtype)
            cache.append(layer_idx, k, v)
        total += cache.total_memory_bytes()
    return total


def human_readable(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def main():
    config = LLAMA_3_2_1B
    print("KV-cache memory: naive vs paged at Llama-3.2-1B dimensions")
    print(f"({config.num_hidden_layers} layers, {config.num_key_value_heads} KV heads, "
          f"{config.head_dim} head_dim, bf16, block_size={BLOCK_SIZE})\n")

    for batch_size in [1, 4, 16]:
        print(f"batch_size={batch_size}")
        print(f"  {'seq_len':>8}  {'naive':>12}  {'paged':>12}  {'overhead':>10}")
        for seq_len in [128, 512, 1024, 2048, 4096]:
            naive_bytes = measure_naive(seq_len, batch_size)
            paged_bytes = measure_paged(seq_len, batch_size)
            # overhead: extra bytes from block-granularity rounding.
            # With block_size=16, at most 15 wasted slots per sequence per layer.
            overhead_pct = (paged_bytes - naive_bytes) / naive_bytes * 100
            print(
                f"  {seq_len:>8}  {human_readable(naive_bytes):>12}  "
                f"{human_readable(paged_bytes):>12}  {overhead_pct:>+.1f}%"
            )
        print()

    print("Notes:")
    print(f"  Block size: {BLOCK_SIZE} tokens/block. Paged overhead bounded by")
    print(f"  at most {BLOCK_SIZE - 1} wasted slots in the tail block per sequence per layer.")
    print()
    print("  The comparison above assumes BOTH caches fill to exactly seq_len tokens.")
    print("  In real serving, the naive cache is typically pre-allocated to max_seq_len")
    print("  to avoid reallocation mid-request. For a max_seq_len=4096 serving a request")
    print("  that actually uses 512 tokens, the naive cache wastes 4096-512=3584 token-slots;")
    print("  the paged cache allocates only the blocks needed for the actual 512 tokens,")
    print("  capping waste at 15 token-slots. That gap -- not visible in the table above --")
    print("  is the paged cache's main practical advantage in multi-request serving.")


if __name__ == "__main__":
    main()
