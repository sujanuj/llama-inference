"""A naive, unbounded-growth KV-cache.

This is deliberately the SIMPLE version, built first so its memory cost
can be measured directly before building a paged cache (fixed-size
blocks, like OS virtual memory pages) to fix that cost. The point isn't
that this version is wrong -- it's correct and is exactly what a
first-pass inference engine would build -- it's that it allocates one
growing tensor per layer per request, with no upper bound and no reuse
between requests, which is precisely the problem real serving engines
(vLLM's PagedAttention, etc.) exist to solve. Building this first and
measuring it gives a real "before" number for that later comparison,
the same way lsmdb's naive memtable.SizeBytes() preceded a more careful
accounting, and the same way the unpaged range scan preceded the
indexed one.

Cached at the KV-HEAD dimension, not the query-head dimension -- this is
the actual point of grouped-query attention (Phase 2): Llama-3.2-1B has
32 query heads but only 8 KV heads, so caching only the 8 KV heads' data
is already a 4x memory reduction versus caching per query-head, before
any further optimization. repeat_kv (from model/attention.py) is applied
AFTER reading from the cache, at attention-compute time -- never before
storing, which would throw that 4x reduction away.
"""

import torch


class LayerKVCache:
    """Per-layer K/V storage. key/value are None until the first
    append; after that they're (batch, num_kv_heads, seq_len, head_dim)
    tensors that grow along the seq_len dimension on each append.
    """

    def __init__(self):
        self.key = None
        self.value = None

    def append(self, new_key: torch.Tensor, new_value: torch.Tensor):
        """Appends new_key/new_value (shape (batch, num_kv_heads,
        new_seq_len, head_dim)) to whatever's already cached, and
        returns the FULL key/value tensors after appending -- this is
        what attention actually needs to compute against (all cached
        history plus the new tokens), not just the newly-appended slice.
        """
        if self.key is None:
            self.key = new_key
            self.value = new_value
        else:
            # Concatenating along the sequence dimension (dim=2) is
            # exactly what makes this cache "naive": every append
            # allocates a brand-new, larger tensor and copies the old
            # contents into it, rather than writing into pre-allocated
            # space. This is the real cost a paged cache is built to
            # eliminate -- worth measuring directly (see
            # benchmark/measure_naive_cache_memory.py) rather than just
            # asserting it's expensive.
            self.key = torch.cat([self.key, new_key], dim=2)
            self.value = torch.cat([self.value, new_value], dim=2)
        return self.key, self.value

    def seq_len(self) -> int:
        if self.key is None:
            return 0
        return self.key.shape[2]

    def memory_bytes(self) -> int:
        """Total bytes currently held by this layer's cached K and V
        tensors combined. Used directly by the memory-measurement
        benchmark -- this is the real number a paged cache's savings
        get compared against.
        """
        if self.key is None:
            return 0
        return self.key.element_size() * self.key.nelement() + self.value.element_size() * self.value.nelement()


class KVCache:
    """One LayerKVCache per decoder layer, for a single generation
    request. A real multi-request serving engine would need one of
    these per concurrent request -- exactly the scenario where this
    cache's per-request, unbounded-growth, no-sharing design becomes a
    real memory problem at scale, motivating the scheduler phase that
    comes after the paged cache.
    """

    def __init__(self, num_layers: int):
        self.layers = [LayerKVCache() for _ in range(num_layers)]

    def append(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor):
        return self.layers[layer_idx].append(new_key, new_value)

    def seq_len(self) -> int:
        """Current cached sequence length. All layers are appended to
        in lockstep during generation (one token's worth of K/V per
        layer per step), so any layer's length represents the whole
        cache's length -- layer 0 is used here as the representative,
        with no special meaning to that choice.
        """
        return self.layers[0].seq_len()

    def total_memory_bytes(self) -> int:
        return sum(layer.memory_bytes() for layer in self.layers)
