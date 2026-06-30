"""A paged KV-cache using fixed-size memory blocks.

This is the follow-up to kvcache/naive_cache.py. The naive cache grows
by concatenating a new, larger tensor on every token append -- a real
O(seq_len) allocation at every decode step that wastes memory in two
ways: the allocation itself copies the old tensor on each step, and any
cache pre-sized for worst-case max_seq_len wastes the gap between actual
length and that max (internal fragmentation).

The paged cache solves both:
  1. Pre-allocates a fixed pool of equal-sized blocks (like OS virtual
     memory pages). Each block holds BLOCK_SIZE tokens' worth of K/V
     data for one layer, pre-allocated once at construction time.
  2. Assigns blocks to sequences on demand via a block table (a mapping
     from logical position to physical block index), and releases them
     when a sequence is done -- no copy, no wasted gap, no per-step
     reallocation. This is the key idea behind vLLM's PagedAttention.

The tradeoff vs. the naive cache is complexity: instead of a single
growing tensor per layer, we now have a block pool, a block table, and
a gather step (reading non-contiguous physical blocks back into a
contiguous view for attention computation). The benchmark in
benchmark/measure_paged_cache_memory.py measures whether that
complexity buys real memory savings at Llama-3.2-1B's dimensions.

Cached at the KV-HEAD dimension (num_key_value_heads), same as the naive
cache -- repeat_kv is applied after reading from the cache, never before
storing, so GQA's 4x reduction vs. full multi-head attention is
preserved here exactly as in naive_cache.py.
"""

import torch


class PagedLayerKVCache:
    """Per-layer paged K/V storage.

    A block pool of shape (max_blocks, BLOCK_SIZE, num_kv_heads, head_dim)
    is pre-allocated at construction time. A block_table (list of block
    indices, one per logical block in the current sequence) tracks which
    physical blocks hold this sequence's data. On each append, we write
    into the current tail block; when it fills, we allocate the next one.

    The result: memory is bounded by the number of blocks allocated, not
    by worst-case max_seq_len, and no tensor is ever reallocated or
    copied at decode time.
    """

    def __init__(
        self,
        max_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Physical block pool: (max_blocks, block_size, num_kv_heads, head_dim).
        # Note the (block_size, num_kv_heads, head_dim) layout -- tokens are the
        # first dimension inside a block, which makes in-place writes to a single
        # block's slot a simple index operation: pool[block_idx, slot, :, :].
        self.key_pool = torch.zeros(max_blocks, block_size, num_kv_heads, head_dim, dtype=dtype)
        self.val_pool = torch.zeros(max_blocks, block_size, num_kv_heads, head_dim, dtype=dtype)

        # Free list: all blocks start free. A deque or set would also work;
        # a list keeps indexing simple and the O(1) pop/append fast enough
        # for the sequence lengths this project benchmarks.
        self._free_blocks: list[int] = list(range(max_blocks))

        # Logical block table: list of physical block indices assigned to
        # this sequence, in order. block_table[i] holds logical tokens
        # [i*block_size .. (i+1)*block_size - 1].
        self.block_table: list[int] = []

        # How many tokens have been written into the current (tail) block.
        self._tail_offset: int = 0

        # Total tokens appended so far.
        self._seq_len: int = 0

    def _allocate_block(self) -> int:
        if not self._free_blocks:
            raise RuntimeError(
                "Paged KV-cache: block pool exhausted. "
                "Increase max_blocks or reduce max sequence length."
            )
        return self._free_blocks.pop()

    def append(self, new_key: torch.Tensor, new_value: torch.Tensor):
        """Append new_key/new_value (shape (batch, num_kv_heads, new_tokens, head_dim))
        to the cache and return the full key/value tensors (all cached tokens) for
        attention computation. batch must be 1 -- paged caches track per-sequence
        block tables, so multi-batch appends would need one block table per batch
        entry (that's the multi-sequence serving extension, not implemented here).
        """
        assert new_key.shape[0] == 1, (
            "PagedLayerKVCache: batch > 1 not supported -- each sequence needs its "
            "own block table. See the docstring."
        )
        # new_tokens: number of new tokens being appended in this call.
        # (batch, kv_heads, new_tokens, head_dim) -> new_tokens is dim 2.
        new_tokens = new_key.shape[2]

        for t in range(new_tokens):
            # If the current tail block is full (or no block allocated yet),
            # allocate a fresh one.
            if self._tail_offset == 0 or self._tail_offset == self.block_size:
                block_idx = self._allocate_block()
                self.block_table.append(block_idx)
                self._tail_offset = 0

            block_idx = self.block_table[-1]
            slot = self._tail_offset

            # Write token t's K/V into the physical slot.
            # new_key shape: (1, num_kv_heads, new_tokens, head_dim)
            # pool shape:    (max_blocks, block_size, num_kv_heads, head_dim)
            # -> transpose kv_heads and head_dim axes for the pool layout.
            self.key_pool[block_idx, slot] = new_key[0, :, t, :]  # (num_kv_heads, head_dim)
            self.val_pool[block_idx, slot] = new_value[0, :, t, :]

            self._tail_offset += 1
            self._seq_len += 1

        return self._gather()

    def _gather(self):
        """Read all allocated blocks back into a contiguous (1, num_kv_heads,
        seq_len, head_dim) tensor for attention computation. This is the extra
        step vs. the naive cache (which never needs a gather since its tensor
        is already contiguous), and is the explicit cost of paging: we trade
        a per-step reallocation for a per-step gather.

        For the sequence lengths benchmarked here (up to 4096 tokens), the
        gather is a single index_select on the pre-allocated pool, which is
        fast in practice. vLLM and similar systems also do this gather step
        (they call it "KV-cache lookup"), but offload it to custom CUDA kernels
        for production throughput -- this project uses plain PyTorch indexing,
        which is correct and measurable, even if not CUDA-optimized.
        """
        if not self.block_table:
            return None, None

        # Gather all assigned blocks from the pool in logical order.
        # key_pool: (max_blocks, block_size, num_kv_heads, head_dim)
        # gathered: (num_logical_blocks, block_size, num_kv_heads, head_dim)
        block_indices = torch.tensor(self.block_table, dtype=torch.long)
        gathered_k = self.key_pool[block_indices]  # (num_blocks, block_size, kv_heads, head_dim)
        gathered_v = self.val_pool[block_indices]

        # Flatten to (num_blocks * block_size, num_kv_heads, head_dim),
        # then slice to actual seq_len (the last block may be partially filled).
        seq_k = gathered_k.reshape(-1, self.num_kv_heads, self.head_dim)[:self._seq_len]
        seq_v = gathered_v.reshape(-1, self.num_kv_heads, self.head_dim)[:self._seq_len]

        # Restore batch dimension: (1, num_kv_heads, seq_len, head_dim).
        # This matches the (batch, kv_heads, seq_len, head_dim) layout
        # expected by the attention computation -- same as naive_cache.py's output.
        full_k = seq_k.permute(1, 0, 2).unsqueeze(0)  # (1, kv_heads, seq_len, head_dim)
        full_v = seq_v.permute(1, 0, 2).unsqueeze(0)

        return full_k, full_v

    def seq_len(self) -> int:
        return self._seq_len

    def memory_bytes(self) -> int:
        """Total bytes held by this layer's ALLOCATED blocks (K + V combined).
        Unlike the naive cache, this includes all pre-allocated pool memory --
        the point of paging is that blocks are pre-allocated and reused, so
        'memory used' means the pool's footprint, not just the filled portion.
        Unallocated blocks in the free list are excluded -- they haven't been
        assigned to this sequence yet.
        """
        allocated = len(self.block_table)
        block_bytes = (
            allocated
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * self.key_pool.element_size()
        )
        return 2 * block_bytes  # K and V


class PagedKVCache:
    """One PagedLayerKVCache per decoder layer, for a single generation
    request. Mirrors the KVCache API in naive_cache.py so that the
    benchmark and tests can compare the two caches directly.
    """

    def __init__(
        self,
        num_layers: int,
        max_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.layers = [
            PagedLayerKVCache(max_blocks, block_size, num_kv_heads, head_dim, dtype)
            for _ in range(num_layers)
        ]

    def append(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor):
        return self.layers[layer_idx].append(new_key, new_value)

    def seq_len(self) -> int:
        """Current cached sequence length. All layers are appended in lockstep,
        so layer 0's length represents the whole cache -- same convention as
        naive_cache.KVCache.seq_len().
        """
        return self.layers[0].seq_len()

    def total_memory_bytes(self) -> int:
        return sum(layer.memory_bytes() for layer in self.layers)
