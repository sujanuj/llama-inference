"""A shared physical block pool for multi-request KV-cache management.

In the paged cache (kvcache/paged_cache.py), each request gets its own
private block pool sized at construction time. That's fine for a single
request but wasteful at serving time: if request A finishes early, its
blocks sit idle even while request B is waiting for memory.

This module provides a single shared BlockPool whose blocks are allocated
to requests on demand and returned to the free list the moment a request
finishes -- exactly like an OS page allocator. The scheduler
(scheduler/scheduler.py) uses this pool to drive multiple concurrent
requests through prefill and decode, allocating blocks as sequences grow
and freeing them when generation completes.

The pool operates at the LOGICAL level -- it hands out block indices and
tracks which sequence owns which blocks. The actual tensor storage still
lives in the paged cache's key_pool/val_pool arrays; the block pool just
manages the mapping from sequence -> list of physical block indices.

Design decision: one BlockPool is shared across ALL layers. Each block
index represents one block's worth of tokens in every layer simultaneously
-- when sequence S is assigned block 7, that means block 7 in layer 0's
key_pool, block 7 in layer 1's key_pool, ..., block 7 in layer 15's
key_pool are all reserved for sequence S. This matches how vLLM manages
its block table: a single block table per sequence, one entry per logical
block, shared across layers. The alternative -- per-layer block pools --
would fragment memory without benefit since all layers always need the
same number of blocks for a given sequence.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SequenceState:
    """All scheduler-visible state for one in-flight request.

    seq_id: unique identifier assigned at request arrival.
    token_ids: the full token sequence so far (prompt + generated tokens).
        Grows by one on each decode step.
    block_table: list of physical block indices assigned to this sequence,
        in logical order. block_table[i] covers logical token positions
        [i*block_size .. (i+1)*block_size - 1].
    max_new_tokens: how many tokens to generate after the prompt.
    prompt_len: length of the original prompt (used to count generated tokens).
    status: 'waiting' -> 'running' -> 'finished' | 'evicted'.
    """
    seq_id: int
    token_ids: List[int]
    max_new_tokens: int
    prompt_len: int
    block_table: List[int] = field(default_factory=list)
    status: str = "waiting"  # waiting | running | finished | evicted

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def is_done(self) -> bool:
        return self.num_generated >= self.max_new_tokens

    @property
    def seq_len(self) -> int:
        return len(self.token_ids)


class BlockPool:
    """A fixed pool of physical block indices shared across all requests.

    Blocks are allocated to sequences by the scheduler and freed when a
    sequence finishes or is evicted. The pool has no knowledge of tensor
    storage -- it only tracks which block indices are free vs. assigned.

    Args:
        total_blocks: total number of physical blocks in the pool. Each
            block holds block_size tokens' worth of KV data for every
            layer. A pool of N blocks can hold at most N*block_size total
            cached tokens across all concurrent requests.
        block_size: tokens per block. Must match the paged cache's
            block_size exactly -- the pool and the cache share the same
            physical block indices.
    """

    def __init__(self, total_blocks: int, block_size: int):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._free: List[int] = list(range(total_blocks))
        # seq_id -> list of physical block indices owned by that sequence
        self._owned: Dict[int, List[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.total_blocks - len(self._free)

    def blocks_needed(self, current_seq_len: int, new_tokens: int = 1) -> int:
        """How many NEW blocks must be allocated to extend a sequence of
        current_seq_len by new_tokens tokens. Zero if the tail block has
        enough room; positive if one or more new blocks are needed.

        Used by the scheduler to check whether a decode step can proceed
        before committing to it -- the check-then-allocate pattern avoids
        partial allocation and simplifies rollback.
        """
        current_blocks = (current_seq_len + self.block_size - 1) // self.block_size
        new_blocks = (current_seq_len + new_tokens + self.block_size - 1) // self.block_size
        return max(0, new_blocks - current_blocks)

    def allocate(self, seq_id: int, num_blocks: int) -> Optional[List[int]]:
        """Allocate num_blocks blocks from the free list for seq_id.
        Returns the list of newly-allocated block indices, or None if
        the pool doesn't have enough free blocks. On None, no state is
        changed -- the caller can safely retry or evict another sequence.
        """
        if num_blocks == 0:
            return []
        if len(self._free) < num_blocks:
            return None
        new_blocks = [self._free.pop() for _ in range(num_blocks)]
        self._owned.setdefault(seq_id, []).extend(new_blocks)
        return new_blocks

    def free(self, seq_id: int) -> int:
        """Return all blocks owned by seq_id to the free list.
        Returns the number of blocks freed. Safe to call on a seq_id
        that owns no blocks (returns 0).
        """
        blocks = self._owned.pop(seq_id, [])
        self._free.extend(blocks)
        return len(blocks)

    def owned_blocks(self, seq_id: int) -> List[int]:
        """Block indices currently assigned to seq_id, in allocation order."""
        return list(self._owned.get(seq_id, []))
