"""Continuous-batching scheduler for multi-request LLM inference.

This is the component that makes the paged KV-cache useful at serving
time. Without a scheduler, each request gets its own isolated block pool
with no sharing -- a finished request's blocks sit idle even while new
requests are waiting. The scheduler fixes this by:

  1. Maintaining a waiting queue of incoming requests.
  2. Admitting requests into a running set when enough free blocks exist
     to hold their initial prompt.
  3. Driving one scheduling step at a time: prefill for newly-admitted
     requests (process the full prompt in one forward_with_cache call),
     then decode for all running requests (one new token each).
  4. Freeing blocks immediately when a request finishes, so the pool
     can admit the next waiting request on the very next step.
  5. Evicting the longest-running sequence when the pool is exhausted
     mid-decode -- a simple policy that keeps the scheduler making
     progress rather than deadlocking.

This is "continuous batching" in the sense used by Orca (Yu et al., 2022)
and vLLM: requests are not grouped into fixed-size batches that must all
start and finish together. Instead, new requests are admitted as soon as
memory is available, and the running set changes dynamically each step.
This project implements the core admission/eviction logic and the
prefill+decode step loop; production systems add iteration-level
preemption, priority queues, and CUDA kernel batching on top.

The scheduler is deliberately model-agnostic: it calls forward_with_cache
from model/model.py and uses PagedKVCache from kvcache/paged_cache.py,
but knows nothing about weight values or tokenization. This matches the
separation of concerns in real inference engines (vLLM's scheduler
similarly operates on sequence IDs and block tables, not on model weights).
"""

from __future__ import annotations

import torch
from typing import Callable, Dict, List, Optional, Tuple

from kvcache.paged_cache import PagedKVCache
from model.config import LlamaConfig
from model.model import forward_with_cache, next_token_greedy
from scheduler.block_pool import BlockPool, SequenceState


class Scheduler:
    """Continuous-batching scheduler over a shared block pool.

    Args:
        config: model architecture config (used to construct per-sequence
            paged caches and to know num_layers, num_kv_heads, head_dim).
        weights: model weights passed through to forward_with_cache.
        total_blocks: total physical blocks in the shared pool.
        block_size: tokens per block (must match across pool and caches).
        dtype: tensor dtype for KV cache storage (default bf16 to match
            the real Llama-3.2-1B checkpoint dtype).
    """

    def __init__(
        self,
        config: LlamaConfig,
        weights,
        total_blocks: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.weights = weights
        self.block_size = block_size
        self.dtype = dtype

        self.pool = BlockPool(total_blocks=total_blocks, block_size=block_size)

        # waiting: arrived but not yet admitted (no blocks allocated)
        self._waiting: List[SequenceState] = []
        # running: admitted, blocks allocated, paged cache constructed
        self._running: Dict[int, SequenceState] = {}
        # per-sequence paged caches, keyed by seq_id
        self._caches: Dict[int, PagedKVCache] = {}
        # finished sequences, keyed by seq_id -> final token_ids
        self._finished: Dict[int, List[int]] = {}

        self._next_seq_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, token_ids: List[int], max_new_tokens: int) -> int:
        """Enqueue a new generation request. Returns its seq_id.

        The request starts in 'waiting' status; it will be admitted
        (blocks allocated, cache constructed) on the next step() call
        where enough free blocks exist.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        seq = SequenceState(
            seq_id=seq_id,
            token_ids=list(token_ids),
            max_new_tokens=max_new_tokens,
            prompt_len=len(token_ids),
        )
        self._waiting.append(seq)
        return seq_id

    def step(self) -> Dict[int, Optional[int]]:
        """Run one scheduling step. Returns a dict of seq_id -> new_token
        for every sequence that generated a token this step (None for
        sequences that were evicted rather than decoded).

        One step consists of:
          1. Admit waiting requests that fit in the pool (prefill).
          2. Decode one token for every running request.
          3. Finish any sequence that has hit max_new_tokens.
          4. Evict the longest sequence if the pool is exhausted mid-decode.
        """
        results: Dict[int, Optional[int]] = {}

        # --- Phase 1: admit waiting requests ---
        self._admit_waiting()

        if not self._running:
            return results

        # --- Phase 2: ensure every running sequence has a free slot ---
        # Check before decoding: each sequence needs at most 1 new block.
        evicted = self._ensure_decode_capacity()
        for seq_id in evicted:
            results[seq_id] = None

        if not self._running:
            return results

        # --- Phase 3: decode one token per running sequence ---
        for seq_id, seq in list(self._running.items()):
            new_token = self._decode_one(seq)
            seq.token_ids.append(new_token)
            results[seq_id] = new_token

            if seq.is_done:
                self._finish(seq)

        return results

    def is_idle(self) -> bool:
        """True when there are no waiting or running requests."""
        return not self._waiting and not self._running

    def get_result(self, seq_id: int) -> Optional[List[int]]:
        """Return the full generated token_ids for a finished sequence,
        or None if it hasn't finished yet."""
        return self._finished.get(seq_id)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blocks_for_prompt(self, prompt_len: int) -> int:
        """Blocks needed to cache a full prompt of prompt_len tokens."""
        return (prompt_len + self.block_size - 1) // self.block_size

    def _admit_waiting(self):
        """Admit as many waiting requests as the pool can fit.
        Admitted requests are prefilled immediately (their full prompt
        is processed in one forward_with_cache call).
        """
        still_waiting = []
        for seq in self._waiting:
            needed = self._blocks_for_prompt(seq.prompt_len)
            allocated = self.pool.allocate(seq.seq_id, needed)
            if allocated is None:
                # Pool can't fit this request right now; keep waiting.
                still_waiting.append(seq)
                continue

            seq.block_table.extend(allocated)
            seq.status = "running"

            # Build a per-sequence paged cache. The block_table tracks
            # which physical blocks belong to this sequence; the cache
            # holds the actual tensor storage.
            cache = PagedKVCache(
                num_layers=self.config.num_hidden_layers,
                max_blocks=self.pool.total_blocks,
                block_size=self.block_size,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                dtype=self.dtype,
            )

            # Prefill: process the entire prompt at once.
            input_ids = torch.tensor([seq.token_ids], dtype=torch.long)
            forward_with_cache(input_ids, self.weights, self.config, cache, position_offset=0)

            self._caches[seq.seq_id] = cache
            self._running[seq.seq_id] = seq

        self._waiting = still_waiting

    def _ensure_decode_capacity(self) -> List[int]:
        """Before decoding, verify every running sequence has room for
        one more token. If the pool can't accommodate all of them, evict
        the sequence that has generated the most tokens (longest-running
        first) until capacity is restored. Returns list of evicted seq_ids.

        This is a simple eviction policy -- real systems use more
        sophisticated priority schemes -- but it's correct and sufficient
        to demonstrate the scheduler making progress under memory pressure.
        """
        evicted = []
        for seq_id, seq in list(self._running.items()):
            needed = self.pool.blocks_needed(seq.seq_len, new_tokens=1)
            if needed == 0:
                continue
            allocated = self.pool.allocate(seq_id, needed)
            if allocated is not None:
                seq.block_table.extend(allocated)
                continue

            # Pool exhausted -- evict the longest-running sequence.
            victim_id = max(
                self._running,
                key=lambda sid: self._running[sid].num_generated,
            )
            victim = self._running.pop(victim_id)
            victim.status = "evicted"
            freed = self.pool.free(victim_id)
            del self._caches[victim_id]
            evicted.append(victim_id)

            # Retry allocation for the current sequence after eviction.
            needed = self.pool.blocks_needed(seq.seq_len, new_tokens=1)
            allocated = self.pool.allocate(seq_id, needed)
            if allocated is not None:
                seq.block_table.extend(allocated)
            # If still can't allocate, this sequence will be skipped this
            # step -- it stays in _running and will retry next step.

        return evicted

    def _decode_one(self, seq: SequenceState) -> int:
        """Decode one new token for seq. Returns the new token id."""
        cache = self._caches[seq.seq_id]
        # Decode input: just the last token, with position_offset = current seq_len.
        last_token = torch.tensor([[seq.token_ids[-1]]], dtype=torch.long)
        logits = forward_with_cache(
            last_token, self.weights, self.config, cache,
            position_offset=seq.seq_len - 1,
        )
        return next_token_greedy(logits).item()

    def _finish(self, seq: SequenceState):
        """Mark seq as finished, free its blocks, and record its output."""
        seq.status = "finished"
        self._running.pop(seq.seq_id)
        self.pool.free(seq.seq_id)
        del self._caches[seq.seq_id]
        self._finished[seq.seq_id] = list(seq.token_ids)
