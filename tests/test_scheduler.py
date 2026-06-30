"""Tests for the continuous-batching scheduler and shared block pool.

Four things verified:
  1. BlockPool mechanics: allocation, freeing, capacity accounting.
  2. Scheduler correctness: single request produces same output as
     direct generate() -- the scheduler must not change the computation.
  3. Multi-request sharing: two requests run concurrently, both finish,
     pool blocks are reused between them.
  4. Eviction under memory pressure: when the pool is too small to hold
     all running sequences, the scheduler evicts rather than deadlocking.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import LlamaConfig
from model.model import generate
from scheduler.block_pool import BlockPool, SequenceState
from scheduler.scheduler import Scheduler
from testutil.random_weights import random_model_weights


# ---------------------------------------------------------------------------
# Shared test config -- tiny enough that tests run fast
# ---------------------------------------------------------------------------

def _tiny_config():
    return LlamaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        intermediate_size=32, rms_norm_eps=1e-5, rope_theta=10000.0,
    )


def _make_scheduler(config, weights, total_blocks=64, block_size=4):
    return Scheduler(
        config=config,
        weights=weights,
        total_blocks=total_blocks,
        block_size=block_size,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# BlockPool unit tests
# ---------------------------------------------------------------------------

def test_block_pool_starts_fully_free():
    pool = BlockPool(total_blocks=8, block_size=4)
    assert pool.num_free_blocks == 8
    assert pool.num_used_blocks == 0


def test_block_pool_allocate_reduces_free_count():
    pool = BlockPool(total_blocks=8, block_size=4)
    blocks = pool.allocate(seq_id=0, num_blocks=3)
    assert blocks is not None
    assert len(blocks) == 3
    assert pool.num_free_blocks == 5
    assert pool.num_used_blocks == 3


def test_block_pool_allocate_returns_none_when_full():
    pool = BlockPool(total_blocks=4, block_size=4)
    pool.allocate(seq_id=0, num_blocks=4)
    result = pool.allocate(seq_id=1, num_blocks=1)
    assert result is None
    assert pool.num_free_blocks == 0


def test_block_pool_free_returns_blocks_to_pool():
    pool = BlockPool(total_blocks=8, block_size=4)
    pool.allocate(seq_id=0, num_blocks=5)
    assert pool.num_free_blocks == 3
    freed = pool.free(seq_id=0)
    assert freed == 5
    assert pool.num_free_blocks == 8


def test_block_pool_freed_blocks_can_be_reallocated():
    pool = BlockPool(total_blocks=4, block_size=4)
    pool.allocate(seq_id=0, num_blocks=4)
    assert pool.allocate(seq_id=1, num_blocks=1) is None  # pool full
    pool.free(seq_id=0)
    result = pool.allocate(seq_id=1, num_blocks=4)
    assert result is not None  # blocks reusable after free


def test_block_pool_blocks_needed_zero_when_tail_has_room():
    pool = BlockPool(total_blocks=8, block_size=4)
    # seq_len=3 fits in 1 block (block_size=4); adding 1 token still fits
    assert pool.blocks_needed(current_seq_len=3, new_tokens=1) == 0


def test_block_pool_blocks_needed_one_when_tail_fills():
    pool = BlockPool(total_blocks=8, block_size=4)
    # seq_len=4 exactly fills 1 block; adding 1 token needs a 2nd block
    assert pool.blocks_needed(current_seq_len=4, new_tokens=1) == 1


def test_block_pool_free_unknown_seq_id_is_safe():
    pool = BlockPool(total_blocks=8, block_size=4)
    freed = pool.free(seq_id=999)
    assert freed == 0
    assert pool.num_free_blocks == 8


# ---------------------------------------------------------------------------
# Scheduler correctness: single request
# ---------------------------------------------------------------------------

def test_scheduler_single_request_finishes():
    torch.manual_seed(0)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)
    sched = _make_scheduler(config, weights)

    prompt = [1, 2, 3, 4]
    seq_id = sched.add_request(prompt, max_new_tokens=5)

    while not sched.is_idle():
        sched.step()

    result = sched.get_result(seq_id)
    assert result is not None
    assert result[:4] == prompt  # prompt preserved unchanged
    assert len(result) == 9      # 4 prompt + 5 generated


def test_scheduler_single_request_matches_direct_generate():
    # The scheduler must produce the same token sequence as calling
    # generate() directly -- it's a serving mechanism, not a different
    # computation.
    torch.manual_seed(0)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)

    prompt_ids = [1, 2, 3, 4]
    max_new = 6

    # Direct generate() baseline.
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    direct_out = generate(prompt_tensor, weights, config, max_new_tokens=max_new)
    direct_tokens = direct_out[0].tolist()

    # Scheduler path.
    sched = _make_scheduler(config, weights)
    seq_id = sched.add_request(prompt_ids, max_new_tokens=max_new)
    while not sched.is_idle():
        sched.step()
    sched_tokens = sched.get_result(seq_id)

    assert sched_tokens == direct_tokens, (
        f"Scheduler output differs from direct generate().\n"
        f"Direct:    {direct_tokens}\n"
        f"Scheduler: {sched_tokens}"
    )


# ---------------------------------------------------------------------------
# Multi-request sharing
# ---------------------------------------------------------------------------

def test_scheduler_two_requests_both_finish():
    torch.manual_seed(1)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)
    sched = _make_scheduler(config, weights, total_blocks=64)

    id_a = sched.add_request([1, 2, 3], max_new_tokens=4)
    id_b = sched.add_request([4, 5, 6, 7], max_new_tokens=3)

    max_steps = 50
    for _ in range(max_steps):
        if sched.is_idle():
            break
        sched.step()

    assert sched.get_result(id_a) is not None, "request A never finished"
    assert sched.get_result(id_b) is not None, "request B never finished"
    assert len(sched.get_result(id_a)) == 7   # 3 prompt + 4 generated
    assert len(sched.get_result(id_b)) == 7   # 4 prompt + 3 generated


def test_scheduler_pool_blocks_reused_across_requests():
    # Submit two requests sequentially (not concurrently) with a pool
    # only large enough for one at a time. The second request should
    # succeed because the first request's blocks were freed on completion.
    torch.manual_seed(2)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)

    # 8 blocks * block_size=4 = 32 token slots. Each request needs
    # ceil(prompt_len / 4) + ceil(max_new_tokens / 4) blocks.
    # prompt=4 tokens, max_new=4 tokens -> 2 blocks. Pool=4 blocks fits
    # one request at a time but not two simultaneously.
    sched = _make_scheduler(config, weights, total_blocks=4, block_size=4)

    id_a = sched.add_request([1, 2, 3, 4], max_new_tokens=4)
    for _ in range(20):
        if sched.is_idle():
            break
        sched.step()
    assert sched.get_result(id_a) is not None

    # Pool should now be fully free -- verify a second request can run.
    assert sched.pool.num_free_blocks == 4

    id_b = sched.add_request([5, 6, 7, 8], max_new_tokens=4)
    for _ in range(20):
        if sched.is_idle():
            break
        sched.step()
    assert sched.get_result(id_b) is not None


# ---------------------------------------------------------------------------
# Eviction under memory pressure
# ---------------------------------------------------------------------------

def test_scheduler_evicts_rather_than_deadlocking():
    # Pool sized so that admitting two requests works initially, but
    # a third request forces eviction of the longest-running one.
    # The scheduler must make progress (not hang) and eventually finish
    # at least one request.
    torch.manual_seed(3)
    config = _tiny_config()
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)

    # 6 blocks total, block_size=4. Each prompt (len=2) needs 1 block.
    # 3 requests -> 3 blocks for prompts; decode steps need more blocks
    # as sequences grow, eventually forcing eviction.
    sched = _make_scheduler(config, weights, total_blocks=6, block_size=4)

    ids = [sched.add_request([i, i+1], max_new_tokens=8) for i in range(3)]

    max_steps = 100
    for _ in range(max_steps):
        if sched.is_idle():
            break
        sched.step()

    # At least one request must have completed.
    finished = [sid for sid in ids if sched.get_result(sid) is not None]
    assert len(finished) >= 1, (
        "Scheduler made no progress under memory pressure -- possible deadlock."
    )
