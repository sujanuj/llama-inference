"""Throughput and latency benchmark for the inference stack.

Measures two things at Llama-3.2-1B's architecture dimensions (random
weights -- no checkpoint needed, since timing depends on tensor shapes
and compute, not on weight values):

  1. LATENCY: wall-clock time from request submission to last token for
     a single request, across a range of prompt lengths and output
     lengths. Reports time-to-first-token (TTFT) and total generation
     time separately -- TTFT measures prefill cost, total time measures
     prefill + decode cost combined.

  2. THROUGHPUT: tokens generated per second when the scheduler runs
     a queue of N requests concurrently, measured as:
       total_generated_tokens / total_wall_clock_seconds
     This is the number that matters for serving: how many output tokens
     per second can the stack sustain under load.

Both measurements use the real scheduler (scheduler/scheduler.py) and
paged KV-cache (kvcache/paged_cache.py) -- the same stack that serves
HTTP requests -- so the numbers reflect the actual end-to-end cost, not
just the forward pass in isolation.

Run with:
  python benchmark/throughput_latency.py

No GPU or real weights needed. Results will vary by machine; the table
printed to stdout is what goes in the README.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import LLAMA_3_2_1B
from scheduler.scheduler import Scheduler
from testutil.random_weights import random_model_weights


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16
TOTAL_BLOCKS = 1024   # enough for all benchmark sequences without eviction
DTYPE = torch.float32  # float32 for CPU; bf16 would be used on GPU


def make_scheduler(weights) -> Scheduler:
    return Scheduler(
        config=LLAMA_3_2_1B,
        weights=weights,
        total_blocks=TOTAL_BLOCKS,
        block_size=BLOCK_SIZE,
        dtype=DTYPE,
    )


def make_prompt(length: int) -> list:
    """A prompt of the given length with token IDs in [1, vocab_size)."""
    torch.manual_seed(42)
    return torch.randint(1, LLAMA_3_2_1B.vocab_size, (length,)).tolist()


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(weights, prompt_len: int, max_new_tokens: int) -> dict:
    """Run a single request and measure TTFT and total generation time.

    TTFT (time to first token): wall-clock time from add_request() to
    the first step() call that returns a token for this request. This
    covers prefill only -- the prompt is processed in one
    forward_with_cache call during the first step().

    Total time: wall-clock time from add_request() to the step() that
    returns the last token. Covers prefill + all decode steps.
    """
    sched = make_scheduler(weights)
    prompt = make_prompt(prompt_len)

    t0 = time.perf_counter()
    seq_id = sched.add_request(prompt, max_new_tokens=max_new_tokens)

    ttft = None
    tokens_generated = 0

    while not sched.is_idle():
        results = sched.step()
        if seq_id in results and results[seq_id] is not None:
            tokens_generated += 1
            if ttft is None:
                ttft = time.perf_counter() - t0

    total_time = time.perf_counter() - t0

    return {
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "ttft_ms": ttft * 1000 if ttft else None,
        "total_ms": total_time * 1000,
        "tokens_generated": tokens_generated,
    }


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------

def measure_throughput(weights, num_requests: int, prompt_len: int, max_new_tokens: int) -> dict:
    """Run num_requests requests concurrently through the scheduler and
    measure total tokens generated per second.

    All requests are submitted before the scheduling loop starts, so the
    scheduler can run them concurrently (admitting as many as the block
    pool allows at each step). The throughput number reflects real
    multi-request serving, not sequential single-request processing.
    """
    sched = make_scheduler(weights)
    prompt = make_prompt(prompt_len)

    seq_ids = [
        sched.add_request(prompt, max_new_tokens=max_new_tokens)
        for _ in range(num_requests)
    ]

    t0 = time.perf_counter()
    while not sched.is_idle():
        sched.step()
    elapsed = time.perf_counter() - t0

    total_tokens = sum(
        len(sched.get_result(sid)) - prompt_len
        for sid in seq_ids
        if sched.get_result(sid) is not None
    )
    finished = sum(1 for sid in seq_ids if sched.get_result(sid) is not None)

    return {
        "num_requests": num_requests,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "finished": finished,
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def human_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"


def main():
    config = LLAMA_3_2_1B
    print("Loading random weights at Llama-3.2-1B dimensions...")
    print(f"({config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads} Q/KV, "
          f"dtype={DTYPE}, device=cpu)\n")
    torch.manual_seed(0)
    weights = random_model_weights(config, num_layers=config.num_hidden_layers)

    # --- Latency table ---
    print("=== Latency (single request) ===")
    print(f"{'prompt_len':>12} {'max_new':>10} {'TTFT':>10} {'total':>10} {'tok/s':>10}")

    latency_cases = [
        (32,  16),
        (64,  32),
        (128, 32),
        (128, 64),
        (256, 64),
    ]
    for prompt_len, max_new in latency_cases:
        r = measure_latency(weights, prompt_len, max_new)
        tps = r["tokens_generated"] / (r["total_ms"] / 1000)
        print(
            f"{prompt_len:>12} {max_new:>10} "
            f"{human_ms(r['ttft_ms']):>10} "
            f"{human_ms(r['total_ms']):>10} "
            f"{tps:>10.1f}"
        )

    print()

    # --- Throughput table ---
    print("=== Throughput (concurrent requests) ===")
    print(f"{'requests':>10} {'prompt_len':>12} {'max_new':>10} {'finished':>10} {'tok/s':>12}")

    throughput_cases = [
        (1,  64, 32),
        (4,  64, 32),
        (8,  64, 32),
        (4,  128, 32),
        (8,  128, 32),
    ]
    for num_req, prompt_len, max_new in throughput_cases:
        r = measure_throughput(weights, num_req, prompt_len, max_new)
        print(
            f"{num_req:>10} {prompt_len:>12} {max_new:>10} "
            f"{r['finished']:>10} {r['tokens_per_sec']:>12.1f}"
        )

    print()
    print("Notes:")
    print("  Random weights: timing reflects real tensor shapes and compute,")
    print("  not weight values. Numbers would be identical with real weights")
    print("  on the same hardware.")
    print("  Device: CPU (Apple M5). A CUDA device would give substantially")
    print("  higher throughput; the architecture and scheduler overhead are")
    print("  the same regardless of device.")
    print("  Throughput scales with num_requests up to the point where the")
    print("  scheduler's step() overhead and block-pool contention dominate.")


if __name__ == "__main__":
    main()
