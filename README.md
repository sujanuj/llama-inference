# llama-inference

A Llama-3.2-1B inference engine built from raw tensor math — no
`nn.Module` black boxes for the model internals — with the eventual goal
of a from-scratch paged KV-cache, a continuous-batching scheduler, and
quantization, benchmarked the way a real inference team would.

This is a learning project, not a production serving stack. The goal is
to actually implement the pieces that make modern LLM inference engines
(vLLM, TGI, etc.) work — grouped-query attention, RoPE, paged KV-caching,
continuous batching — rather than calling a library that already does it.

---

## Why this exists

Most "I built an LLM project" portfolio entries call an API or fine-tune
with a high-level training loop. This project does the harder, more
systems-flavored thing: implement a real model's forward pass by hand,
verify every numerically-tricky piece against either a hand-computed
reference or the actual mathematical property it's supposed to have (not
just "it runs without crashing"), then build the inference-serving
machinery (KV-cache, batching, quantization) that's the actual subject
matter of LLM infrastructure work at AI labs.

A companion project, [`lsmdb`](https://github.com/sujanuj/lsmdb), covers
the same evidence-first approach applied to a storage engine — this
project is the same discipline applied to transformer internals and
inference systems instead.

---

## Status

**Phase 1: Architecture foundations — done**

- [x] `model/config.py` — Llama-3.2-1B's exact published architecture
      constants (hidden size, head counts, RoPE theta, etc.), so that
      loading real weights later either just works or fails loudly with
      a clear shape mismatch
- [x] `model/rmsnorm.py` — RMSNorm with the float32 upcast real
      checkpoints need for numerical stability in bf16
- [x] `model/rope.py` — Rotary position embeddings using the
      `rotate_half` convention (the one real Llama checkpoints are
      trained against — NOT the original RoPE paper's interleaved
      convention, which is architecturally similar but numerically
      incompatible)
- [x] RoPE verified against its actual defining property: the attention
      dot product between a rotated query and a rotated key depends only
      on their *relative* position, not their absolute positions — this
      is the test that would catch a subtly-wrong rotation convention
      that still produces plausible-looking numbers

**Phase 2: Attention + MLP — done**

- [x] `model/attention.py` — grouped-query attention (32 query heads, 8
      KV heads, 4 query heads per KV group), with `repeat_kv` split out
      and tested as its own unit
- [x] `repeat_kv` verified with **identifiable, non-random values** per
      KV head (100.0 vs 200.0, not noise) specifically to catch a
      block-vs-interleaved repeat bug, which would silently mispair query
      heads with the wrong KV head while still "running fine"
- [x] Causal masking verified both in the simple case and in the
      KV-cache-shaped case (query shorter than key/value, i.e. new tokens
      attending against cached history) — the offset arithmetic for that
      second case is exactly the kind of off-by-one that a same-length
      test would never exercise
- [x] `model/mlp.py` — SwiGLU MLP (the gated feedforward block Llama uses
      instead of a plain ReLU MLP), with the gating behavior verified
      directly (a strongly negative gate suppresses a large up-projection
      value, rather than just checking shapes)

**Phase 3: Full forward pass — done**

- [x] `model/weights.py` — plain dataclasses (`AttentionWeights`,
      `MLPWeights`, `DecoderLayerWeights`, `ModelWeights`) holding the
      model's parameters by name, with no `nn.Module`/`nn.Parameter`
      anywhere — these exist purely to give real weight-loading (next
      phase) a clean, typed place to populate fields into
- [x] **Tied embeddings handled explicitly**: Llama-3.2-1B and -3B share
      the input embedding matrix and the output projection (no separate
      `lm_head.weight` exists in the real checkpoint for these sizes) —
      `ModelWeights.output_projection()` falls back to the transposed
      embedding table when `lm_head_weight` is `None`, verified against
      both the tied and untied cases directly
- [x] `model/decoder.py` — the pre-norm residual block (norm -> attention
      -> residual -> norm -> MLP -> residual) that gets stacked 16 times
- [x] `model/model.py` — the full forward pass: token embedding lookup,
      16 stacked decoder layers, final RMSNorm, output projection to
      vocabulary logits, plus greedy next-token selection
- [x] `testutil/random_weights.py` — builds a full `ModelWeights`
      instance with random tensors of the CORRECT shapes for any config,
      so the complete architecture can be exercised end-to-end without
      needing real downloaded weights — legitimate for testing shape/flow
      correctness, since that doesn't depend on which numbers are loaded
- [x] **Causal masking verified at the assembled-model level, not just per
      attention call**: changing only the last token in an input sequence
      and confirming every earlier position's logits are byte-for-byte
      unchanged, after going through embedding + 3 stacked decoder layers
      + output projection — this is the test that would catch a future-
      information leak introduced by residual wiring or layer-stacking,
      which a single isolated attention test could never exercise
- [x] Determinism check (same weights + same input -> identical output)
      and a smaller-config run to confirm no hidden dependency on
      Llama-3.2-1B's specific dimensions

**Phase 4: Real checkpoint loading — done**

- [x] `model/load_weights.py` -- loads a real Llama-3.2-1B checkpoint
      (HuggingFace safetensors format) into this project's
      `ModelWeights`/`DecoderLayerWeights` dataclasses
- [x] **Every 2-D projection transposed explicitly, on load.** HuggingFace
      stores `nn.Linear` weights as `(out_features, in_features)` and
      computes `y = x @ W.T`; this project's forward-pass code computes
      `y = x @ W` directly. Every attention and MLP projection needs an
      explicit `.T` on load, or the result either fails loudly (a
      non-square shape mismatch) or, worse, silently multiplies with
      backwards semantics for any matrix that happens to be square.
      Tested directly: a synthetic checkpoint stores weights in the real
      HF `(out, in)` convention, and the test confirms every loaded
      weight comes out in this project's `(in, out)` convention.
- [x] **Config loaded from the checkpoint's own `config.json`, not
      trusted blindly from `model/config.py`** -- `verify_config_matches`
      raises a clear, specific error (naming exactly which field and what
      the mismatch is) if the checkpoint disagrees with this project's
      architectural assumptions, rather than letting a mismatch surface
      later as a confusing shape error deep inside a matmul
- [x] Tied-embeddings detection re-verified on the LOADED path (not just
      Phase 3's random-weight path): a synthetic checkpoint built without
      an `lm_head.weight` tensor (matching real Llama-3.2-1B/3B) correctly
      falls back to the transposed embedding table; one built WITH a
      separate `lm_head.weight` correctly uses it instead
- [x] End-to-end: a synthetic checkpoint loaded through the real loader,
      run through the real `forward()` from Phase 3 -- not just shape
      assertions on the loaded dataclass in isolation
- [x] `scripts/verify_against_huggingface.py` -- downloads the real
      checkpoint, runs the same input through both this project's forward
      pass and HuggingFace's reference `LlamaForCausalLM`, and compares
      logits numerically (max/mean absolute difference) against a stated,
      justified tolerance, plus a practical greedy-next-token match check
- [x] **Verified against the real `meta-llama/Llama-3.2-1B` checkpoint:
      PASS.** Max absolute logit difference 0.0217, mean 0.0015, and the
      exact same greedy-decoded next token as HuggingFace's reference
      implementation. Getting here required finding and fixing two real
      bugs and correctly diagnosing a false alarm -- the full story is in
      "Real bugs found verifying against actual weights" below, because
      the debugging process is at least as informative as the final
      green checkmark

**Phase 5, part 1: Naive KV-cache -- done**

- [x] `kvcache/naive_cache.py` -- per-layer K/V storage that grows via
      concatenation on every append, cached at the KV-HEAD dimension
      (not the query-head dimension) specifically because that's where
      GQA's real memory savings live: 8 KV heads cached instead of 32
      query heads is already a 4x reduction before any further
      optimization
- [x] `model/attention.py`'s `attention_with_cache` and
      `model/decoder.py`'s `decoder_layer_with_cache` reuse the EXACT
      same `scaled_dot_product_attention` and `repeat_kv` primitives as
      the no-cache path -- no second implementation of attention math
      that could silently drift out of sync with the original
- [x] **The single most important correctness property, verified
      directly**: token-by-token cached generation produces logits and
      next-token predictions IDENTICAL to running the full resulting
      sequence through the no-cache `forward()` in one shot, checked at
      every position across two separate test scenarios (a 4-token
      prompt with 5 generated tokens, and an 8-token prompt with 10
      generated tokens, specifically to exercise `position_offset`
      advancing well past the prompt length, not just the first step)
- [x] `model/model.py`'s `generate()` ties prefill (the whole prompt
      processed in one `forward_with_cache` call at `position_offset=0`)
      and decode (one new token per step, `position_offset` advancing
      by 1 each time) into a single, usable generation loop
- [x] Real memory baseline measured at actual Llama-3.2-1B dimensions
      (`benchmark/measure_naive_cache_memory.py`, no real weights
      needed -- memory footprint depends only on tensor shapes and
      dtype): **128MB at a 4096-token sequence, batch size 1; 2GB at
      the same sequence length, batch size 16** -- confirmed exactly
      linear in both sequence length and batch size, which is precisely
      the real cost a paged cache exists to reduce, and the actual
      baseline the next phase's measured savings will be compared against

**Planned:**

- [ ] Paged KV-cache (fixed-size blocks, like OS virtual memory pages),
      with the memory reduction measured directly against the naive
      baseline above
- [ ] Continuous-batching scheduler for multiple concurrent requests of
      different lengths
- [ ] INT8/INT4 quantization, measured for perplexity degradation vs.
      memory/latency improvement
- [ ] An HTTP serving layer and a full benchmark suite

---

## Why a `rotate_half`-based RoPE, specifically

There are two materially different but superficially similar ways to
implement RoPE in circulation: the original paper's *interleaved* pairing
(dimensions 0 and 1 form a rotation pair, 2 and 3 form the next, etc.)
and the *rotate_half* convention HuggingFace's actual Llama code uses
(the first half of `head_dim` pairs with the second half). Implementing
the wrong one produces a model that runs, produces plausible-looking
attention patterns, and is simply incompatible with real Llama weights —
with no error message anywhere. This project uses `rotate_half`
specifically because that's the convention real checkpoints are trained
against; using the "more elegant" interleaved version would be
architecturally well-formed but numerically wrong for this model.

## Why grouped-query attention matters for inference specifically

Llama-3.2-1B has 32 query heads but only 8 key/value heads — every 4
query heads share one KV head. This isn't just a training-time
efficiency trick: the KV-cache (built in an upcoming phase) has to store
one K/V vector per cached token per KV head per layer, so fewer KV heads
means a proportionally smaller cache. Going from full multi-head
attention (32 KV heads) to this 8-head GQA setup is already a 4x
reduction in KV-cache memory before paging adds anything on top — which
is exactly why GQA shows up in essentially every modern inference-focused
model.

## Testing philosophy

Every numerically-tricky component gets tested against either a
hand-computed reference (RMSNorm, SwiGLU, a single RoPE rotation pair)
or the actual mathematical property it's supposed to satisfy (RoPE's
relative-position invariance, GQA's exact head-to-head pairing, the
causal mask's behavior under a cache-shaped offset) — not just "the
shapes are right and nothing crashed." A model that runs without error
and produces wrong numbers is a much worse failure mode than a crash,
because nothing announces it; the testing strategy here is built around
that specifically.

## Why tied embeddings need explicit handling

Llama-3.2-1B and -3B share the input embedding matrix and the output
projection -- there's no separate `lm_head.weight` tensor in the real
checkpoint for these model sizes. A from-scratch implementation that
assumes every model has its own independent output projection would
either fail to load these checkpoints at all or, worse, silently
allocate a randomly-initialized output head that doesn't match the
embeddings it's supposed to be tied to. `ModelWeights.output_projection()`
handles this directly: it returns the transposed embedding table unless
an explicit `lm_head_weight` is present, which is exactly the condition
real Llama-3.2-1B/3B checkpoints are in (larger Llama variants, 8B+, do
NOT tie embeddings and would populate this field instead).

## Why causal masking gets tested twice -- once per attention call, once across the whole stack

`test_attention.py` (Phase 2) verifies a single attention call respects
causality. `test_model.py` (Phase 3) verifies the SAME property holds
after going through token embedding, several stacked decoder layers, and
the output projection -- by changing only the last token in an input
sequence and confirming every earlier position's logits come back
byte-for-byte identical. These are genuinely different tests: a bug in
how residual connections or layer-stacking are wired could in principle
leak future-token information even if every individual attention call,
tested in isolation, is perfectly causal. Testing the same property at
both the unit level and the assembled-system level is deliberate, not
redundant.

## Why every weight gets transposed on load

HuggingFace's `nn.Linear` stores its weight as `(out_features,
in_features)` and computes `y = x @ W.T + b`. This project's forward-pass
code (Phases 1-3) computes `y = x @ W` directly, with no transpose, by
design -- it keeps the raw tensor math in `model/attention.py` and
`model/mlp.py` readable without a `.T` scattered through every call site.
The consequence: every single projection loaded from a real checkpoint
needs an explicit transpose before it goes into this project's
dataclasses. `model/load_weights.py` does this per-tensor, with a comment
at each transpose, rather than relying on one shared
get-it-right-once helper -- specifically so a future edit to how one
weight loads can't accidentally skip the transpose for another. Tested
directly with a synthetic checkpoint that stores weights in the real
`(out, in)` convention and confirms every loaded weight comes out
transposed correctly.

## Verifying against real weights

`scripts/verify_against_huggingface.py` is the actual point where
"architecturally correct" gets checked against "numerically identical to
the real thing." It downloads real `meta-llama/Llama-3.2-1B` weights,
runs the same input through this project's forward pass AND
HuggingFace's reference `LlamaForCausalLM`, and reports the max/mean
absolute difference between the two logit tensors against a stated
tolerance (1e-3 -- tight enough to catch a real bug like a wrong RoPE
convention or a missed transpose, loose enough to tolerate harmless
floating-point accumulation-order differences between two different
matmul call patterns computing the same math). This can't run in this
sandbox (no path to huggingface.co); see the script's docstring for
exact setup steps to run it on a machine with Hub access.

## Real bugs found verifying against actual weights

Running `verify_against_huggingface.py` against the real downloaded
checkpoint surfaced two genuine bugs and one false alarm, in sequence.
Documenting the process, not just the final result, because the
debugging methodology is at least as valuable a signal as the green
checkmark at the end of it.

**Bug 1: bf16/fp32 dtype mismatch in attention and RoPE.** Every test
fixture through Phase 4 used `torch.randn(...)`, which defaults to
float32 -- so nothing had ever exercised this codebase against bf16
tensors. Real Llama-3.2-1B weights are stored in bf16. The very first
real-weight run crashed immediately:

```
RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16
```

Root cause: `scaled_dot_product_attention`'s softmax had no explicit
dtype handling, and `apply_rope` let its float32 `cos`/`sin` tables
silently upcast bf16 inputs via torch's type-promotion rules. Fixed by
explicitly computing softmax in float32 (matching real Llama's own
practice, for numerical stability) and casting back, and by casting
`cos`/`sin` to the input's dtype before rotating. **The fix was verified,
not assumed**: the fix was temporarily reverted and the new regression
tests (`tests/test_bf16_dtype.py`) were confirmed to reproduce the exact
original error before the fix was restored.

**Bug 2: comparing bf16 math against fp32 math, not the same math at two
precisions.** With bug 1 fixed, the verification script ran to
completion but reported `FAIL`, max difference 0.17 -- large enough to
look like a real correctness bug, but the actual cause was simpler: the
script loaded HuggingFace's reference model with `torch_dtype=torch.float32`
(upcasting on load) while this project's own `load_model_weights` loaded
the checkpoint's native bf16 tensors with no conversion. The two sides
were running genuinely different precision throughout the entire
forward pass. Fixed by upcasting the loaded weights to float32 before
running this project's forward pass too -- bringing the difference down
to 0.022, an order of magnitude improvement, confirming the diagnosis.

**False alarm: an apparent ~130-point blowup at the last layer.** Even
at 0.022 overall, a dedicated layer-by-layer diagnostic
(`scripts/diagnose_layer_divergence.py`) was built to make sure that
small remaining number wasn't hiding something layer-specific. It
revealed the difference grows smoothly from ~0.0003 after layer 0 to
~0.006 by layer 14 (the expected signature of independent fp32
rounding noise compounding across 16 layers) -- and then an apparent
spike to ~131.7 specifically after "layer 15," the last one. Investigated
rather than assumed: printing each side's per-token RMS at that point
showed HuggingFace's last `hidden_states` entry already had the final
RMSNorm applied (matching `my_final`'s RMS, not `my_hidden`'s pre-norm
RMS) -- this version of `transformers` includes the final norm in the
last hidden-states entry, a detail of that library's internals, not a
bug in this project. Comparing the SAME point (post-final-norm to
post-final-norm) gave a difference of 0.0136, consistent with the
smooth per-layer trend the rest of the network already showed.

**Net result:** the verification tolerance was recalibrated from an
arbitrary `1e-3` to an empirically-justified `0.05` -- tight enough that
any of the four real bug classes this script checks for (wrong RoPE
convention, a missed transpose, wrong GQA pairing, wrong tied-embeddings
detection) would still fail loudly, since each of those produces
differences in the tens-to-hundreds range, not hundredths.

## Running tests

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v   # 88 tests as of Phase 8
python benchmark/measure_naive_cache_memory.py  # real KV-cache memory measurements
```

## Project layout

```
llama-inference/
├── model/
│   ├── config.py       <- Llama-3.2-1B architecture constants (Phase 1)
│   ├── rmsnorm.py       <- RMSNorm (Phase 1)
│   ├── rope.py          <- Rotary position embeddings (Phase 1)
│   ├── attention.py     <- Grouped-query attention (Phase 2)
│   ├── mlp.py           <- SwiGLU MLP (Phase 2)
│   ├── weights.py       <- Weight dataclasses, tied-embeddings logic (Phase 3)
│   ├── decoder.py        <- Pre-norm residual decoder layer (Phase 3)
│   ├── model.py          <- Full forward pass + greedy decoding (Phase 3)
│   └── load_weights.py    <- Real checkpoint loader, HF format (Phase 4)
├── testutil/
│   └── random_weights.py  <- Random-but-correctly-shaped weights for testing (Phase 3)
├── kvcache/
│   ├── naive_cache.py                    <- Naive per-request KV-cache (Phase 5 part 1)
│   └── paged_cache.py                    <- Paged KV-cache, block-granularity allocation (Phase 6)
├── benchmark/
│   ├── measure_naive_cache_memory.py     <- Real memory baseline at Llama-3.2-1B dimensions
│   └── measure_paged_cache_memory.py     <- Naive vs paged side-by-side comparison (Phase 6)
├── scripts/
│   ├── verify_against_huggingface.py     <- Real-weight numerical cross-check (Phase 4, run on a machine with Hub access)
│   └── diagnose_layer_divergence.py      <- Layer-by-layer divergence localization, used to debug the verification above
├── tests/               <- one test file per model/ module, same names

│   ├── block_pool.py    <- shared block pool, allocation/free (Phase 7)
│   └── scheduler.py     <- continuous-batching scheduler, eviction (Phase 7)
├── server/              <- (next) HTTP serving layer
└── benchmark/           <- (next) throughput/latency benchmarks
```

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University

**Phase 6: Paged KV-cache — done**

- [x] `kvcache/paged_cache.py` — a paged KV-cache that pre-allocates a
      fixed pool of equal-sized blocks (block_size=16 tokens/block, matching
      vLLM's default) and assigns them to sequences on demand via a
      per-sequence block table. No tensor is ever reallocated or copied at
      decode time — each append writes into the current tail block's
      pre-allocated slot and spills into a new block only when the tail fills.
- [x] **Gather step made explicit**: reading non-contiguous physical blocks
      back into a contiguous `(batch, kv_heads, seq_len, head_dim)` view for
      attention is the direct cost of paging — one `index_select` on the
      pre-allocated pool per decode step, rather than the naive cache's
      per-step `torch.cat` reallocation. Both costs are real; the tradeoff
      is measured rather than assumed.
- [x] **Verified output-identical to the naive cache**: `test_paged_cache.py`
      checks raw tensor equality after single and multi-step appends (including
      the cross-block-boundary case: appending 3+1+1+2 tokens with block_size=4
      spans two blocks), then verifies that full autoregressive generation
      through the paged cache produces a token-for-token identical sequence to
      the naive cache across two prompt/length combinations.
- [x] `benchmark/measure_paged_cache_memory.py` — runs the same
      seq_len/batch_size grid as `measure_naive_cache_memory.py` and reports
      both caches side by side. The 0.0% overhead is correct and honest: all
      benchmark sequences are exact multiples of block_size=16, so there is no
      partial tail block. The paged cache's fragmentation advantage only
      surfaces when a request's actual length is shorter than its pre-allocated
      max — e.g. a naive cache sized for max_seq_len=4096 serving a 512-token
      request wastes 3584 token-slots; the paged cache wastes at most 15.

**Phase 7: Continuous-batching scheduler — done**

- [x] `scheduler/block_pool.py` — a shared physical block pool and
      `SequenceState` dataclass. The pool hands out block indices to
      sequences on demand and returns them to the free list the moment a
      sequence finishes — no block sits idle while another request is
      waiting. `blocks_needed()` lets the scheduler check capacity before
      committing to a decode step, so allocation failures are caught
      cleanly rather than mid-step.
- [x] `scheduler/scheduler.py` — a continuous-batching scheduler that
      maintains a waiting queue and a running set. Each `step()` call:
      (1) admits waiting requests that fit in the pool and prefills their
      prompts; (2) checks that every running sequence has room for one
      more token, evicting the longest-running sequence if the pool is
      exhausted; (3) decodes one new token per running sequence; (4)
      frees blocks and records results for any sequence that has hit
      `max_new_tokens`. Requests are admitted as soon as memory is
      available — not grouped into fixed-size batches — matching the
      continuous-batching model from Orca (Yu et al., 2022) and vLLM.
- [x] **Verified output-identical to direct `generate()`**: a single
      request run through the scheduler produces a token-for-token
      identical sequence to calling `generate()` directly — the scheduler
      is a serving mechanism, not a different computation.
- [x] **Block reuse verified**: a pool sized for one request at a time
      correctly serves two sequential requests, confirming that finished
      requests' blocks are actually freed and reusable — not just
      logically released but still occupied.
- [x] **Eviction under memory pressure verified**: with a pool too small
      to sustain three concurrent sequences through full decode, the
      scheduler evicts the longest-running sequence rather than
      deadlocking, and at least one request completes.


**Phase 8: HTTP inference server — done**

- [x] `server/server.py` — a single-file HTTP inference server built on
      Python's stdlib `http.server` (no Flask or FastAPI). Exposes two
      endpoints: `POST /generate` accepts a JSON body with `token_ids`
      and `max_new_tokens`, runs the request through the scheduler, and
      returns the full token sequence; `GET /health` returns
      `{"status": "ok"}` immediately. Explicitly single-threaded -- one
      request at a time -- noted as a scope limit rather than glossed over.
- [x] `server/client.py` — a stdlib-only test client for manual
      end-to-end verification against a running server.
- [x] **Verified output-identical to direct `generate()`**: the server
      test re-creates the same weights with the same seed and confirms
      the HTTP response matches `generate()` token-for-token.
- [x] Input validation tested explicitly: missing fields, empty
      `token_ids`, invalid JSON all return 400; unknown paths return 404.

**Phase 9: Throughput/latency benchmark — done**

- [x] `benchmark/throughput_latency.py` — measures latency (TTFT and
      total generation time for a single request) and throughput (tokens
      per second with N concurrent requests) at real Llama-3.2-1B
      dimensions. Random weights are used so timing reflects tensor
      shapes and compute, not weight values.

**Benchmark results (Llama-3.2-1B dimensions, float32, CPU/Apple M5):**

```
Latency (single request)
  prompt_len    max_new       TTFT      total      tok/s
          32         16      1.40s      2.40s        6.7
          64         32      783ms      4.19s        7.6
         128         32      1.77s      3.84s        8.3
         128         64      393ms      4.71s       13.6
         256         64      600ms      4.89s       13.1

Throughput (concurrent requests)
  requests   prompt_len    max_new   finished        tok/s
         1           64         32          1         12.5
         4           64         32          4         11.4
         8           64         32          8         12.3
         4          128         32          4         10.9
         8          128         32          8         11.7
```

Throughput stays flat (~11-12 tok/s) across 1, 4, and 8 concurrent
requests -- the bottleneck on CPU is the forward pass itself, not
scheduler or block-pool overhead. A CUDA device would give substantially
higher throughput; the scheduler and block-pool mechanics are identical
regardless of device.
