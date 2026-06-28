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

**Planned:**

- [ ] Real weight loading from `meta-llama/Llama-3.2-1B` and a numerical
      cross-check against the reference HuggingFace implementation
- [ ] Naive (unbounded) KV-cache, measured, then a paged KV-cache (fixed-
      size blocks, like OS virtual memory pages), with the memory
      reduction measured directly
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

## Running tests

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v   # 31 tests as of Phase 3
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
│   └── model.py          <- Full forward pass + greedy decoding (Phase 3)
├── testutil/
│   └── random_weights.py  <- Random-but-correctly-shaped weights for testing (Phase 3)
├── tests/               <- one test file per model/ module, same names
├── kvcache/             <- (next) paged KV-cache
├── scheduler/           <- (next) continuous-batching scheduler
├── server/              <- (next) HTTP serving layer
└── benchmark/           <- (next) throughput/latency benchmarks
```

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University
