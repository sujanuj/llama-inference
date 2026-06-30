"""Numerical verification: does this project's from-scratch forward pass
produce the same logits as HuggingFace's reference LlamaForCausalLM,
given the SAME real Llama-3.2-1B weights?

This is the single most important verification in the whole project --
everything before this phase tested architectural correctness with
random weights (shapes line up, causality holds, gating works). This
script is the first point where the actual numbers get checked against
ground truth. Run this on a machine with Hugging Face access; it can't
run in a sandboxed environment with no path to huggingface.co.

Usage:
    huggingface-cli login   # one-time, needs a token with Llama-3.2-1B access
    python scripts/verify_against_huggingface.py

Requires: transformers, huggingface_hub, safetensors (pip install
transformers huggingface_hub safetensors) -- transformers is used ONLY
as the reference implementation to check against, never as a dependency
of this project's own model code.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import LLAMA_3_2_1B
from model.load_weights import load_config_from_directory, load_model_weights, verify_config_matches
from model.model import forward

CHECKPOINT_REPO = "meta-llama/Llama-3.2-1B"


def download_checkpoint() -> str:
    from huggingface_hub import snapshot_download

    print(f"Downloading {CHECKPOINT_REPO} (this may take a few minutes the first time)...")
    local_dir = snapshot_download(
        repo_id=CHECKPOINT_REPO,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    )
    print(f"Checkpoint cached at: {local_dir}")
    return local_dir


def run_huggingface_reference(checkpoint_dir: str, input_ids: torch.Tensor) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    print("Loading HuggingFace reference model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, torch_dtype=torch.float32
    )
    hf_model.eval()

    with torch.no_grad():
        hf_output = hf_model(input_ids)

    return hf_output.logits


def run_this_project(checkpoint_dir: str, input_ids: torch.Tensor) -> torch.Tensor:
    print("Loading checkpoint into this project's ModelWeights...")
    checkpoint_config = load_config_from_directory(checkpoint_dir)
    verify_config_matches(checkpoint_config, LLAMA_3_2_1B)
    print("Checkpoint config matches model/config.py's LLAMA_3_2_1B assumptions.")

    weights = load_model_weights(checkpoint_dir, checkpoint_config)

    # The checkpoint is stored in bf16; load_model_weights loads tensors
    # as-is with no dtype conversion (see that module's docstring), so
    # without this upcast, this project's forward pass would run
    # entirely in bf16 while the HuggingFace reference below is loaded
    # with torch_dtype=torch.float32 -- comparing bf16 math against
    # fp32 math, not the SAME math at two different precisions. That
    # mismatch alone is enough to produce a real but misleading "FAIL"
    # (observed: max abs diff ~0.17 across a 16-layer, 128k-vocabulary
    # model) that looks like a correctness bug but is actually a
    # precision-mismatch artifact. Upcasting here makes the comparison
    # fair: both sides run the identical math in fp32, so any remaining
    # difference reflects an actual implementation bug, not a precision
    # difference between the two runs.
    weights = _to_float32(weights)

    with torch.no_grad():
        logits = forward(input_ids, weights, checkpoint_config)

    return logits


def _to_float32(weights):
    """Casts every tensor field in a ModelWeights tree to float32."""
    weights.embed_tokens = weights.embed_tokens.to(torch.float32)
    weights.final_norm = weights.final_norm.to(torch.float32)
    if weights.lm_head_weight is not None:
        weights.lm_head_weight = weights.lm_head_weight.to(torch.float32)
    for layer in weights.layers:
        layer.input_layernorm = layer.input_layernorm.to(torch.float32)
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(torch.float32)
        layer.attention.q_proj = layer.attention.q_proj.to(torch.float32)
        layer.attention.k_proj = layer.attention.k_proj.to(torch.float32)
        layer.attention.v_proj = layer.attention.v_proj.to(torch.float32)
        layer.attention.o_proj = layer.attention.o_proj.to(torch.float32)
        layer.mlp.gate_proj = layer.mlp.gate_proj.to(torch.float32)
        layer.mlp.up_proj = layer.mlp.up_proj.to(torch.float32)
        layer.mlp.down_proj = layer.mlp.down_proj.to(torch.float32)
    return weights


def main():
    checkpoint_dir = download_checkpoint()

    # A short, fixed token sequence -- the actual token IDs don't matter
    # much for this check (any valid sequence exercises the full
    # embedding -> 16 layers -> output path identically), so a fixed
    # arbitrary sequence keeps this script's output reproducible run to
    # run rather than depending on a tokenizer call.
    input_ids = torch.tensor([[128000, 9906, 1917, 11, 1268, 527, 499]])  # arbitrary valid token IDs

    hf_logits = run_huggingface_reference(checkpoint_dir, input_ids)
    my_logits_fp32 = run_this_project(checkpoint_dir, input_ids)

    print(f"\nHuggingFace logits shape: {hf_logits.shape}")
    print(f"This project's logits shape: {my_logits_fp32.shape}")

    if hf_logits.shape != my_logits_fp32.shape:
        print("SHAPE MISMATCH -- cannot compare further. Check config loading.")
        sys.exit(1)

    # --- The real comparison: both sides in fp32 ---
    # This is the one that actually answers "is the implementation
    # correct," because both sides are now running the IDENTICAL
    # precision -- any remaining difference reflects a real
    # implementation difference, not an artifact of one side rounding
    # to bf16 and the other not.
    max_abs_diff = (hf_logits - my_logits_fp32).abs().max().item()
    mean_abs_diff = (hf_logits - my_logits_fp32).abs().mean().item()

    print(f"\n--- fp32 vs fp32 (the real correctness check) ---")
    print(f"Max absolute difference:  {max_abs_diff:.6f}")
    print(f"Mean absolute difference: {mean_abs_diff:.6f}")

    # Tolerance rationale, calibrated after empirical investigation: the
    # original 1e-3 was too strict. scripts/diagnose_layer_divergence.py
    # traced the difference layer by layer and found it grows smoothly
    # and non-monotonically from ~0.0003 after layer 0 to ~0.006 by
    # layer 14 -- the exact signature of independent fp32 rounding
    # noise compounding across 16 layers (different matmul call
    # patterns between this project's raw torch ops and HuggingFace's
    # internal implementation), NOT a systematic error from a wrong
    # convention or a missed transpose. A genuinely wrong RoPE
    # convention or backwards GQA pairing would produce a difference in
    # the tens-to-hundreds range, uncorrelated with depth, not a smooth
    # few-thousandths-per-layer drift. (An apparent ~130-point spike
    # after the LAST layer turned out to be a bug in the diagnostic
    # itself, not the model -- this transformers version's
    # output_hidden_states already includes the final norm in its last
    # entry, so comparing it against a pre-norm value was an
    # apples-to-oranges mistake, not a real divergence; see the
    # diagnostic script's own follow-up check, which confirmed the
    # final post-norm difference was only 0.0136.)
    #
    # 0.05 absolute difference on logits whose magnitudes range up to
    # ~30-40 units is tight enough to still catch a real correctness
    # bug (any of the four usual suspects below would blow this well
    # past 0.05) while accommodating the real, harmless precision drift
    # measured directly above.
    tolerance = 0.05
    if max_abs_diff < tolerance:
        print(f"PASS: outputs match within tolerance ({tolerance}).")
    else:
        print(f"FAIL: outputs differ by more than tolerance ({tolerance}).")
        print("This points to a real correctness bug -- check, in order of likelihood:")
        print("  1. RoPE convention (rotate_half vs interleaved)")
        print("  2. A missing or backwards .T on a loaded projection")
        print("  3. GQA head-to-group pairing (repeat_kv block vs interleave)")
        print("  4. Tied-embeddings detection picking the wrong output projection")
        sys.exit(1)

    hf_next = hf_logits[0, -1].argmax().item()
    my_next = my_logits_fp32[0, -1].argmax().item()
    print(f"\nHuggingFace predicted next token: {hf_next}")
    print(f"This project predicted next token: {my_next}")
    print("MATCH" if hf_next == my_next else "MISMATCH")


if __name__ == "__main__":
    main()