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

    with torch.no_grad():
        logits = forward(input_ids, weights, checkpoint_config)

    return logits


def main():
    checkpoint_dir = download_checkpoint()

    # A short, fixed token sequence -- the actual token IDs don't matter
    # much for this check (any valid sequence exercises the full
    # embedding -> 16 layers -> output path identically), so a fixed
    # arbitrary sequence keeps this script's output reproducible run to
    # run rather than depending on a tokenizer call.
    input_ids = torch.tensor([[128000, 9906, 1917, 11, 1268, 527, 499]])  # arbitrary valid token IDs

    hf_logits = run_huggingface_reference(checkpoint_dir, input_ids)
    my_logits = run_this_project(checkpoint_dir, input_ids)

    print(f"\nHuggingFace logits shape: {hf_logits.shape}")
    print(f"This project's logits shape: {my_logits.shape}")

    if hf_logits.shape != my_logits.shape:
        print("SHAPE MISMATCH -- cannot compare further. Check config loading.")
        sys.exit(1)

    max_abs_diff = (hf_logits - my_logits).abs().max().item()
    mean_abs_diff = (hf_logits - my_logits).abs().mean().item()

    print(f"\nMax absolute difference:  {max_abs_diff:.6f}")
    print(f"Mean absolute difference: {mean_abs_diff:.6f}")

    # Tolerance rationale: fp32 matmul accumulation order can differ
    # slightly between this project's raw torch ops and HuggingFace's
    # internal implementation (different einsum/matmul call patterns,
    # different intermediate fusions) even when the MATH is identical --
    # so exact equality isn't the right bar. 1e-3 absolute difference on
    # logits that typically range over tens of units is tight enough to
    # catch a real correctness bug (wrong RoPE convention, a missed
    # transpose, wrong GQA head pairing) while tolerant of harmless
    # floating-point accumulation differences.
    tolerance = 1e-3
    if max_abs_diff < tolerance:
        print(f"\nPASS: outputs match within tolerance ({tolerance}).")
    else:
        print(f"\nFAIL: outputs differ by more than tolerance ({tolerance}).")
        print("This points to a real correctness bug -- check, in order of likelihood:")
        print("  1. RoPE convention (rotate_half vs interleaved)")
        print("  2. A missing or backwards .T on a loaded projection")
        print("  3. GQA head-to-group pairing (repeat_kv block vs interleave)")
        print("  4. Tied-embeddings detection picking the wrong output projection")
        sys.exit(1)

    # Also compare the actual predicted next tokens, not just raw
    # logits -- this is the practical, "does it actually generate the
    # same thing" check on top of the numerical one above.
    hf_next = hf_logits[0, -1].argmax().item()
    my_next = my_logits[0, -1].argmax().item()
    print(f"\nHuggingFace predicted next token: {hf_next}")
    print(f"This project predicted next token: {my_next}")
    print("MATCH" if hf_next == my_next else "MISMATCH")


if __name__ == "__main__":
    main()
