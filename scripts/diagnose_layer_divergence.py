import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import LLAMA_3_2_1B
from model.decoder import decoder_layer
from model.load_weights import load_config_from_directory, load_model_weights, verify_config_matches
from model.rmsnorm import rms_norm
from model.rope import compute_rope_frequencies

CHECKPOINT_REPO = "meta-llama/Llama-3.2-1B"

def _to_float32(weights):
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
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    print(f"Downloading {CHECKPOINT_REPO}...")
    checkpoint_dir = snapshot_download(
        repo_id=CHECKPOINT_REPO,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    )

    input_ids = torch.tensor([[128000, 9906, 1917, 11, 1268, 527, 499]])

    print("Loading HuggingFace reference model (fp32) with hidden states...")
    hf_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.float32)
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(input_ids, output_hidden_states=True)
    hf_hidden = hf_out.hidden_states

    print("Loading this project's checkpoint (fp32)...")
    checkpoint_config = load_config_from_directory(checkpoint_dir)
    verify_config_matches(checkpoint_config, LLAMA_3_2_1B)
    weights = load_model_weights(checkpoint_dir, checkpoint_config)
    weights = _to_float32(weights)

    seq_len = input_ids.shape[1]
    my_hidden = weights.embed_tokens[input_ids]

    cos_table, sin_table = compute_rope_frequencies(
        checkpoint_config.head_dim, max_seq_len=seq_len, theta=checkpoint_config.rope_theta
    )
    cos, sin = cos_table[:seq_len], sin_table[:seq_len]

    print("\n--- Layer-by-layer max absolute difference ---")
    diff = (hf_hidden[0] - my_hidden).abs().max().item()
    print(f"After embedding lookup: diff={diff:.6f}  my_max={my_hidden.abs().max().item():.4f}  hf_max={hf_hidden[0].abs().max().item():.4f}")

    with torch.no_grad():
        for i, layer_weights in enumerate(weights.layers):
            my_hidden = decoder_layer(
                my_hidden, layer_weights, cos, sin,
                num_heads=checkpoint_config.num_attention_heads,
                num_kv_heads=checkpoint_config.num_key_value_heads,
                head_dim=checkpoint_config.head_dim,
                rms_norm_eps=checkpoint_config.rms_norm_eps,
                causal=True,
            )
            diff = (hf_hidden[i + 1] - my_hidden).abs().max().item()
            my_max = my_hidden.abs().max().item()
            hf_max = hf_hidden[i + 1].abs().max().item()
            my_nan = torch.isnan(my_hidden).any().item()
            hf_nan = torch.isnan(hf_hidden[i + 1]).any().item()
            print(f"After layer {i:2d}: diff={diff:.6f}  my_max={my_max:.4f}(nan={my_nan})  hf_max={hf_max:.4f}(nan={hf_nan})")

    with torch.no_grad():
        my_final = rms_norm(my_hidden, weights.final_norm, checkpoint_config.rms_norm_eps)
    hf_final = hf_out.hidden_states[-1]
    diff = (hf_final - my_final).abs().max().item()
    print(f"After final norm: diff={diff:.6f}")

    print("\n--- Checking whether hf_hidden[-1] is pre- or post-final-norm ---")
    last_hf_hidden = hf_hidden[-1]
    rms_per_token = last_hf_hidden.pow(2).mean(dim=-1).sqrt()
    print(f"hf_hidden[-1] per-token RMS: {rms_per_token}")
    print(f"my_hidden (pre-norm, after layer 15) per-token RMS: {my_hidden.pow(2).mean(dim=-1).sqrt()}")
    print(f"my_final (post-norm) per-token RMS: {my_final.pow(2).mean(dim=-1).sqrt()}")

if __name__ == "__main__":
    main()