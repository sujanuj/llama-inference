"""Architecture configuration for Llama-3.2-1B.

These values are taken from Meta's published config.json for
meta-llama/Llama-3.2-1B. They are NOT loaded from the network here (this
environment can't reach huggingface.co), but every shape and numeric
constant in this file should match the real checkpoint exactly — when
real weights are loaded on a machine with Hub access, a config mismatch
would surface immediately as a tensor shape error, which is itself a
useful sanity check that this file is right.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int = 128_256
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # grouped-query attention: 4 query heads per KV head
    head_dim: int = 64  # hidden_size // num_attention_heads
    intermediate_size: int = 8192  # SwiGLU MLP inner dimension
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    max_position_embeddings: int = 131_072

    @property
    def num_query_groups(self) -> int:
        """How many query heads share each single KV head.

        32 query heads / 8 KV heads = 4. This is the entire point of
        grouped-query attention (GQA): instead of every query head
        having its own K/V projection (full multi-head attention,
        num_key_value_heads == num_attention_heads), groups of query
        heads share one K/V head, cutting the KV-cache's memory
        footprint by this same factor — directly relevant later when
        measuring paged KV-cache memory usage, since GQA is already a
        4x reduction before paging adds anything on top.
        """
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self.num_attention_heads // self.num_key_value_heads


LLAMA_3_2_1B = LlamaConfig()
