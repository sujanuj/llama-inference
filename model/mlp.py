"""SwiGLU MLP — the feedforward block Llama uses inside each transformer
layer, in place of a plain Linear -> ReLU -> Linear MLP.

Standard MLP:   down(activation(up(x)))
SwiGLU MLP:     down(SiLU(gate(x)) * up(x))

The key difference is the GATING: there are TWO parallel "up" 
projections (conventionally called `gate_proj` and `up_proj`), and the
gate projection's output, after a SiLU activation, multiplies elementwise
against the up projection's output before the result is projected back
down. This gives the network a data-dependent way to scale each
intermediate feature, rather than always passing every feature through
the same fixed nonlinearity — empirically this is part of what makes
modern LLMs (Llama, PaLM, etc.) work better than older ReLU-MLP
transformers at the same parameter count.

SiLU(x) = x * sigmoid(x), also known as the "swish" activation —
provided directly by torch as F.silu, used here rather than
hand-deriving sigmoid manually, since SiLU itself isn't the
bug-prone part of this file (the gating multiplication and the THREE
separate weight matrices being wired together correctly is).
"""

import torch
import torch.nn.functional as F


def swiglu_mlp(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Args:
        x: (..., hidden_size)
        gate_proj: (hidden_size, intermediate_size)
        up_proj: (hidden_size, intermediate_size)
        down_proj: (intermediate_size, hidden_size)

    Returns:
        Same leading shape as x, last dim back to hidden_size.
    """
    gate = F.silu(x @ gate_proj)  # (..., intermediate_size)
    up = x @ up_proj  # (..., intermediate_size)
    return (gate * up) @ down_proj  # (..., hidden_size)
