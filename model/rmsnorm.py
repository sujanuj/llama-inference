"""RMSNorm — the normalization Llama uses instead of LayerNorm.

Implemented as a plain function operating on raw tensors, not an
nn.Module, so every operation here is visible rather than delegated to a
library black box. The actual math is simple but worth being precise
about, since getting normalization subtly wrong is a classic way to
silently degrade a model's outputs without throwing any error at all —
which is exactly the kind of bug a shape-only test would never catch.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

Unlike LayerNorm, there's no mean-centering step (no `x - mean(x)`) and
no learned bias — just a single learned per-channel scale. This is
cheaper to compute and, empirically, works just as well for transformers
at scale, which is why Llama (and most modern LLMs) use it instead of
LayerNorm.
"""

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm over the last dimension of x.

    Args:
        x: (..., hidden_size) — input activations.
        weight: (hidden_size,) — learned per-channel scale.
        eps: small constant added inside the sqrt for numerical
            stability when the mean square is near zero.

    Returns:
        Same shape as x.
    """
    # Computed in float32 regardless of the input's dtype. Real
    # checkpoints are often loaded in bf16/fp16 for memory efficiency,
    # but bf16 in particular has very few mantissa bits — squaring and
    # averaging values in that precision can lose meaningful accuracy.
    # Llama's actual reference implementation does this same upcast for
    # exactly this reason; skipping it would be a subtle but real
    # numerical-fidelity bug, not just a style choice.
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)

    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)

    return weight * x_normed.to(input_dtype)
