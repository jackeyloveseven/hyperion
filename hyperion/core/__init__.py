"""
Hyperion Core Module
====================
Exposes core mathematical formulations:
- Sinkhorn optimal transport attention
- Sinkhorn Latent Transport (SLT)
- Layout Residual Suppression (LRS)
- Hyperion optimization losses
- AdaIN & terminal color correction
"""

from hyperion.core.sinkhorn import (
    sinkhorn_normalize,
    sinkhorn_attn,
    sinkhorn_attn_freq,
)
from hyperion.core.slt import sinkhorn_latent_transport, compute_latent_effective_rank
from hyperion.core.lrs import apply_layout_residual_suppression
from hyperion.core.loss import HyperionLoss, sliced_wasserstein_distance
from hyperion.core.color import feature_adain, adain_1d

__all__ = [
    "sinkhorn_normalize",
    "sinkhorn_attn",
    "sinkhorn_attn_freq",
    "sinkhorn_latent_transport",
    "compute_latent_effective_rank",
    "apply_layout_residual_suppression",
    "HyperionLoss",
    "sliced_wasserstein_distance",
    "feature_adain",
    "adain_1d",
]
