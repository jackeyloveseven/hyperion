"""
Hyperion Metric: Update-Space Spatial Coherence rho(d)
======================================================
Implements update-space neighbor correlation metric rho(d) introduced in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI Sec. 3.7)

Quantifies the spatial coupling of latent update fields across distance d.
Proves that MM-DiT backpropagation propagates coherent gradient directions across spatial neighbors
(rho(1) = 0.64, generating brushstrokes), whereas feedforward injection or per-token SLT lacks
local structural coupling (rho(1) = 0.24).
"""

from typing import Union
import numpy as np
import torch


def compute_spatial_neighbor_correlation(
    field: Union[np.ndarray, torch.Tensor],
    distance: int = 1,
    eps: float = 1e-8,
) -> float:
    """
    Computes spatial neighbor cosine similarity rho(d) across distance d.

    Args:
        field: 3D spatial field of shape [H, W, C] (e.g. gradient or latent update per token).
        distance: Token distance d (default: 1 for nearest neighbors).
        eps: Small epsilon for numerical stability.

    Returns:
        rho(d) correlation score in [-1.0, 1.0].
    """
    if isinstance(field, torch.Tensor):
        field_np = field.detach().float().cpu().numpy()
    else:
        field_np = np.asarray(field, dtype=np.float32)

    if field_np.ndim == 2:
        field_np = field_np[:, :, np.newaxis]
    elif field_np.ndim != 3:
        raise ValueError(f"Expected 2D field [H, W] or 3D field [H, W, C], got shape {field_np.shape}")

    h, w, c = field_np.shape
    if h <= distance or w <= distance:
        raise ValueError(f"Spatial dimensions ({h}, {w}) must be greater than distance {distance}")

    norms = np.linalg.norm(field_np, axis=-1, keepdims=True) + eps
    normalized = field_np / norms

    # Vertical neighbor cosine
    cos_h = np.sum(normalized[:-distance, :, :] * normalized[distance:, :, :], axis=-1).mean()
    # Horizontal neighbor cosine
    cos_w = np.sum(normalized[:, :-distance, :] * normalized[:, distance:, :], axis=-1).mean()

    rho = float((cos_h + cos_w) / 2.0)
    return float(np.clip(rho, -1.0, 1.0))
