"""
Hyperion: Color Alignment & Normalization Utilities
===================================================
Implements latent-space AdaIN warm-start and terminal color correction
disclosed in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI)
"""

from typing import Optional
import torch


def feature_adain(
    content: torch.Tensor,
    style: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Per-step latent AdaIN warm-start (disclosed in TPAMI paper Sec. 3.6 / Algorithm 1).

    Shifts the content latent's per-channel / per-token statistics toward the style latent
    before SLT pre-alignment and Adam optimization.

    Args:
        content: Content latent of shape [B, L, D] or [B, 1, L, D].
        style: Noised style latent of shape [B, Ls, D] or [B, 1, Ls, D].
        alpha: Mixing coefficient (alpha_w = 0.5 for T2I, 0.3 for I2I).
        eps: Small epsilon for numerical stability.

    Returns:
        Latent tensor with warm-started statistics.
    """
    orig_dtype = content.dtype
    c_f = content.float()
    s_f = style.float()

    # Determine token dimension: dim (-2, -1) for [B, C, H, W], or dim -2 for [B, L, D] / [B, 1, L, D]
    if content.dim() == 4 and content.shape[1] > 1:
        reduce_dim = (-2, -1)
    else:
        reduce_dim = -2

    c_mean = c_f.mean(dim=reduce_dim, keepdim=True)
    c_std = torch.clamp(c_f.std(dim=reduce_dim, keepdim=True), min=eps)

    s_mean = s_f.mean(dim=reduce_dim, keepdim=True)
    s_std = torch.clamp(s_f.std(dim=reduce_dim, keepdim=True), min=eps)

    transferred = ((c_f - c_mean) / c_std) * s_std + s_mean
    out = alpha * transferred + (1.0 - alpha) * c_f
    return out.to(orig_dtype)


def adain_1d(
    source_lat: torch.Tensor,
    target_lat: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Terminal 1-D color correction (Algorithm 1, line 89 in TPAMI paper).

    Matches the global per-channel mean and standard deviation of the final denoised
    latent z_0 to the clean style latent z_s before VAE decoding.

    Args:
        source_lat: Generated latent z_0 of shape [B, L, D], [L, D], or [B, C, H, W].
        target_lat: Clean reference style latent z_s of shape [B, Ls, D], [Ls, D], or [B, C, H, W].
        eps: Small epsilon for numerical stability.

    Returns:
        Color-corrected latent tensor of shape matching source_lat.
    """
    orig_dtype = source_lat.dtype
    s_f = source_lat.float()
    t_f = target_lat.float()

    if s_f.dim() == 4 and s_f.shape[1] > 1:
        # [B, C, H, W] format
        s_mean = s_f.mean(dim=(-2, -1), keepdim=True)
        s_std = torch.clamp(s_f.std(dim=(-2, -1), keepdim=True), min=eps)
        if t_f.dim() == 4:
            t_mean = t_f.mean(dim=(0, 2, 3), keepdim=True)
            t_std = torch.clamp(t_f.std(dim=(0, 2, 3), keepdim=True), min=eps)
        else:
            t_reduce = tuple(range(t_f.dim() - 1)) if t_f.dim() > 1 else (0,)
            t_mean = t_f.mean(dim=t_reduce, keepdim=True).view(1, -1, 1, 1)
            t_std = torch.clamp(t_f.std(dim=t_reduce, keepdim=True), min=eps).view(1, -1, 1, 1)
    else:
        # Sequence token format: dim -2 (L)
        s_mean = s_f.mean(dim=-2, keepdim=True)
        s_std = torch.clamp(s_f.std(dim=-2, keepdim=True), min=eps)
        target_reduce_dims = tuple(range(t_f.dim() - 1)) if t_f.dim() > 1 else (0,)
        t_mean = t_f.mean(dim=target_reduce_dims, keepdim=True)
        t_std = torch.clamp(t_f.std(dim=target_reduce_dims, keepdim=True), min=eps)

    normalized = (s_f - s_mean) / s_std
    corrected = normalized * t_std + t_mean
    return corrected.to(orig_dtype)
