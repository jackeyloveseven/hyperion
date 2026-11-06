"""
Hyperion: Sinkhorn Latent Transport (SLT)
=========================================
Implements closed-form optimal-transport latent pre-alignment described in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI)

SLT performs a single closed-form optimal transport step in latent token space
to pre-align color and texture distributions before Adam-in-loop optimization,
eliminating the early gradient budget waste on low-frequency color mismatch.
"""

from typing import Literal, Optional
import numpy as np
import torch
import torch.nn.functional as F
from hyperion.core.sinkhorn import sinkhorn_normalize


def compute_latent_effective_rank(
    latent: torch.Tensor,
    variance_threshold: float = 0.95,
) -> int:
    """
    Computes the effective rank of a latent token distribution (cumulative covariance energy).

    Used as the pre-screening filter disclosed in TPAMI Sec. 6 (Discussion) to detect
    degenerate, near-uniform style references (e.g. sty04 with effective rank < 5)
    before running optimal transport.

    Args:
        latent: Style latent tensor of shape [B, L, D] or [L, D].
        variance_threshold: Cumulative energy threshold (default: 0.95).

    Returns:
        Effective rank integer r_95.
    """
    if latent.dim() == 3:
        sl = latent[0].float()
    else:
        sl = latent.float()

    sl_c = sl - sl.mean(dim=0, keepdim=True)
    l_toks, d = sl_c.shape
    c = sl_c.T @ sl_c / max(l_toks, 1)
    c = (c + c.T) / 2.0
    s = torch.linalg.eigvalsh(c + 1e-8 * torch.eye(d, device=sl.device))
    s = torch.clamp(s, min=0.0).flip(0)
    total_var = s.sum().item()
    if total_var < 1e-8:
        return 0
    p = (s / total_var).cpu().numpy()
    cum = np.cumsum(p)
    return int(np.searchsorted(cum, variance_threshold) + 1)


def sinkhorn_latent_transport(
    content_lat: torch.Tensor,
    style_lat_noisy: torch.Tensor,
    style_lat_clean: torch.Tensor,
    alpha: float = 0.10,
    n_iters: int = 10,
    tau: float = 0.02,
    grag_delta: float = 1.0,
    mode: Literal["ot_adain", "blend"] = "ot_adain",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sinkhorn Latent Transport (SLT, Eq. 5 in TPAMI paper).

    Pre-aligns the content latent z_t with the reference style latent z_s
    using an optimal transport plan computed on high-frequency residual cosine similarities.

    Args:
        content_lat: Current denoising latent z_t of shape [B, Lc, D].
        style_lat_noisy: Style latent noised to step t of shape [B, Ls, D] (for matching).
        style_lat_clean: Clean style latent z_s of shape [B, Ls, D] (for statistics target).
        alpha: Transport interpolation strength (default: 0.10).
        n_iters: Number of Sinkhorn iterations (default: 10).
        tau: Entropic regularization temperature (default: 0.02).
        grag_delta: Style token compression coefficient towards mean (LRS analog, default: 1.0).
        mode: "ot_adain" (per-token OT-matched AdaIN statistics, recommended) or "blend".
        eps: Small epsilon for numerical stability.

    Returns:
        Transported content latent of shape [B, Lc, D], preserving content dtype.
    """
    orig_dtype = content_lat.dtype
    b, lc, d = content_lat.shape

    # Apply GRAG suppression on style latent tokens if grag_delta < 1.0
    style_noisy_g = style_lat_noisy
    style_clean_g = style_lat_clean
    if grag_delta < 1.0:
        s_mean = style_lat_noisy.mean(dim=1, keepdim=True)
        style_noisy_g = s_mean + grag_delta * (style_lat_noisy - s_mean)
        sc_mean = style_lat_clean.mean(dim=1, keepdim=True)
        style_clean_g = sc_mean + grag_delta * (style_lat_clean - sc_mean)

    # Convert to float32 for stable OT and statistics computation
    c_f = content_lat.float()
    s_f = style_noisy_g.float()

    # High-frequency residual (spatial mean subtracted) to match by texture rather than absolute position
    c_hi = c_f - c_f.mean(dim=1, keepdim=True)
    s_hi = s_f - s_f.mean(dim=1, keepdim=True)

    c_norm = F.normalize(c_hi, dim=-1)
    s_norm = F.normalize(s_hi, dim=-1)

    # Pairwise cosine similarity matrix scaled by tau
    logits = torch.bmm(c_norm, s_norm.transpose(1, 2)) / max(tau, eps)  # [B, Lc, Ls]

    # Compute doubly-stochastic transport plan T
    transport_plan = sinkhorn_normalize(logits, n_iters=n_iters, eps=eps)  # [B, Lc, Ls]

    if mode == "blend":
        # Direct embedding interpolation
        target = torch.bmm(transport_plan, style_clean_g.float())
        out = (1.0 - alpha) * c_f + alpha * target
        return out.to(orig_dtype)

    # Canonical TPAMI: OT-AdaIN
    # Match each content token to the OT-weighted style distribution statistics
    style_target = style_clean_g.float()
    matched_mean = torch.bmm(transport_plan, style_target)  # [B, Lc, D]
    matched_second_moment = torch.bmm(transport_plan, style_target ** 2)
    matched_var = matched_second_moment - (matched_mean ** 2)
    matched_std = torch.clamp(matched_var, min=eps).sqrt()

    # Content global statistics
    c_mean = c_f.mean(dim=1, keepdim=True)
    c_std = torch.clamp(c_f.std(dim=1, keepdim=True), min=eps)

    # Normalize content and rescale with OT-matched style statistics
    normed = (c_f - c_mean) / c_std
    transported = normed * matched_std + matched_mean

    out = (1.0 - alpha) * c_f + alpha * transported
    return out.to(orig_dtype)
