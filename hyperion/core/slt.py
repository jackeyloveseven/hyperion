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
        latent: Style latent tensor of shape [B, L, D], [B, 1, L, D], or [L, D].
        variance_threshold: Cumulative energy threshold (default: 0.95).

    Returns:
        Effective rank integer r_95.
    """
    if latent.numel() == 0:
        return 0

    if latent.dim() >= 3:
        sl = latent[0].view(-1, latent.shape[-1]).float()
    else:
        sl = latent.float()

    if sl.shape[0] == 0 or sl.shape[1] == 0:
        return 0

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
    mode: Literal["ot_adain", "blend", "w2", "sw"] = "ot_adain",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sinkhorn Latent Transport (SLT, Eq. 5 in TPAMI paper).

    Pre-aligns the content latent z_t with the reference style latent z_s
    using an optimal transport plan computed on high-frequency residual cosine similarities.

    Args:
        content_lat: Current denoising latent z_t of shape [B, Lc, D] or [Lc, D].
        style_lat_noisy: Style latent noised to step t of shape [B, Ls, D] or [Ls, D].
        style_lat_clean: Clean style latent z_s of shape [B, Ls, D] or [Ls, D].
        alpha: Transport interpolation strength (default: 0.10).
        n_iters: Number of Sinkhorn iterations (default: 10).
        tau: Entropic regularization temperature (default: 0.02).
        grag_delta: Style token compression coefficient towards mean (LRS analog, default: 1.0).
        mode: Transport algorithm:
            - "ot_adain": Per-token OT-matched AdaIN statistics (canonical TPAMI paper).
            - "blend": Direct embedding interpolation via transport plan.
            - "w2": Closed-form Gaussian-Wasserstein (Monge/Knott-Smith) covariance alignment.
            - "sw": Sliced-Wasserstein 1D barycentric projection mapping.
        eps: Small epsilon for numerical stability.

    Returns:
        Transported content latent of shape matching content_lat, preserving dtype.
    """
    orig_dtype = content_lat.dtype
    orig_dim = content_lat.dim()

    # Normalize shapes to [B, L, D]
    if orig_dim == 2:
        content_lat = content_lat.unsqueeze(0)
        style_lat_noisy = style_lat_noisy.unsqueeze(0)
        style_lat_clean = style_lat_clean.unsqueeze(0)
    elif orig_dim == 4 and content_lat.shape[1] == 1:
        content_lat = content_lat.squeeze(1)
        style_lat_noisy = style_lat_noisy.squeeze(1)
        style_lat_clean = style_lat_clean.squeeze(1)

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

    # Closed-form Gaussian-W2 mapping (Monge/Knott-Smith map across token distribution)
    if mode == "w2":
        s_clean_f = style_clean_g.float()
        mu_c = c_f.mean(dim=1, keepdim=True)
        mu_s = s_clean_f.mean(dim=1, keepdim=True)
        xc = c_f - mu_c
        xs = s_clean_f - mu_s
        eye = torch.eye(d, device=c_f.device, dtype=c_f.dtype).unsqueeze(0)
        sc_cov = torch.bmm(xc.transpose(1, 2), xc) / max(xc.shape[1], 1) + 1e-4 * eye
        ss_cov = torch.bmm(xs.transpose(1, 2), xs) / max(xs.shape[1], 1) + 1e-4 * eye

        def _mpow(mx: torch.Tensor, p: float) -> torch.Tensor:
            w, v = torch.linalg.eigh((mx + mx.transpose(-1, -2)) / 2.0)
            w = w.clamp(min=1e-6).pow(p)
            return v @ torch.diag_embed(w) @ v.transpose(-1, -2)

        sc_h = _mpow(sc_cov, 0.5)
        sc_ih = _mpow(sc_cov, -0.5)
        mid_h = _mpow(torch.bmm(sc_h, torch.bmm(ss_cov, sc_h)), 0.5)
        a_mat = torch.bmm(sc_ih, torch.bmm(mid_h, sc_ih))
        transported = torch.bmm(xc, a_mat) + mu_s
        out = (1.0 - alpha) * c_f + alpha * transported
        if torch.isnan(out).any() or out.abs().max() > 1e4:
            out = c_f
        if orig_dim == 2:
            return out.squeeze(0).to(orig_dtype)
        elif orig_dim == 4:
            return out.unsqueeze(1).to(orig_dtype)
        return out.to(orig_dtype)

    # Sliced-Wasserstein barycentric map (1D random projection OT)
    if mode == "sw":
        s_clean_f = style_clean_g.float()
        p_projections = 64
        theta = F.normalize(torch.randn(p_projections, d, device=c_f.device, dtype=c_f.dtype), dim=1)
        cp = torch.einsum('bnd,pd->bnp', c_f, theta)
        sp = torch.einsum('bmd,pd->bmp', s_clean_f, theta)
        sp_sorted, _ = torch.sort(sp, dim=1)
        nn = cp.shape[1]
        mm = sp.shape[1]
        ranks = cp.argsort(dim=1).argsort(dim=1)
        idx = (ranks.float() / max(nn - 1, 1) * (mm - 1)).round().long().clamp(0, mm - 1)
        mapped = torch.gather(sp_sorted, 1, idx)
        delta = torch.einsum('bnp,pd->bnd', mapped - cp, theta) / p_projections
        out = c_f + alpha * delta
        if orig_dim == 2:
            return out.squeeze(0).to(orig_dtype)
        elif orig_dim == 4:
            return out.unsqueeze(1).to(orig_dtype)
        return out.to(orig_dtype)

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
    else:
        # Canonical TPAMI: OT-AdaIN
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

    if orig_dim == 2:
        return out.squeeze(0).to(orig_dtype)
    elif orig_dim == 4:
        return out.unsqueeze(1).to(orig_dtype)
    return out.to(orig_dtype)
