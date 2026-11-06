"""
Hyperion: Sinkhorn Optimal Transport Attention
==============================================
Implements doubly-stochastic optimal transport attention and frequency-domain
Sinkhorn loss formulations described in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI)
"""

from typing import Optional
import torch
import torch.nn.functional as F


def sinkhorn_normalize(
    logits: torch.Tensor,
    n_iters: int = 6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes a doubly-stochastic transport plan via the Sinkhorn-Knopp algorithm
    in log-space for numerical stability.

    Args:
        logits: Unnormalized affinity/logit matrix of shape [..., N, M].
        n_iters: Number of alternating row/col normalization iterations.
        eps: Small epsilon for numerical stabilization.

    Returns:
        Doubly-stochastic (or row-stochastic) transport plan P of shape [..., N, M],
        where rows sum to 1.
    """
    log_p = logits
    for _ in range(n_iters):
        # Row normalization: log_p[..., i, :] = log_p[..., i, :] - logsumexp_j(log_p[..., i, j])
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        # Column normalization: log_p[..., :, j] = log_p[..., :, j] - logsumexp_i(log_p[..., i, j])
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)

    # Final row-wise softmax to guarantee exact row stochasticity (sum along cols = 1)
    p = torch.softmax(log_p, dim=-1)
    return p


def sinkhorn_attn(
    q: torch.Tensor,
    k_style: torch.Tensor,
    v_style: torch.Tensor,
    tau: float = 0.1,
    n_iters: int = 6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Sinkhorn Optimal Transport Attention.

    Standard attention (softmax over key tokens) can cause all query tokens
    to collapse onto a handful of high-salience style tokens (mode collapse).
    Sinkhorn attention applies doubly-stochastic normalization so that every style
    token is utilized and query tokens obtain a balanced distribution of style features.

    Args:
        q: Content query tensor of shape [B, H, Lc, D].
        k_style: Style key tensor of shape [B, H, Ls, D].
        v_style: Style value tensor of shape [B, H, Ls, D].
        tau: Entropic regularization temperature (lower = sharper transport).
        n_iters: Number of Sinkhorn iterations.
        eps: Small epsilon for numerical stability.

    Returns:
        OT-transported style values of shape [B, H, Lc, D].
    """
    d = q.shape[-1]
    # Compute cross-similarity logits scaled by sqrt(d) and temperature tau
    scale = (d ** 0.5) * max(tau, eps)
    logits = torch.matmul(q, k_style.transpose(-1, -2)) / scale  # [B, H, Lc, Ls]

    plan = sinkhorn_normalize(logits, n_iters=n_iters, eps=eps)
    out = torch.matmul(plan, v_style)  # [B, H, Lc, D]
    return out


def sinkhorn_attn_freq(
    q: torch.Tensor,
    k_style: torch.Tensor,
    v_style: torch.Tensor,
    tau: float = 0.1,
    n_iters: int = 6,
    hi_boost: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Frequency-Domain Sinkhorn Attention (Eq. 4 in TPAMI paper).

    Decomposes query and key tokens into spatial mean (low frequency tone/layout)
    and residual (high frequency brushstroke/texture), boosting the high-frequency
    coupling to encourage rich painterly textures.

    Args:
        q: Content query tensor of shape [B, H, Lc, D].
        k_style: Style key tensor of shape [B, H, Ls, D].
        v_style: Style value tensor of shape [B, H, Ls, D].
        tau: Entropic regularization temperature.
        n_iters: Number of Sinkhorn iterations.
        hi_boost: High-frequency boost coefficient (gamma_hi, default: 2.0).
        eps: Small epsilon for numerical stability.

    Returns:
        Frequency-boosted OT style output of shape [B, H, Lc, D].
    """
    d = q.shape[-1]
    scale = (d ** 0.5) * max(tau, eps)

    # Decompose into low-frequency (mean over sequence tokens) and high-frequency (residual)
    q_lo = q.mean(dim=2, keepdim=True).expand_as(q)
    k_lo = k_style.mean(dim=2, keepdim=True).expand_as(k_style)
    q_hi = q - q_lo
    k_hi = k_style - k_lo

    logits_lo = torch.matmul(q_lo, k_lo.transpose(-1, -2)) / scale
    logits_hi = torch.matmul(q_hi, k_hi.transpose(-1, -2)) / scale
    logits = logits_lo + hi_boost * logits_hi

    plan = sinkhorn_normalize(logits, n_iters=n_iters, eps=eps)
    out = torch.matmul(plan, v_style)
    return out
