"""
Hyperion: Optimization Losses
==============================
Implements style, content anchor, and distribution alignment losses for Adam-in-loop
described in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI)
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperion.core.sinkhorn import sinkhorn_attn, sinkhorn_attn_freq


class HyperionLoss(nn.Module):
    """
    Combined Hyperion loss module for Adam-in-loop latent optimization.
    Computes style-alignment loss across MM-DiT layers + content anchor drift resistance.
    """

    def __init__(
        self,
        style_weight: float = 2.0,
        content_weight: float = 1.5,
        freq_hi_boost: float = 2.0,
        ot_layer_frac: float = 0.5,
        ot_iters: int = 6,
    ):
        super().__init__()
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.freq_hi_boost = freq_hi_boost
        self.ot_layer_frac = ot_layer_frac
        self.ot_iters = ot_iters
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        latents: torch.Tensor,
        content_anchor: torch.Tensor,
        q_list: List[torch.Tensor],
        self_out_list: List[torch.Tensor],
        ref_k_list: List[torch.Tensor],
        ref_v_list: List[torch.Tensor],
        tau: float = 0.1,
    ) -> torch.Tensor:
        """
        Computes the combined Adam-in-loop loss (Eq. 2 & Eq. 4 in TPAMI paper).

        Args:
            latents: Current latent tensor z_t being optimized (requires_grad=True).
            content_anchor: Latent state before Adam steps (anchor to prevent semantic drift).
            q_list: Cached query tensors from content forward pass.
            self_out_list: Cached attention outputs from content forward pass.
            ref_k_list: LRS-suppressed style key tensors.
            ref_v_list: LRS-suppressed style value tensors.
            tau: Annealed Sinkhorn temperature for the current timestep.

        Returns:
            Scalar loss tensor with attached gradient computation graph.
        """
        n_layers = len(q_list)
        n_ot_layers = max(1, int(n_layers * self.ot_layer_frac))
        cutoff = n_layers - n_ot_layers

        style_loss = torch.tensor(0.0, device=latents.device, dtype=torch.float32)

        for idx, (q, so, ref_k, ref_v) in enumerate(zip(q_list, self_out_list, ref_k_list, ref_v_list)):
            q_det = q.detach()
            # Deep layers: Frequency-Domain Sinkhorn Attention
            if idx >= cutoff:
                ref_self_out = sinkhorn_attn_freq(
                    q_det,
                    ref_k,
                    ref_v,
                    tau=tau,
                    n_iters=self.ot_iters,
                    hi_boost=self.freq_hi_boost,
                )
            else:
                # Shallow layers: Standard scaled dot-product attention
                ref_self_out = F.scaled_dot_product_attention(q_det, ref_k, ref_v)

            layer_loss = self.l1_loss(so, ref_self_out.detach())
            style_loss = style_loss + layer_loss

        # Content anchor loss: L1 distance to prevent spatial structure drift
        content_loss = self.l1_loss(latents, content_anchor)

        total_loss = self.style_weight * style_loss + self.content_weight * content_loss
        return total_loss


def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 64,
) -> torch.Tensor:
    """
    1D-projected Sliced Wasserstein distance between two token distributions.
    Differentiable with respect to x.

    Args:
        x: Source token tensor of shape [B, Lx, D].
        y: Target token tensor of shape [B, Ly, D].
        n_projections: Number of random 1D projections on the unit sphere.

    Returns:
        Scalar Wasserstein distance.
    """
    b, lx, d = x.shape
    x_f = x.float()
    y_f = y.float()

    theta = torch.randn(n_projections, d, device=x.device, dtype=torch.float32)
    theta = theta / (theta.norm(dim=1, keepdim=True) + 1e-8)

    x_proj = torch.matmul(x_f, theta.T)
    y_proj = torch.matmul(y_f, theta.T)

    x_sorted = torch.sort(x_proj, dim=1).values
    y_sorted = torch.sort(y_proj, dim=1).values

    return (x_sorted - y_sorted).abs().mean()
