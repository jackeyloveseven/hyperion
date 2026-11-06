"""
Hyperion Metric: DINO Layout Leakage
====================================
Implements the DINO layout-leakage metric introduced in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI Sec. 3.5 & Appendix B)

Evaluates whether the reference style image's spatial layout has leaked into the generated image
via Pearson correlation of DINOv2 patch self-similarity matrices.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_patch_self_similarity_matrix(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    Computes spatial self-similarity matrix of patch tokens.

    Args:
        patch_tokens: Patch embedding tensor of shape [N, D] or [B, N, D].

    Returns:
        Cosine similarity matrix of shape [N, N] (if 2D or B=1) or [B, N, N].
    """
    p_norm = F.normalize(patch_tokens, dim=-1)
    sim = torch.matmul(p_norm, p_norm.transpose(-1, -2))
    if sim.dim() == 3 and sim.shape[0] == 1:
        sim = sim.squeeze(0)
    return sim


def compute_layout_leakage_from_patches(
    patches_out: torch.Tensor,
    patches_style: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Computes Pearson correlation between off-diagonal self-similarity entries.

    Args:
        patches_out: Output image DINO patch tokens [N, D] or [1, N, D].
        patches_style: Style reference DINO patch tokens [N, D] or [1, N, D].
        eps: Small epsilon for numerical stability.

    Returns:
        Layout leakage correlation score in [-1, 1]. Lower means less layout leakage.
    """
    s_out = compute_patch_self_similarity_matrix(patches_out)
    s_sty = compute_patch_self_similarity_matrix(patches_style)

    if s_out.dim() == 3:
        s_out = s_out[0]
    if s_sty.dim() == 3:
        s_sty = s_sty[0]

    n_out = s_out.shape[0]
    n_sty = s_sty.shape[0]
    if n_out <= 1 or n_sty <= 1:
        return 0.0

    if s_out.shape != s_sty.shape:
        target_n = max(n_out, n_sty)
        if s_out.shape[0] != target_n:
            s_out = F.interpolate(
                s_out.unsqueeze(0).unsqueeze(0),
                size=(target_n, target_n),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        if s_sty.shape[0] != target_n:
            s_sty = F.interpolate(
                s_sty.unsqueeze(0).unsqueeze(0),
                size=(target_n, target_n),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

    n = s_out.shape[0]
    tri = torch.triu_indices(n, n, offset=1, device=s_out.device)

    a = s_out[tri[0], tri[1]].float()
    b = s_sty[tri[0], tri[1]].float()

    a_c = a - a.mean()
    b_c = b - b.mean()

    denom = (a_c.norm() * b_c.norm()).clamp_min(eps)
    corr = (a_c * b_c).sum() / denom
    return float(corr.item())


def extract_dinov2_features(
    dino_model: torch.nn.Module,
    image_tensor: torch.Tensor,
    num_prefix_tokens: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts CLS token and spatial patch tokens from DINOv2.

    Args:
        dino_model: Timm or torch DINOv2 model (e.g. vit_base_patch14_dinov2).
        image_tensor: Normalized image tensor of shape [1, 3, H, W].
        num_prefix_tokens: Number of prefix tokens (CLS=1).

    Returns:
        Tuple of (cls_token [D], patch_tokens [N, D]).
    """
    with torch.no_grad():
        tok = dino_model.forward_features(image_tensor)
        cls_tok = tok[:, 0, :].squeeze(0)
        patch_tok = tok[:, num_prefix_tokens:, :].squeeze(0)
    return cls_tok, patch_tok
