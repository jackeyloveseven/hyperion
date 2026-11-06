"""
Hyperion Metrics: Perceptual Metrics (DINO-Style, Gram Distance, ArtFID)
========================================================================
Implements supplementary perceptual metrics described in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI Appendix B & G)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def _matrix_sqrt(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Computes matrix square root with scipy if available, or numpy eigh."""
    try:
        from scipy import linalg
        covmean, _ = linalg.sqrtm(m, disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(m.shape[0]) * eps
            covmean = linalg.sqrtm(m + offset)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return covmean
    except ImportError:
        # Exact square root for symmetric positive semi-definite matrix
        sym = (m + m.T) / 2.0
        w, v = np.linalg.eigh(sym)
        w_clamped = np.sqrt(np.maximum(w, 0.0))
        return (v * w_clamped) @ v.T


def compute_dino_style_similarity(
    cls_token_out: torch.Tensor,
    cls_token_style: torch.Tensor,
) -> float:
    """
    Computes DINOv2 CLS cosine similarity between output and style reference.

    Args:
        cls_token_out: CLS embedding tensor of output image [D].
        cls_token_style: CLS embedding tensor of style image [D].

    Returns:
        Cosine similarity score in [-1, 1]. Higher is better.
    """
    sim = F.cosine_similarity(cls_token_out.view(1, -1), cls_token_style.view(1, -1))
    return float(sim.item())


def extract_vgg_gram_matrices(
    vgg_features: torch.nn.Module,
    image_tensor: torch.Tensor,
    gram_layers: Optional[Dict[int, str]] = None,
) -> List[torch.Tensor]:
    """
    Computes normalized Gram matrices across classic Gatys style layers of VGG-19.

    Args:
        vgg_features: Pretrained VGG-19 features sequential module.
        image_tensor: ImageNet-normalized tensor of shape [1, 3, H, W].
        gram_layers: Dict mapping layer index to name (default: relu1_1, 2_1, 3_1, 4_1, 5_1).

    Returns:
        List of Gram matrices, each normalized by C * H * W.
    """
    if gram_layers is None:
        gram_layers = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 20: "relu4_1", 29: "relu5_1"}

    grams = []
    h = image_tensor
    max_layer = max(gram_layers.keys())

    with torch.no_grad():
        for idx, layer in enumerate(vgg_features):
            h = layer(h)
            if idx in gram_layers:
                b, c, hh, ww = h.shape
                features = h.view(c, hh * ww)
                gram = torch.matmul(features, features.t()) / (c * hh * ww)
                grams.append(gram)
            if idx >= max_layer:
                break
    return grams


def compute_gram_distance(
    grams_out: List[torch.Tensor],
    grams_style: List[torch.Tensor],
) -> float:
    """
    Computes mean Frobenius distance across VGG-19 Gram matrices.

    Args:
        grams_out: List of Gram matrices from output image.
        grams_style: List of Gram matrices from style reference.

    Returns:
        Mean Frobenius distance. Lower is better.
    """
    dists = [torch.norm(go - gs).item() for go, gs in zip(grams_out, grams_style)]
    return float(np.mean(dists))


def compute_frechet_distance(
    act1: np.ndarray,
    act2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Computes Frechet Inception Distance between two feature activation arrays.

    Uses the mathematically exact symmetric positive semi-definite formulation
    Tr((C1 C2)^(1/2)) = Tr((C1^(1/2) C2 C1^(1/2))^(1/2)) to guarantee real, calibrated,
    and numerically stable values even without scipy installed.

    Args:
        act1: Activations of first set [N1, D].
        act2: Activations of second set [N2, D].
        eps: Small epsilon for numerical stabilization.

    Returns:
        Frechet distance.
    """
    m1, m2 = act1.mean(axis=0), act2.mean(axis=0)
    c1 = np.atleast_2d(np.cov(act1, rowvar=False))
    c2 = np.atleast_2d(np.cov(act2, rowvar=False))

    diff = m1 - m2

    try:
        from scipy import linalg
        covmean, _ = linalg.sqrtm(c1.dot(c2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        covmean_tr = np.trace(covmean)
    except ImportError:
        # Exact symmetric formulation: Tr((C1 C2)^(1/2)) = sum(sqrt(eigvals(C1^(1/2) C2 C1^(1/2))))
        w1, v1 = np.linalg.eigh((c1 + c1.T) / 2.0)
        c1_sqrt = v1 @ np.diag(np.sqrt(np.maximum(w1, 0.0))) @ v1.T
        k = c1_sqrt @ c2 @ c1_sqrt
        k_sym = (k + k.T) / 2.0
        wk, _ = np.linalg.eigh(k_sym)
        covmean_tr = np.sum(np.sqrt(np.maximum(wk, 0.0)))

    fid = float(diff.dot(diff) + np.trace(c1) + np.trace(c2) - 2.0 * covmean_tr)
    return max(0.0, fid)


def compute_artfid(
    lpips_distance: float,
    fid_style: float,
) -> float:
    """
    Computes ArtFID = (1 + LPIPS) * (1 + FID_style).

    Args:
        lpips_distance: Perceptual content distance (AlexNet LPIPS).
        fid_style: Style distribution Frechet Inception Distance.

    Returns:
        ArtFID score. Lower is better.
    """
    return float((1.0 + lpips_distance) * (1.0 + fid_style))
