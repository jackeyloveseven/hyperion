"""
Hyperion Metrics: Texture Energy & Orientation Coherence
=========================================================
Implements output-space texture metrics introduced in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI Sec. 4.3)
"""

from typing import Union
import numpy as np
from PIL import Image
from hyperion.metrics.stm import _to_grayscale, _uniform_filter, _sobel


def compute_texture_energy(
    img: Union[str, Image.Image, np.ndarray],
    blur_kernel: int = 9,
    size: int = 256,
) -> float:
    """
    Computes output texture energy (mean absolute high-pass residual).

    Higher values indicate stronger texture presence (brushstrokes, impasto, grain).

    Args:
        img: Input image (path, PIL Image, or numpy array).
        blur_kernel: Kernel size for uniform low-pass filter (default: 9).
        size: Target image size for evaluation (default: 256).

    Returns:
        Texture energy score (float >= 0).
    """
    g = _to_grayscale(img, size=size)
    low_pass = _uniform_filter(g, size=blur_kernel)
    high_pass = g - low_pass
    return float(np.abs(high_pass).mean())


def compute_orientation_coherence(
    img: Union[str, Image.Image, np.ndarray],
    patch_size: int = 16,
    blur_kernel: int = 9,
    size: int = 256,
    eps: float = 1e-8,
) -> float:
    """
    Computes local orientation coherence via structure tensor eigenvalues.

    Measures whether brushstrokes and textures are coherently oriented along local
    streamlines rather than isotropic noise.

    Args:
        img: Input image.
        patch_size: Size of local spatial patches P (default: 16).
        blur_kernel: Low-pass filter size (default: 9).
        size: Evaluation image size (default: 256).
        eps: Small epsilon for numerical stability.

    Returns:
        Orientation coherence score in [0, 1].
    """
    g = _to_grayscale(img, size=size)
    hp = g - _uniform_filter(g, size=blur_kernel)

    gx = _sobel(hp, axis=1)
    gy = _sobel(hp, axis=0)

    h, w = g.shape
    p = patch_size
    n_h = h // p
    n_w = w // p

    coherence_list = []
    weight_list = []

    for i in range(n_h):
        for j in range(n_w):
            sx = gx[i * p : (i + 1) * p, j * p : (j + 1) * p]
            sy = gy[i * p : (i + 1) * p, j * p : (j + 1) * p]

            jxx = np.sum(sx * sx)
            jyy = np.sum(sy * sy)
            jxy = np.sum(sx * sy)

            tr = jxx + jyy
            det = jxx * jyy - (jxy * jxy)

            # Eigenvalues of 2x2 structure tensor
            rad = np.sqrt(max(0.0, (tr * tr) / 4.0 - det))
            l1 = tr / 2.0 + rad
            l2 = tr / 2.0 - rad

            coh = (l1 - l2) / (l1 + l2 + eps)
            coherence_list.append(coh)
            weight_list.append(tr)

    weights = np.array(weight_list, dtype=np.float64)
    coherences = np.array(coherence_list, dtype=np.float64)

    total_weight = weights.sum()
    if total_weight < eps:
        return 0.0

    weighted_coh = float((coherences * weights).sum() / total_weight)
    return float(np.clip(weighted_coh, 0.0, 1.0))
