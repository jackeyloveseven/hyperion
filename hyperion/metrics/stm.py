"""
Hyperion Metric: Style-Texture Match (STM)
===========================================
Implements the Style-Texture Match (STM) metric introduced in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI Sec. 4.3)

STM measures the correlation between the orientation histogram of an output's
high-pass residual and that of its style reference — computed as a cosine
similarity after mean-centering both histograms. This specifically diagnoses whether
the generated image reproduces the style's orientation profile rather than generic textures.
"""

from typing import Union
import numpy as np
from PIL import Image


def _uniform_filter(img: np.ndarray, size: int = 9) -> np.ndarray:
    """Computes 2D uniform box filter with scipy if available, or pure numpy."""
    try:
        import scipy.ndimage as ndi
        return ndi.uniform_filter(img, size=size)
    except ImportError:
        pad_before = (size - 1) // 2
        pad_after = size // 2
        padded = np.pad(img, ((pad_before, pad_after), (pad_before, pad_after)), mode="reflect")
        kernel = np.ones(size, dtype=np.float32) / size
        # Separable 1D filtering across rows then cols
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=padded)
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=tmp)
        return out


def _sobel(img: np.ndarray, axis: int) -> np.ndarray:
    """Computes Sobel gradient along specified axis (1=dx, 0=dy)."""
    try:
        import scipy.ndimage as ndi
        return ndi.sobel(img, axis=axis)
    except ImportError:
        if axis == 1:
            k_row = np.array([1, 0, -1], dtype=np.float32)
            k_col = np.array([1, 2, 1], dtype=np.float32)
        else:
            k_row = np.array([1, 2, 1], dtype=np.float32)
            k_col = np.array([1, 0, -1], dtype=np.float32)

        padded = np.pad(img, 1, mode="reflect")
        tmp = np.apply_along_axis(lambda m: np.convolve(m, k_row, mode="valid"), axis=1, arr=padded)
        out = np.apply_along_axis(lambda m: np.convolve(m, k_col, mode="valid"), axis=0, arr=tmp)
        return out


def _to_grayscale(img: Union[str, Image.Image, np.ndarray], size: int = 256) -> np.ndarray:
    """Converts image to float32 grayscale array in [0, 1]."""
    if isinstance(img, str):
        pil_img = Image.open(img).convert("L")
    elif isinstance(img, Image.Image):
        pil_img = img.convert("L")
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)).convert("L")
        else:
            pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    if size is not None:
        pil_img = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil_img, dtype=np.float32) / 255.0


def compute_orientation_histogram(
    gray: np.ndarray,
    num_bins: int = 36,
    blur_kernel: int = 9,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Computes magnitude-weighted orientation histogram of high-pass residual.

    Args:
        gray: Grayscale 2D numpy array in [0, 1].
        num_bins: Number of angular bins spanning [0, pi).
        blur_kernel: Uniform filter kernel size to extract high-pass residual.
        eps: Small epsilon for numerical stability.

    Returns:
        1D numpy array of shape [num_bins], normalized to unit sum.
    """
    low_pass = _uniform_filter(gray, size=blur_kernel)
    high_pass = gray - low_pass

    # Compute spatial gradients
    gx = _sobel(high_pass, axis=1)
    gy = _sobel(high_pass, axis=0)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # Orientation angle in [0, pi)
    angle = np.arctan2(gy, gx) % np.pi

    # Bin indices
    bin_width = np.pi / num_bins
    bin_indices = np.clip((angle / bin_width).astype(int), 0, num_bins - 1)

    hist = np.zeros(num_bins, dtype=np.float64)
    np.add.at(hist, bin_indices, magnitude)

    total_weight = hist.sum()
    if total_weight > eps:
        hist = hist / total_weight
    return hist


def compute_style_texture_match(
    output_img: Union[str, Image.Image, np.ndarray],
    style_img: Union[str, Image.Image, np.ndarray],
    num_bins: int = 36,
    blur_kernel: int = 9,
    size: int = 256,
    eps: float = 1e-8,
) -> float:
    """
    Computes Style-Texture Match (STM).

    STM = <h_out - mean(h_out), h_sty - mean(h_sty)> / (||centered_h_out|| * ||centered_h_sty||)

    Args:
        output_img: Output image (path, PIL Image, or numpy array).
        style_img: Reference style image.
        num_bins: Number of orientation bins (default: 36).
        blur_kernel: Kernel size for high-pass extraction (default: 9).
        size: Resize resolution for standard evaluation (default: 256).
        eps: Small epsilon for numerical stability.

    Returns:
        STM score in [-1.0, 1.0]. Higher is better.
    """
    g_out = _to_grayscale(output_img, size=size)
    g_sty = _to_grayscale(style_img, size=size)

    h_out = compute_orientation_histogram(g_out, num_bins=num_bins, blur_kernel=blur_kernel, eps=eps)
    h_sty = compute_orientation_histogram(g_sty, num_bins=num_bins, blur_kernel=blur_kernel, eps=eps)

    # Mean-center both histograms to remove dominant shared axis
    h_out_c = h_out - h_out.mean()
    h_sty_c = h_sty - h_sty.mean()

    norm_out = np.linalg.norm(h_out_c)
    norm_sty = np.linalg.norm(h_sty_c)

    if norm_out < eps or norm_sty < eps:
        return 0.0

    stm = float(np.dot(h_out_c, h_sty_c) / (norm_out * norm_sty))
    return float(np.clip(stm, -1.0, 1.0))
