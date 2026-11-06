"""
Hyperion: Layout Residual Suppression (LRS)
===========================================
Implements depth-dependent key/value residual attenuation described in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI)

LRS prevents reference image layout leakage into the generated content while
preserving essential texture directionality cues across MM-DiT transformer layers.
"""

from typing import List, Tuple
import torch


def apply_layout_residual_suppression(
    ref_k_list: List[torch.Tensor],
    ref_v_list: List[torch.Tensor],
    txt_len: int = 0,
    gamma_shallow: float = 0.70,
    gamma_deep: float = 0.50,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Applies Layout Residual Suppression (LRS, Eq. 6 in TPAMI paper) to reference key and value tensors.

    Decomposes each style key/value tensor into a token mean mu (global style statistics)
    and a position-dependent residual R (spatial layout). The residual is attenuated
    by gamma^(l), which linearly decreases from gamma_shallow (top layers) to gamma_deep (deep layers).

    Args:
        ref_k_list: List of style key tensors across layers, each of shape [B, H, L, D].
        ref_v_list: List of style value tensors across layers, each of shape [B, H, L, D].
        txt_len: Length of text prefix tokens (which are preserved without attenuation).
        gamma_shallow: Attenuation coefficient for shallow layers (default: 0.70).
        gamma_deep: Attenuation coefficient for deep layers (default: 0.50).

    Returns:
        Tuple of (suppressed_k_list, suppressed_v_list).
    """
    n_layers = len(ref_k_list)
    out_k_list = []
    out_v_list = []

    for idx in range(n_layers):
        depth_frac = idx / max(n_layers - 1, 1)
        gamma_l = gamma_shallow + (gamma_deep - gamma_shallow) * depth_frac

        k = ref_k_list[idx]
        v = ref_v_list[idx]

        # Sequence dimension is dim -2 for both [B, H, L, D] and [B, L, D]
        seq_dim = -2
        seq_len = k.shape[seq_dim]

        if txt_len > 0 and seq_len > txt_len:
            # Separate text prefix and image tokens
            k_txt = k.narrow(seq_dim, 0, txt_len)
            k_img = k.narrow(seq_dim, txt_len, seq_len - txt_len)
            v_txt = v.narrow(seq_dim, 0, txt_len)
            v_img = v.narrow(seq_dim, txt_len, seq_len - txt_len)

            # Spatial mean over image tokens along sequence dimension
            k_mean = k_img.mean(dim=seq_dim, keepdim=True)
            v_mean = v_img.mean(dim=seq_dim, keepdim=True)

            # Attenuate position-dependent layout residual
            k_img_suppressed = k_mean + gamma_l * (k_img - k_mean)
            v_img_suppressed = v_mean + gamma_l * (v_img - v_mean)

            out_k = torch.cat([k_txt, k_img_suppressed], dim=seq_dim)
            out_v = torch.cat([v_txt, v_img_suppressed], dim=seq_dim)
        else:
            k_mean = k.mean(dim=seq_dim, keepdim=True)
            v_mean = v.mean(dim=seq_dim, keepdim=True)
            out_k = k_mean + gamma_l * (k - k_mean)
            out_v = v_mean + gamma_l * (v - v_mean)

        out_k_list.append(out_k)
        out_v_list.append(out_v)

    return out_k_list, out_v_list
