"""
Hyperion Metrics Package
========================
Exposes all unique and standard quantitative metrics introduced in the TPAMI paper:
- Style-Texture Match (STM)
- Texture Energy & Orientation Coherence
- Style-Content Harmonic (SCH)
- DINO Layout Leakage (LL)
- Spatial Neighbour Correlation rho(d)
- Contrastive Style Descriptor (CSD)
- CLIP Semantic Alignment (CLIP-T & CLIP-I)
- Perceptual Metrics (DINO-Style, Gram Distance, ArtFID)
"""

from hyperion.metrics.stm import compute_style_texture_match, compute_orientation_histogram
from hyperion.metrics.texture import compute_texture_energy, compute_orientation_coherence
from hyperion.metrics.sch import compute_style_content_harmonic
from hyperion.metrics.layout_leakage import (
    compute_layout_leakage_from_patches,
    compute_patch_self_similarity_matrix,
    extract_dinov2_features,
)
from hyperion.metrics.spatial_coherence import compute_spatial_neighbor_correlation
from hyperion.metrics.csd import compute_csd_similarity, preprocess_for_csd, load_csd_model
from hyperion.metrics.clip_eval import compute_clip_t, compute_clip_i
from hyperion.metrics.perceptual import (
    compute_dino_style_similarity,
    extract_vgg_gram_matrices,
    compute_gram_distance,
    compute_frechet_distance,
    compute_artfid,
)

__all__ = [
    "compute_style_texture_match",
    "compute_orientation_histogram",
    "compute_texture_energy",
    "compute_orientation_coherence",
    "compute_style_content_harmonic",
    "compute_layout_leakage_from_patches",
    "compute_patch_self_similarity_matrix",
    "extract_dinov2_features",
    "compute_spatial_neighbor_correlation",
    "compute_csd_similarity",
    "load_csd_model",
    "preprocess_for_csd",
    "compute_clip_t",
    "compute_clip_i",
    "compute_dino_style_similarity",
    "extract_vgg_gram_matrices",
    "compute_gram_distance",
    "compute_frechet_distance",
    "compute_artfid",
]
