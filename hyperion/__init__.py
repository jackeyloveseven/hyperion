"""
Hyperion: Training-Free Style Transfer on Diffusion Transformers
================================================================
Official implementation for the IEEE TPAMI paper:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers"

Authors:
Yi Li, Chengyan Li, Zhirui Fang, Junchuan Zhang, Yan Luo, Yanqing Guo, Xuan Wang, Huaibo Huang
"""

__version__ = "1.0.0"

from hyperion.pipeline.config import HyperionConfig
from hyperion.pipeline.hyperion_pipeline import HyperionPipeline
from hyperion.core import (
    sinkhorn_normalize,
    sinkhorn_attn,
    sinkhorn_attn_freq,
    sinkhorn_latent_transport,
    apply_layout_residual_suppression,
    HyperionLoss,
    feature_adain,
    adain_1d,
)
from hyperion.metrics import (
    compute_style_texture_match,
    compute_orientation_histogram,
    compute_texture_energy,
    compute_orientation_coherence,
    compute_style_content_harmonic,
    compute_layout_leakage_from_patches,
    compute_spatial_neighbor_correlation,
    compute_csd_similarity,
    compute_clip_t,
    compute_clip_i,
    compute_dino_style_similarity,
    compute_gram_distance,
    compute_artfid,
)

__all__ = [
    "__version__",
    "HyperionConfig",
    "HyperionPipeline",
    "sinkhorn_normalize",
    "sinkhorn_attn",
    "sinkhorn_attn_freq",
    "sinkhorn_latent_transport",
    "apply_layout_residual_suppression",
    "HyperionLoss",
    "feature_adain",
    "adain_1d",
    "compute_style_texture_match",
    "compute_orientation_histogram",
    "compute_texture_energy",
    "compute_orientation_coherence",
    "compute_style_content_harmonic",
    "compute_layout_leakage_from_patches",
    "compute_spatial_neighbor_correlation",
    "compute_csd_similarity",
    "compute_clip_t",
    "compute_clip_i",
    "compute_dino_style_similarity",
    "compute_gram_distance",
    "compute_artfid",
]
