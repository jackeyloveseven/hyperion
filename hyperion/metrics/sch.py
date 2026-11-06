"""
Hyperion Metric: Style-Content Harmonic (SCH)
==============================================
Implements the Style-Content Harmonic (SCH) metric introduced in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI Sec. 4.4)

SCH is the geometric mean of style fidelity (CSD) and content preservation (CLIP-I),
penalizing degenerate behavior at either extreme (e.g. methods that fail to stylize
or methods that completely erase content structure).
"""

import math
from typing import Union
import numpy as np


def compute_style_content_harmonic(
    csd: Union[float, np.ndarray],
    clip_i: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Computes Style-Content Harmonic: SCH = sqrt(max(0, CSD) * max(0, CLIP-I)).

    Args:
        csd: Contrastive Style Descriptor score (typically in [0, 1]).
        clip_i: Content image cosine similarity (typically in [0, 1]).

    Returns:
        SCH score in [0, 1]. Higher is better.
    """
    if isinstance(csd, np.ndarray) or isinstance(clip_i, np.ndarray):
        csd_clamped = np.clip(np.asarray(csd, dtype=float), 0.0, 1.0)
        clip_clamped = np.clip(np.asarray(clip_i, dtype=float), 0.0, 1.0)
        return np.sqrt(csd_clamped * clip_clamped)

    csd_val = max(0.0, float(csd))
    clip_val = max(0.0, float(clip_i))
    return float(math.sqrt(csd_val * clip_val))
