"""
Hyperion Pipeline: Configuration Dataclass
==========================================
Encapsulates all hyperparameters defined in Table 10 of:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI)
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json


@dataclass
class HyperionConfig:
    """
    Hyperion Style Transfer Pipeline Configuration.

    Defaults correspond to canonical settings from Table 10 in the TPAMI paper.
    """

    # Generation dimensions & scheduler
    width: int = 512
    height: int = 512
    num_inference_steps: int = 25
    guidance_scale: float = 3.5
    seed: int = 43

    # Denoising start threshold: only optimize when timestep t < start_time
    start_time: int = 850  # 850 for T2I, 870 for I2I

    # Active attention layers to hook (double/single stream blocks)
    self_layers: Tuple[int, int] = (12, 28)

    # Sinkhorn Latent Transport (SLT pre-alignment)
    slt_enabled: bool = True
    slt_alpha: float = 0.10
    slt_tau: float = 0.02
    slt_iters: int = 10
    slt_mode: str = "ot_adain"  # "ot_adain" or "blend"

    # Adam-in-loop latent optimization
    adam_enabled: bool = True
    adam_iters: int = 2
    adam_lr: float = 0.03
    style_weight: float = 2.0
    content_weight: float = 1.5

    # Frequency-domain style loss
    freq_hi_boost: float = 2.0
    ot_layer_frac: float = 0.5  # bottom 50% layers use Sinkhorn OT attention
    ot_tau_max: float = 0.15
    ot_tau_min: float = 0.07
    ot_iters: int = 6

    # Layout Residual Suppression (LRS)
    lrs_enabled: bool = True
    grag_shallow: float = 0.70
    grag_deep: float = 0.50

    # Multi-shot style K/V sampling
    kshot: int = 1

    # AdaIN warm-start
    warm_adain_enabled: bool = True
    adain_warm_alpha: float = 0.50  # 0.50 for T2I, 0.30 for I2I

    # Terminal 1-D color correction (Algorithm 1, line 89)
    terminal_color_correction: bool = True

    # Mode: "hybrid" (SLT + Adamx2), "adam_only", "slt_only", "inject", or "vshuffle"
    mode: str = "hybrid"
    inject_beta: float = 0.50  # Blending weight for attention injection baseline (beta=0.50 in TPAMI Sec. 4.1)

    @classmethod
    def t2i_default(cls, **overrides: Any) -> "HyperionConfig":
        """Returns official TPAMI T2I configuration."""
        cfg = cls(
            start_time=850,
            adain_warm_alpha=0.50,
            adam_iters=2,
            slt_alpha=0.10,
            mode="hybrid",
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def i2i_default(cls, **overrides: Any) -> "HyperionConfig":
        """Returns official TPAMI I2I configuration."""
        cfg = cls(
            start_time=870,
            adain_warm_alpha=0.30,
            adam_iters=2,
            slt_alpha=0.10,
            mode="hybrid",
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary."""
        d = asdict(self)
        d["self_layers"] = list(self.self_layers)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperionConfig":
        """Builds configuration from dictionary."""
        data_copy = data.copy()
        if "self_layers" in data_copy and isinstance(data_copy["self_layers"], list):
            data_copy["self_layers"] = tuple(data_copy["self_layers"])
        return cls(**data_copy)

    def save_json(self, path: Union[str, Path]) -> None:
        """Saves configuration to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "HyperionConfig":
        """Loads configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
