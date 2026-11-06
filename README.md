# Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers

[![TPAMI](https://img.shields.io/badge/IEEE%20TPAMI-2026-blue.svg)](https://github.com/jackeyloveseven/hyperion-style-transfer)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

Official open-source implementation of **Hyperion**, a training-free style transfer method for Multimodal Diffusion Transformers (MM-DiT).

> **Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers**  
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*  
> Yi Li, Chengyan Li, Zhirui Fang, Junchuan Zhang, Yan Luo, Yanqing Guo (Dalian University of Technology), Xuan Wang (Beihang University), Huaibo Huang (CASIA).  
> Paper Repository: [hyperion-style-transfer](https://github.com/jackeyloveseven/hyperion-style-transfer)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Method Highlights](#-method-highlights)
- [Unique Quantitative Metrics](#-unique-quantitative-metrics)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quickstart & CLI](#-quickstart--cli)
  - [Text-to-Image (T2I) Style Transfer](#1-text-to-image-t2i-style-transfer)
  - [Image-to-Image (I2I) Style Transfer](#2-image-to-image-i2i-style-transfer)
  - [Quantitative Metrics Evaluation](#3-quantitative-metrics-evaluation)
- [Python API Usage](#-python-api-usage)
- [Hyperparameters & Reproducibility](#-hyperparameters--reproducibility)
- [Citation](#-citation)

---

## 🌟 Overview

Training-free style transfer on Multimodal Diffusion Transformers (MM-DiT) has historically suffered from a fundamental performance gap compared to trained adapters (e.g. CSGO, USO, StyleShot): forward injection methods produce only faint hue shifts without rich painterly textures.

**Hyperion** resolves this gap through two complementary components that freeze all network parameters:
1. **Adam-in-loop Latent Optimization**: Propagates spatial coherence into the latent update via backpropagation through the frozen MM-DiT, recovering the gradient-driven value-path spatial coupling ($\rho = 0.64$) responsible for expressive brushstrokes.
2. **Sinkhorn Latent Transport (SLT)**: A single closed-form optimal-transport step that pre-aligns latent color and distribution statistics before Adam optimization, eliminating gradient budget waste on low-frequency color mismatch.

On the full 234-pair benchmark, Hyperion outperforms equal-budget Adam-only baselines by **+16.6% CSD** and prior training-free methods by **+5.2%**, achieving the highest style fidelity while strictly preserving content semantics.

---

## 🔬 Method Highlights

### 1. Sinkhorn Latent Transport (SLT)
Before backpropagation begins, SLT calculates an optimal transport plan $T = \text{Sinkhorn}(\tilde{S}/\tau)$ from high-frequency residual cosine similarities $\tilde{S}_{ij} = \cos(\tilde{z}_t[i], \tilde{z}_s[j])$, transferring distribution statistics via per-token **OT-AdaIN**:
$$\hat{z}_t[i] = \sigma^*_i \frac{z_t[i] - \mu_t}{\sigma_t} + \mu^*_i, \quad z_t \leftarrow (1-\alpha)z_t + \alpha \hat{z}_t$$
SLT runs in $\approx 0.1\text{ ms}$ with **zero forward passes**, instantly resolving the global color mismatch.

### 2. Frequency-Domain Sinkhorn Attention
In deep transformer blocks, the self-attention style loss decomposes queries and keys into low-frequency and high-frequency components:
$$\text{logits} = \text{logits}_\text{lo} + \gamma_\text{hi} \text{logits}_\text{hi}, \quad \gamma_\text{hi} = 2.0$$
followed by doubly-stochastic Sinkhorn normalization. This amplifies texture energy and suppresses oversmoothing.

### 3. Layout Residual Suppression (LRS)
Style key/value tensors carry both texture directionality and position-dependent spatial layout. LRS attenuates spatial residuals:
$$K_s^{(\ell)\prime} = \mu_K^{(\ell)} + \gamma^{(\ell)}(K_s^{(\ell)} - \mu_K^{(\ell)})$$
where $\gamma^{(\ell)}$ linearly scales from $0.70$ (shallow) to $0.50$ (deep), eliminating reference layout leakage.

---

## 📊 Unique Quantitative Metrics

This repository open-sources all novel evaluation metrics introduced in the TPAMI paper:

| Metric | Function / Module | Description | Direction |
|---|---|---|---|
| **STM** | `compute_style_texture_match` | **Style-Texture Match**: Pearson correlation between mean-centered orientation histograms of high-pass residuals (tracks CSD at $r=+0.98$, discriminates across trained and training-free methods). | $\uparrow$ (Higher is better) |
| **Tex. Energy** | `compute_texture_energy` | Mean absolute high-pass residual quantifying brushstroke and texture presence. | $\uparrow$ |
| **Orient. Coh.** | `compute_orientation_coherence` | Structure tensor eigenvalue ratio $\frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \epsilon}$ weighted by gradient trace, quantifying streamline coherence. | $\uparrow$ |
| **SCH** | `compute_style_content_harmonic` | **Style-Content Harmonic**: $\mathrm{SCH} = \sqrt{\mathrm{CSD} \cdot \mathrm{CLIP\text{-}I}}$, balancing style transfer and content retention. | $\uparrow$ |
| **Layout Leakage** | `compute_layout_leakage_from_patches` | DINOv2 patch self-similarity correlation diagnosing reference subject layout leakage. | $\downarrow$ (Lower is better) |
| **$\rho(d)$** | `compute_spatial_neighbor_correlation` | Update-space neighbor cosine correlation diagnosing spatial gradient coupling. | Diagnostic ($\approx 0.64$) |

---

## 📁 Repository Structure

```
hyperion/
├── hyperion/                     # Core Python Package
│   ├── core/                     # Algorithmic & mathematical primitives
│   │   ├── sinkhorn.py           # Sinkhorn-Knopp normalization & OT attention
│   │   ├── slt.py                # Sinkhorn Latent Transport (SLT, OT-AdaIN)
│   │   ├── lrs.py                # Layout Residual Suppression (LRS)
│   │   ├── loss.py               # HyperionLoss & frequency-domain loss
│   │   └── color.py              # Feature AdaIN & terminal 1-D color correction
│   ├── models/                   # Transformer integrations & hooks
│   │   ├── hooks.py              # AttentionCache & AttentionController
│   │   ├── klein9b.py            # Klein 9B (DiffSynth MM-DiT) attention hooks
│   │   └── flux.py               # FLUX.1-dev / Kontext (diffusers) hooks
│   ├── pipeline/                 # High-level pipeline API
│   │   ├── config.py             # HyperionConfig dataclass (TPAMI Table 10)
│   │   └── hyperion_pipeline.py  # HyperionPipeline (T2I & I2I inference)
│   └── metrics/                  # Open-sourced evaluation metrics
│       ├── stm.py                # Style-Texture Match (STM)
│       ├── texture.py            # Texture energy & orientation coherence
│       ├── sch.py                # Style-Content Harmonic (SCH)
│       ├── layout_leakage.py     # DINO layout leakage
│       ├── spatial_coherence.py  # Spatial neighbor correlation rho(d)
│       ├── perceptual.py         # DINO-style, Gram distance, ArtFID
│       ├── csd.py                # Contrastive Style Descriptor wrapper
│       └── clip_eval.py          # CLIP-T & CLIP-I alignment
├── scripts/                      # CLI entrypoints
│   ├── generate_t2i.py           # Standalone T2I generation
│   ├── generate_i2i.py           # Standalone I2I generation
│   ├── evaluate_metrics.py       # Comprehensive metric evaluation runner
│   └── download_models.py        # Model download assistant
├── configs/                      # Canonical JSON configurations
│   ├── tpami_t2i_default.json    # Canonical T2I settings
│   └── tpami_i2i_default.json    # Canonical I2I settings
├── tests/                        # Comprehensive unit test suite (54 tests)
├── pyproject.toml                # Packaging metadata
├── requirements.txt              # Production dependencies
└── LICENSE                       # Apache 2.0 License
```

---

## ⚙️ Installation

### 1. Clone repository & install dependencies
```bash
git clone https://github.com/jackeyloveseven/hyperion.git
cd hyperion

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Download Model Checkpoints
Hyperion operates on frozen MM-DiT backbones (primarily **FLUX.2-Klein-9B** or **FLUX.1-dev**):
```bash
# Download backbone and metric models
python scripts/download_models.py --target all
```

### 3. Run Unit Tests
Verify mathematical algorithms, metrics, attention controllers, and pipeline logic:
```bash
python -m unittest discover -s tests -v
```

---

## 🚀 Quickstart & CLI

### 1. Text-to-Image (T2I) Style Transfer
```bash
python scripts/generate_t2i.py \
    --prompt "A sports car on a mountain road" \
    --style assets/style_example.png \
    --output outputs/car_stylized.png \
    --steps 25 \
    --seed 43
```

### 2. Image-to-Image (I2I) Style Transfer
Preserves content layout via in-context token concatenation while transferring style:
```bash
python scripts/generate_i2i.py \
    --content assets/content_example.png \
    --style assets/style_example.png \
    --prompt "masterpiece" \
    --output outputs/i2i_stylized.png \
    --steps 25 \
    --seed 43
```

### 3. Quantitative Metrics Evaluation
Evaluate your generated image with Hyperion's unique metrics:
```bash
python scripts/evaluate_metrics.py \
    --output outputs/car_stylized.png \
    --style assets/style_example.png \
    --prompt "A sports car on a mountain road" \
    --save_csv results.csv
```

---

## 🐍 Python API Usage

```python
import torch
from PIL import Image
from hyperion import HyperionPipeline, HyperionConfig
from hyperion.metrics import compute_style_texture_match, compute_texture_energy

# 1. Load pipeline
pipe = HyperionPipeline.from_pretrained_klein9b("models/klein9b", device="cuda")

# 2. Configure with TPAMI defaults
config = HyperionConfig.t2i_default(
    num_inference_steps=25,
    guidance_scale=3.5,
    seed=43,
)

# 3. Generate
output = pipe.generate_t2i(
    prompt="A cat sitting on a wooden table",
    style_image="path/to/style.png",
    config=config,
)
output.save("cat_stylized.png")

# 4. Compute unique style metrics
stm_score = compute_style_texture_match(output, "path/to/style.png")
energy = compute_texture_energy(output)
print(f"Style-Texture Match (STM): {stm_score:.4f}")
print(f"Texture Energy: {energy:.4f}")
```

---

## 📋 Hyperparameters & Reproducibility

Canonical hyperparameters from **Table 10** in the TPAMI publication:

| Parameter | T2I Setting | I2I Setting | Description |
|---|---|---|---|
| Backbone | Klein 9B (FLUX.2) | Klein 9B (FLUX.2) | Frozen MM-DiT architecture |
| Denoising Steps | 25 | 25 | Flow-matching schedule |
| Guidance Scale | 3.5 | 3.5 | Classifier-free guidance |
| Active Window $t_\text{start}$ | 850 | 870 | Denoising steps with active style optimization |
| SLT $\alpha$ | 0.10 | 0.10 | Latent transport interpolation weight |
| SLT $\tau_\text{SLT}$ | 0.02 | 0.02 | Sinkhorn temperature for transport plan |
| Adam Iterations $N$ | 2 | 2 | In-loop backprop iterations per active step |
| Adam Learning Rate | 0.03 | 0.03 | Latent optimizer learning rate |
| Style Weight $w_s$ | 2.0 | 2.0 | Multi-layer attention matching weight |
| Content Weight $w_c$ | 1.5 | 1.5 | Latent anchor drift resistance weight |
| Frequency Boost $\gamma_\text{hi}$ | 2.0 | 2.0 | High-frequency residual boost coefficient |
| LRS Schedule $\gamma^{(\ell)}$ | $0.70 \to 0.50$ | $0.70 \to 0.50$ | Depth-dependent layout attenuation |
| AdaIN Warm-Start $\alpha_w$ | 0.50 | 0.30 | Per-step statistics pre-alignment |
| Terminal 1-D Correction | Enabled | Enabled | Global per-channel color alignment |

---

## 📝 Citation

If you find Hyperion or our evaluation metrics useful in your research, please cite:

```bibtex
@article{li2026bridging,
  title={Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers},
  author={Li, Yi and Li, Chengyan and Fang, Zhirui and Zhang, Junchuan and Luo, Yan and Guo, Yanqing and Wang, Xuan and Huang, Huaibo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
