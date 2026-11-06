# Hyperion: 基于扩散 Transformer 的免训练风格迁移方法

[![TPAMI](https://img.shields.io/badge/IEEE%20TPAMI-2026-blue.svg)](https://github.com/jackeyloveseven/hyperion-style-transfer)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

[English](README.md) | [中文说明](README_CN.md)

本仓库是发表于 **IEEE TPAMI** 的论文 **《Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers》** 的官方开源实现。

> **论文题目**：Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers  
> **发表期刊**：IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)  
> **作者单位**：大连理工大学、北京航空航天大学、中国科学院自动化研究所 (CASIA)  
> **论文仓库**：[hyperion-style-transfer](https://github.com/jackeyloveseven/hyperion-style-transfer)

---

## 🌟 核心方法简介

在多模态扩散 Transformer（MM-DiT）架构上，传统的免训练（Training-Free）风格迁移主要采用前向前馈特征注入，容易出现仅能改变浅层色调、无法生成真实画笔笔触纹理（brushstrokes）的问题；而已有的训练式适配器（如 CSGO、USO、StyleShot）虽然具有较强风格，但需要昂贵的微调训练，且存在内容崩塌的风险。

**Hyperion** 提出了一套**全程冻结主干网络参数（全程不微调任何权重）**的免训练风格迁移算法：
1. **Adam-in-loop 潜空间梯度优化**：在去噪过程中通过冻结的 MM-DiT 进行反向传播优化 latent，恢复了自注意力 Value 路径的空间耦合能力（邻域相关度 $\rho = 0.64$），从而生成连贯、真实的笔触与纹理。
2. **Sinkhorn Latent Transport (SLT) 预对齐**：在每步去噪前，利用高频残差余弦相似度构建闭式最优传输（OT）矩阵，通过 OT-AdaIN 一次性对齐 latent 颜色与多阶统计量（耗时仅约 $0.1\text{ ms}$，**0 次额外前向计算**），彻底免除早期 Adam 迭代浪费在低频色彩校正上的梯度预算。
3. **频域 Sinkhorn 损失（Frequency-Domain Sinkhorn Loss）**：在深层注意力的风格损失中引入高频残差增强（$\gamma_\text{hi} = 2.0$），显著强化笔触线条细节。
4. **布局残差抑制（Layout Residual Suppression, LRS）**：沿网络深度对风格参考图的 Key/Value 进行渐进衰减（$\gamma^{(\ell)}: 0.70 \to 0.50$），阻断风格图构图布局对生成主体的泄露（Anti-Layout-Leakage）。

---

## 📊 本文开源的独特量化指标体系

除了核心算法外，本仓库完整开源了论文提出的全部独特创新量化评估指标：

| 指标名称 | Python 函数 | 作用与物理含义 | 优劣方向 |
|---|---|---|---|
| **STM** | `compute_style_texture_match` | **风格纹理匹配度（Style-Texture Match）**：计算输出图与风格参考图高频残差方向梯度的去中心化直方图余弦相关性。与 CSD 皮尔逊相关系数高达 $+0.98$，能精准区分不同方法的笔触方向一致性。 | $\uparrow$ 越大越好 |
| **Tex. Energy** | `compute_texture_energy` | **纹理能量**：高频残差绝对均值，衡量笔触、厚涂与颗粒感的丰富度。 | $\uparrow$ 越大越好 |
| **Orient. Coh.** | `compute_orientation_coherence` | **方向相干性**：通过结构张量特征值比率衡量笔触是否沿流线方向连贯排列。 | $\uparrow$ 越大越好 |
| **SCH** | `compute_style_content_harmonic` | **风格-内容调和均值（Style-Content Harmonic）**：$\mathrm{SCH} = \sqrt{\mathrm{CSD} \cdot \mathrm{CLIP\text{-}I}}$，平衡风格化强度与内容保真度。 | $\uparrow$ 越大越好 |
| **Layout Leakage** | `compute_layout_leakage_from_patches` | **DINO 布局泄露率**：DINOv2 patch 空间自相似性皮尔逊相关系数，量化风格图构图是否对内容造成污染。 | $\downarrow$ 越小越好 |
| **$\rho(d)$** | `compute_spatial_neighbor_correlation` | **潜空间更新相干度**：衡量梯度在相邻 token 间的耦合强度（证明 Adam-in-loop 笔触机理的关键指标）。 | 诊断参考（$\approx 0.64$） |

---

## ⚙️ 环境依赖与安装

```bash
git clone https://github.com/jackeyloveseven/hyperion.git
cd hyperion

python -m venv .venv
source .venv/bin/activate  # Windows 用户运行 .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. 权重下载
```bash
python scripts/download_models.py --target all
```

### 3. 执行单元测试验证
运行完整单元测试集（54 项单测，涵盖核心算法、注意力控制器与全部论文指标）：
```bash
python -m unittest discover -s tests -v
```

---

## 🚀 命令行快速上手

### 1. 文生图风格迁移（Text-to-Image / T2I）
```bash
python scripts/generate_t2i.py \
    --prompt "A sports car on a mountain road" \
    --style assets/style_example.png \
    --output outputs/car_stylized.png \
    --steps 25 \
    --seed 43
```

### 2. 图生图风格迁移（Image-to-Image / I2I）
通过 In-Context Token Concatenation 完整保留内容图结构与主体姿态：
```bash
python scripts/generate_i2i.py \
    --content assets/content_example.png \
    --style assets/style_example.png \
    --prompt "masterpiece" \
    --output outputs/i2i_stylized.png \
    --steps 25 \
    --seed 43
```

### 3. 指标计算与评估
使用 Hyperion 独创的 STM、SCH、Tex. Energy 等指标对生成图片打分：
```bash
python scripts/evaluate_metrics.py \
    --output outputs/car_stylized.png \
    --style assets/style_example.png \
    --prompt "A sports car on a mountain road" \
    --save_csv results.csv
```

---

## 📋 论文官方超参规范（TPAMI Table 10）

| 超参数 | T2I 设定 | I2I 设定 | 含义与设计考量 |
|---|---|---|---|
| 主干模型（Backbone） | Klein 9B (FLUX.2) | Klein 9B (FLUX.2) | 冻结 MM-DiT 主干 |
| 去噪步数（Steps） | 25 | 25 | Flow-matching schedule |
| Guidance Scale | 3.5 | 3.5 | Classifier-free guidance |
| 激活时间步 $t_\text{start}$ | 850 | 870 | 高噪声阶段仅做标准去噪，低于该值启动风格优化 |
| SLT 强度 $\alpha$ | 0.10 | 0.10 | 闭式最优传输统计对齐权重 |
| SLT 温度 $\tau_\text{SLT}$ | 0.02 | 0.02 | 传输计划 Sinkhorn 退火温度 |
| Adam 迭代数 $N$ | 2 | 2 | 每步去噪内的反向传播优化次数 |
| Adam 学习率 | 0.03 | 0.03 | 潜变量更新步长 |
| 风格损失权重 $w_s$ | 2.0 | 2.0 | 多层自注意力匹配损失权重 |
| 内容锚点权重 $w_c$ | 1.5 | 1.5 | 抑制结构漂移的 L1 锚点权重 |
| 频域增强系数 $\gamma_\text{hi}$ | 2.0 | 2.0 | 强化高频残差纹理的权重 |
| LRS 衰减系数 $\gamma^{(\ell)}$ | $0.70 \to 0.50$ | $0.70 \to 0.50$ | 浅层到深层渐进抑制构图泄露 |
| AdaIN Warm-Start $\alpha_w$ | 0.50 | 0.30 | 每步进入 SLT 前的统计预对齐 |
| 终端 1-D 色彩校正 | 开启 | 开启 | 去噪结束后对 $z_0$ 进行单次全局通道对齐 |

---

## 📝 引用

```bibtex
@article{li2026bridging,
  title={Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers},
  author={Li, Yi and Li, Chengyan and Fang, Zhirui and Zhang, Junchuan and Luo, Yan and Guo, Yanqing and Wang, Xuan and Huang, Huaibo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2026}
}
```

---

## 📄 开源许可证

本项目基于 [Apache 2.0](LICENSE) 许可证开源。
