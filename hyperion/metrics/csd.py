"""
Hyperion Metric: Contrastive Style Descriptor (CSD)
===================================================
Wrapper for the Contrastive Style Descriptor (Somepalli et al., 2024)
used as the primary quantitative style fidelity metric in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI)
"""

from typing import Optional, Union
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Standard CLIP normalization constants used by CSD
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def preprocess_for_csd(
    img_or_path: Union[str, Image.Image, torch.Tensor],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Preprocesses an input image to CSD normalized tensor [1, 3, 224, 224].
    """
    if isinstance(img_or_path, torch.Tensor):
        if img_or_path.dim() == 3:
            img_or_path = img_or_path.unsqueeze(0)
        t = F.interpolate(img_or_path, size=(224, 224), mode="bilinear", align_corners=False)
        return t.to(device)

    if isinstance(img_or_path, str):
        pil_img = Image.open(img_or_path).convert("RGB")
    elif isinstance(img_or_path, Image.Image):
        pil_img = img_or_path.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img_or_path)}")

    pil_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    t = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
    mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std


def compute_csd_similarity(
    csd_model: torch.nn.Module,
    output_image: Union[str, Image.Image, torch.Tensor],
    style_image: Union[str, Image.Image, torch.Tensor],
    device: Optional[str] = None,
) -> float:
    """
    Computes CSD style embedding cosine similarity between output and style reference.

    Args:
        csd_model: Loaded CSD_CLIP model.
        output_image: Generated image (path, PIL Image, or tensor).
        style_image: Style reference image.
        device: Device to run evaluation on (defaults to csd_model device).

    Returns:
        CSD cosine similarity score (float in [-1, 1]). Higher is better.
    """
    if device is None:
        device = next(csd_model.parameters()).device

    t_out = preprocess_for_csd(output_image, device=device)
    t_sty = preprocess_for_csd(style_image, device=device)

    with torch.no_grad():
        # CSD model forward returns (content_feat, _, style_feat)
        _, _, s_out = csd_model(t_out.float())
        _, _, s_sty = csd_model(t_sty.float())

        s_out_norm = F.normalize(s_out, dim=-1)
        s_sty_norm = F.normalize(s_sty, dim=-1)
        score = (s_out_norm * s_sty_norm).sum(dim=-1).item()

    return float(score)
