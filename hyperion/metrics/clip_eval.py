"""
Hyperion Metric: CLIP Semantic Alignment (CLIP-T & CLIP-I)
===========================================================
Evaluates text-prompt semantic consistency (CLIP-T) and content image preservation (CLIP-I)
using OpenCLIP ViT-B/32, as reported in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI Sec. 4.1)
"""

from typing import Any, Optional, Union
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _preprocess_clip_image(img_or_path: Union[str, Image.Image, torch.Tensor], device: str) -> torch.Tensor:
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    elif isinstance(img_or_path, Image.Image):
        img = img_or_path.convert("RGB")
    elif isinstance(img_or_path, torch.Tensor):
        if img_or_path.dim() == 3:
            img_or_path = img_or_path.unsqueeze(0)
        t = F.interpolate(img_or_path, size=(224, 224), mode="bicubic", align_corners=False)
        mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)
        return (t - mean) / std
    else:
        raise TypeError(f"Unsupported image type: {type(img_or_path)}")

    preproc = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])
    return preproc(img).unsqueeze(0).to(device)


def compute_clip_t(
    clip_model: torch.nn.Module,
    clip_tokenizer: Any,
    output_image: Union[str, Image.Image, torch.Tensor],
    text_prompt: str,
    device: Optional[str] = None,
) -> float:
    """
    Computes CLIP image-text cosine similarity (CLIP-T) between output and prompt.

    Args:
        clip_model: Loaded OpenCLIP model (e.g. ViT-B/32).
        clip_tokenizer: OpenCLIP tokenizer for the model.
        output_image: Generated image.
        text_prompt: Content text prompt.
        device: Device to run evaluation on.

    Returns:
        CLIP-T cosine similarity score. Higher is better.
    """
    if device is None:
        device = next(clip_model.parameters()).device

    img_tensor = _preprocess_clip_image(output_image, device=device)
    text_tokens = clip_tokenizer([text_prompt]).to(device)

    with torch.no_grad():
        img_features = clip_model.encode_image(img_tensor)
        txt_features = clip_model.encode_text(text_tokens)

        img_norm = F.normalize(img_features, dim=-1)
        txt_norm = F.normalize(txt_features, dim=-1)
        sim = (img_norm * txt_norm).sum(dim=-1).item()

    return float(sim)


def compute_clip_i(
    clip_model: torch.nn.Module,
    output_image: Union[str, Image.Image, torch.Tensor],
    content_image: Union[str, Image.Image, torch.Tensor],
    device: Optional[str] = None,
) -> float:
    """
    Computes CLIP image-image cosine similarity (CLIP-I) between output and content image.

    Args:
        clip_model: Loaded OpenCLIP model (e.g. ViT-B/32).
        output_image: Generated stylized image.
        content_image: Reference content image.
        device: Device to run evaluation on.

    Returns:
        CLIP-I cosine similarity score. Higher is better.
    """
    if device is None:
        device = next(clip_model.parameters()).device

    t_out = _preprocess_clip_image(output_image, device=device)
    t_cnt = _preprocess_clip_image(content_image, device=device)

    with torch.no_grad():
        feat_out = clip_model.encode_image(t_out)
        feat_cnt = clip_model.encode_image(t_cnt)

        out_norm = F.normalize(feat_out, dim=-1)
        cnt_norm = F.normalize(feat_cnt, dim=-1)
        sim = (out_norm * cnt_norm).sum(dim=-1).item()

    return float(sim)
