"""
Hyperion Pipeline: End-to-End Style Transfer
=============================================
Implements the full Hyperion style-transfer pipeline described in:
"Bridging the Training Gap with Hyperion: A Training-Free Style Transfer Method on Diffusion Transformers" (IEEE TPAMI Algorithm 1)

Features:
- Closed-form Sinkhorn Latent Transport (SLT pre-alignment)
- Multi-step Adam-in-loop gradient latent optimization
- Frequency-domain Sinkhorn style loss with texture boosting (gamma_hi=2.0)
- Depth-dependent Layout Residual Suppression (LRS)
- In-context content image token conditioning for Image-to-Image (I2I)
- Per-step AdaIN warm-start & terminal 1-D color correction
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from hyperion.core.color import adain_1d, feature_adain
from hyperion.core.loss import HyperionLoss
from hyperion.core.lrs import apply_layout_residual_suppression
from hyperion.core.slt import sinkhorn_latent_transport
from hyperion.models.hooks import AttentionCache, AttentionController
from hyperion.models.klein9b import register_klein9b_attention_hooks
from hyperion.pipeline.config import HyperionConfig


def _pack_latents(t: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, H*W, C]"""
    b, c, h, w = t.shape
    return t.permute(0, 2, 3, 1).reshape(b, h * w, c)


def _unpack_latents(t: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
    """[B, H*W, C] -> [B, C, H, W]"""
    b, hw, c = t.shape
    return t.reshape(b, lat_h, lat_w, c).permute(0, 3, 1, 2)


class HyperionPipeline:
    """
    Hyperion Style Transfer Pipeline for Multimodal Diffusion Transformers.
    """

    def __init__(
        self,
        pipe: Any,
        backbone_type: str = "klein9b",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.pipe = pipe
        self.backbone_type = backbone_type
        self.device = device
        self.torch_dtype = torch_dtype

        self.cache = AttentionCache()
        self.controller = AttentionController()

        # Register attention hooks on the backbone transformer
        if backbone_type == "klein9b" and hasattr(pipe, "dit"):
            register_klein9b_attention_hooks(pipe.dit, self.controller, self.cache)
            pipe.dit.requires_grad_(False)
        elif hasattr(pipe, "transformer"):
            from hyperion.models.flux import register_flux_attention_hooks
            register_flux_attention_hooks(pipe.transformer, self.controller, self.cache)
            pipe.transformer.requires_grad_(False)

    @classmethod
    def from_pretrained_klein9b(
        cls,
        model_root: Union[str, Path],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "HyperionPipeline":
        """
        Loads DiffSynth-Studio Flux2ImagePipeline (Klein 9B MM-DiT).

        Args:
            model_root: Directory containing Klein 9B checkpoints (text_encoder, transformer, vae, tokenizer).
            device: Target device ("cuda").
            torch_dtype: Precision (torch.bfloat16).
        """
        import glob
        from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

        def _sorted_files(pattern: str) -> List[str]:
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No checkpoint files matched: {pattern}")
            return files

        root = Path(model_root)
        pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=[
                ModelConfig(path=_sorted_files(str(root / "text_encoder" / "*.safetensors"))),
                ModelConfig(path=_sorted_files(str(root / "transformer" / "*.safetensors"))),
                ModelConfig(path=str(root / "vae" / "diffusion_pytorch_model.safetensors")),
            ],
            tokenizer_config=ModelConfig(path=str(root / "tokenizer")),
        )
        pipe.vram_management_enabled = False
        return cls(pipe=pipe, backbone_type="klein9b", device=device, torch_dtype=torch_dtype)

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        pipe = self.pipe
        if hasattr(pipe, "encode_prompt"):
            return pipe.encode_prompt(prompt)

        try:
            if hasattr(pipe, "text_encoder_qwen3") and pipe.text_encoder_qwen3 is not None:
                from diffsynth.pipelines.flux2_image import Flux2Unit_Qwen3PromptEmbedder
                pipe.load_models_to_device(["text_encoder_qwen3"])
                embedder = Flux2Unit_Qwen3PromptEmbedder()
                embeds, ids = embedder.encode_prompt(
                    pipe.text_encoder_qwen3, pipe.tokenizer, prompt,
                    dtype=self.torch_dtype, device=self.device
                )
                return embeds, ids
            elif hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                from diffsynth.pipelines.flux2_image import Flux2Unit_PromptEmbedder
                pipe.load_models_to_device(["text_encoder"])
                embedder = Flux2Unit_PromptEmbedder()
                embeds, ids = embedder.encode_prompt(
                    pipe.text_encoder, pipe.tokenizer, prompt,
                    dtype=self.torch_dtype, device=self.device
                )
                return embeds, ids
        except (ImportError, Exception):
            pass

        # Fallback embeddings for lightweight mock pipelines or environments without text_encoder
        embeds = torch.zeros(1, 16, 4096, dtype=self.torch_dtype, device=self.device)
        ids = torch.zeros(1, 16, 3, dtype=torch.long, device=self.device)
        return embeds, ids

    def _load_image(self, img_or_path: Union[str, Path, Image.Image], size: Tuple[int, int]) -> Image.Image:
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(str(img_or_path)).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img_or_path)}")
        return img.resize(size, Image.Resampling.BICUBIC)

    def generate_t2i(
        self,
        prompt: str,
        style_image: Union[str, Path, Image.Image],
        config: Optional[HyperionConfig] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Performs Text-to-Image (T2I) style transfer using Hyperion.

        Args:
            prompt: Text describing content semantics.
            style_image: Reference style image.
            config: HyperionConfig instance (defaults to HyperionConfig.t2i_default()).
            seed: Random seed for noise generation.

        Returns:
            Stylized output image as PIL Image.
        """
        if config is None:
            config = HyperionConfig.t2i_default()
        if seed is not None:
            config.seed = seed

        return self._run_hyperion(
            prompt=prompt,
            style_image=style_image,
            content_image=None,
            config=config,
            is_i2i=False,
        )

    def generate_i2i(
        self,
        content_image: Union[str, Path, Image.Image],
        style_image: Union[str, Path, Image.Image],
        prompt: str = "masterpiece",
        config: Optional[HyperionConfig] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Performs Image-to-Image (I2I) style transfer using Hyperion.

        Preserves the layout and semantic structure of content_image via in-context token
        concatenation while transferring painterly textures and palette of style_image.

        Args:
            content_image: Reference content image.
            style_image: Reference style image.
            prompt: Optional guiding text prompt.
            config: HyperionConfig instance (defaults to HyperionConfig.i2i_default()).
            seed: Random seed.

        Returns:
            Stylized output image as PIL Image.
        """
        if config is None:
            config = HyperionConfig.i2i_default()
        if seed is not None:
            config.seed = seed

        return self._run_hyperion(
            prompt=prompt,
            style_image=style_image,
            content_image=content_image,
            config=config,
            is_i2i=True,
        )

    def _run_hyperion(
        self,
        prompt: str,
        style_image: Union[str, Path, Image.Image],
        content_image: Optional[Union[str, Path, Image.Image]],
        config: HyperionConfig,
        is_i2i: bool = False,
    ) -> Image.Image:
        pipe = self.pipe
        device = self.device
        dtype = self.torch_dtype

        # 1. Prepare images & VAE encodings
        sty_pil = self._load_image(style_image, (config.width, config.height))
        pipe.load_models_to_device(["vae"])

        with torch.no_grad():
            sty_pix = pipe.preprocess_image(sty_pil)
            style_latent_4d = pipe.vae.encode(sty_pix)
            style_latent = _pack_latents(style_latent_4d).to(dtype=dtype, device=device)

            content_latent_packed = None
            if is_i2i and content_image is not None:
                cnt_pil = self._load_image(content_image, (config.width, config.height))
                cnt_pix = pipe.preprocess_image(cnt_pil)
                content_latent_4d = pipe.vae.encode(cnt_pix)
                content_latent_packed = _pack_latents(content_latent_4d).to(dtype=dtype, device=device)

        # 2. Setup positional token coordinate IDs
        lat_h, lat_w = config.height // 16, config.width // 16
        image_ids = torch.cartesian_prod(
            torch.arange(1), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)
        ).unsqueeze(0).to(device)

        style_image_ids = torch.cartesian_prod(
            torch.arange(1), torch.arange(style_latent_4d.shape[2]), torch.arange(style_latent_4d.shape[3]), torch.arange(1)
        ).unsqueeze(0).to(device)

        if is_i2i and content_latent_packed is not None:
            cnt_image_ids = torch.cartesian_prod(
                torch.tensor([10]), torch.arange(lat_h), torch.arange(lat_w), torch.arange(1)
            ).unsqueeze(0).to(device)
            denoise_img_ids = torch.cat([image_ids, cnt_image_ids], dim=1)
        else:
            denoise_img_ids = image_ids

        # 3. Text prompt embeddings
        prompt_embeds, text_ids = self._encode_prompt(prompt)
        null_embeds, null_text_ids = self._encode_prompt("")

        # 4. Initialize random latent noise
        torch.manual_seed(config.seed)
        noise = torch.randn(1, 128, lat_h, lat_w, device=device, dtype=dtype)
        latents = _pack_latents(noise)

        # 5. Flow-matching scheduler
        try:
            from diffsynth.diffusion.flow_match import FlowMatchScheduler
            scheduler = FlowMatchScheduler("FLUX.2")
        except (ImportError, Exception):
            from hyperion.pipeline.scheduler import FlowMatchScheduler
            scheduler = FlowMatchScheduler(shift=3.0)

        scheduler.set_timesteps(config.num_inference_steps, dynamic_shift_len=lat_h * lat_w)
        timesteps = scheduler.timesteps

        # Loss function for Adam optimization
        loss_module = HyperionLoss(
            style_weight=config.style_weight,
            content_weight=config.content_weight,
            freq_hi_boost=config.freq_hi_boost,
            ot_layer_frac=config.ot_layer_frac,
            ot_iters=config.ot_iters,
        )

        pipe.load_models_to_device(["dit"])
        dit = pipe.dit
        dit.to(device)

        latent_seq_len = latents.shape[1]
        guidance_tensor = (
            torch.tensor([config.guidance_scale], device=device, dtype=dtype)
            if (config.guidance_scale is not None and config.guidance_scale > 1.0)
            else None
        )

        # 6. Denoising & Optimization Loop (Algorithm 1)
        for i, t_val in enumerate(timesteps):
            timestep = t_val.unsqueeze(0).to(dtype=dtype, device=device)
            self.cache.clear()
            self.controller.step()

            # --- Injection / V-Shuffle baseline branch (TPAMI Sec. 3.2 & 4.1) ---
            if config.mode in ("inject", "vshuffle") and t_val < config.start_time:
                if i < config.num_inference_steps - 1:
                    timestep_style = timesteps[i + 1 : i + 2]
                    sty_noise_inj = torch.randn_like(style_latent)
                    style_latent_inj = scheduler.add_noise(style_latent, sty_noise_inj, timestep_style)
                else:
                    timestep_style = torch.tensor([0.0])
                    style_latent_inj = style_latent

                with torch.no_grad():
                    _ = dit(
                        hidden_states=style_latent_inj.to(device=device, dtype=dtype),
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=null_embeds,
                        txt_ids=null_text_ids,
                        img_ids=style_image_ids,
                    )
                _, inj_k_list, inj_v_list, _ = self.cache.get()

                if config.mode == "vshuffle":
                    self.controller.set_vshuffle(inj_k_list, inj_v_list, beta=config.inject_beta)
                else:
                    self.controller.set_inject(inj_k_list, inj_v_list, beta=config.inject_beta)

                self.cache.clear()
                self.controller.step()
                with torch.no_grad():
                    noise_pred_styled = dit(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance_tensor,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=denoise_img_ids,
                    )
                    if is_i2i:
                        noise_pred_styled = noise_pred_styled[:, :latent_seq_len]

                latents = scheduler.step(noise_pred_styled, t_val, latents)
                self.controller.reset_mode()
                continue

            # --- Denoise step: standard forward pass ---
            with torch.no_grad():
                if is_i2i and content_latent_packed is not None:
                    model_input = torch.cat([latents, content_latent_packed], dim=1)
                else:
                    model_input = latents

                noise_pred = dit(
                    hidden_states=model_input,
                    timestep=timestep / 1000,
                    guidance=guidance_tensor,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=denoise_img_ids,
                )
                if is_i2i:
                    noise_pred = noise_pred[:, :latent_seq_len]

            latents = scheduler.step(noise_pred, t_val, latents)

            # High-noise steps (t >= start_time): only standard denoising
            if t_val >= config.start_time:
                continue

            # --- Sample noisy style latent ---
            if i < config.num_inference_steps - 1:
                timestep_style = timesteps[i + 1 : i + 2]
                sty_noise = torch.randn_like(style_latent)
                style_latent_noised = scheduler.add_noise(style_latent, sty_noise, timestep_style)
            else:
                timestep_style = torch.tensor([0.0])
                style_latent_noised = style_latent

            # --- Style forward pass -> extract reference K/V ---
            self.cache.clear()
            self.controller.step()
            with torch.no_grad():
                _ = dit(
                    hidden_states=style_latent_noised.to(device=device, dtype=dtype),
                    timestep=timestep_style.to(device=device, dtype=dtype) / 1000,
                    guidance=None,
                    encoder_hidden_states=null_embeds,
                    txt_ids=null_text_ids,
                    img_ids=style_image_ids,
                )
            _, ref_k_list, ref_v_list, _ = self.cache.get()

            # K-shot averaging if requested (manifold expectation)
            if config.kshot > 1 and i < config.num_inference_steps - 1:
                for _ in range(1, config.kshot):
                    aug_noise = torch.randn_like(style_latent)
                    aug_latent = scheduler.add_noise(style_latent, aug_noise, timestep_style)
                    self.cache.clear()
                    self.controller.step()
                    with torch.no_grad():
                        _ = dit(
                            hidden_states=aug_latent.to(device=device, dtype=dtype),
                            timestep=timestep_style.to(device=device, dtype=dtype) / 1000,
                            guidance=None,
                            encoder_hidden_states=null_embeds,
                            txt_ids=null_text_ids,
                            img_ids=style_image_ids,
                        )
                    _, k_aug, v_aug, _ = self.cache.get()
                    for idx in range(len(ref_k_list)):
                        ref_k_list[idx] = ref_k_list[idx] + k_aug[idx]
                        ref_v_list[idx] = ref_v_list[idx] + v_aug[idx]
                for idx in range(len(ref_k_list)):
                    ref_k_list[idx] = ref_k_list[idx] / config.kshot
                    ref_v_list[idx] = ref_v_list[idx] / config.kshot

            # --- Layout Residual Suppression (LRS) ---
            if config.lrs_enabled:
                ref_k_list, ref_v_list = apply_layout_residual_suppression(
                    ref_k_list=ref_k_list,
                    ref_v_list=ref_v_list,
                    txt_len=self.controller.txt_len,
                    gamma_shallow=config.grag_shallow,
                    gamma_deep=config.grag_deep,
                )

            # --- AdaIN warm-start ---
            if config.warm_adain_enabled:
                with torch.no_grad():
                    latents = feature_adain(
                        latents.unsqueeze(1),
                        style_latent_noised.unsqueeze(1),
                        alpha=config.adain_warm_alpha,
                    ).squeeze(1).detach()

            content_anchor = latents.detach().clone()

            # --- Sinkhorn Latent Transport (SLT pre-alignment) ---
            if config.slt_enabled and config.mode in ("hybrid", "slt_only") and config.slt_alpha > 0:
                with torch.no_grad():
                    latents = sinkhorn_latent_transport(
                        content_lat=latents,
                        style_lat_noisy=style_latent_noised,
                        style_lat_clean=style_latent,
                        alpha=config.slt_alpha,
                        n_iters=config.slt_iters,
                        tau=config.slt_tau,
                        grag_delta=config.grag_deep,
                        mode=config.slt_mode,
                    ).detach()

            if config.mode == "slt_only":
                continue

            # --- Adam-in-loop Latent Optimization ---
            if config.adam_enabled and config.adam_iters > 0 and config.mode in ("hybrid", "adam_only"):
                latents.requires_grad_(True)
                optimizer = torch.optim.Adam([latents], lr=config.adam_lr, betas=(0.9, 0.999), weight_decay=0.0)
                ts_cuda = timestep_style.to(device=device, dtype=dtype) / 1000

                # Anneal Sinkhorn temperature across the active window
                t_float = float(t_val.item())
                progress = 1.0 - (t_float / max(config.start_time, 1.0))
                tau = config.ot_tau_max - (config.ot_tau_max - config.ot_tau_min) * progress

                for _ in range(config.adam_iters):
                    self.cache.clear()
                    self.controller.step()
                    optimizer.zero_grad()

                    with torch.enable_grad():
                        _ = dit(
                            hidden_states=latents,
                            timestep=ts_cuda.detach(),
                            guidance=None,
                            encoder_hidden_states=null_embeds.detach(),
                            txt_ids=null_text_ids.detach(),
                            img_ids=image_ids.detach(),
                        )
                    q_list, _, _, self_out_list = self.cache.get()

                    loss = loss_module(
                        latents=latents,
                        content_anchor=content_anchor,
                        q_list=q_list,
                        self_out_list=self_out_list,
                        ref_k_list=ref_k_list,
                        ref_v_list=ref_v_list,
                        tau=tau,
                    )
                    loss.backward()
                    optimizer.step()

                latents = latents.detach()

        # 7. Terminal Color Correction (Algorithm 1, line 89)
        if config.terminal_color_correction and config.mode not in ("inject", "vshuffle"):
            latents = adain_1d(latents, style_latent)

        # 8. VAE Decode
        pipe.load_models_to_device(["vae"])
        latents_4d = _unpack_latents(latents, lat_h, lat_w)
        with torch.no_grad():
            image = pipe.vae.decode(latents_4d)
            output_pil = pipe.vae_output_to_image(image)

        pipe.load_models_to_device(["dit"])
        return output_pil
