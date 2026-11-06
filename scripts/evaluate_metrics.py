#!/usr/bin/env python3
"""
Hyperion Metric Evaluation CLI
==============================
Computes comprehensive quantitative metrics on stylized images:
- STM (Style-Texture Match)
- Texture Energy & Orientation Coherence
- SCH (Style-Content Harmonic)
- CSD (Contrastive Style Descriptor)
- CLIP-T (Image-Text semantic consistency)
- CLIP-I (Image-Image content preservation)
- DINO-Style similarity & DINO Layout Leakage
- Gram Matrix distance

Usage:
    python scripts/evaluate_metrics.py \
        --output path/to/output.png \
        --style path/to/style.png \
        --content path/to/content.png \
        --prompt "A panda eat bamboo" \
        --save_csv results.csv
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hyperion.metrics.stm import compute_style_texture_match
from hyperion.metrics.texture import compute_texture_energy, compute_orientation_coherence
from hyperion.metrics.sch import compute_style_content_harmonic


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Style Transfer Metrics")
    parser.add_argument("--output", type=str, required=True, help="Path to generated output image")
    parser.add_argument("--style", type=str, required=True, help="Path to reference style image")
    parser.add_argument("--content", type=str, default=None, help="Path to content image (for I2I metrics)")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (for CLIP-T)")
    parser.add_argument("--csd_weights", type=str, default=None, help="Path to CSD-CLIP checkpoint")
    parser.add_argument("--eval_dino", action="store_true", help="Evaluate DINO-style similarity and DINO layout leakage")
    parser.add_argument("--eval_gram", action="store_true", help="Evaluate VGG-19 Gram matrix distance")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--save_csv", type=str, default=None, help="Path to output CSV file")
    parser.add_argument("--save_json", type=str, default=None, help="Path to output JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    results: Dict[str, Any] = {
        "output": args.output,
        "style": args.style,
        "content": args.content,
        "prompt": args.prompt,
    }

    print("=" * 65)
    print(" Hyperion Quantitative Metrics Evaluation")
    print("=" * 65)

    # 1. Output-space texture & orientation metrics (Lightweight, CPU/GPU)
    print("\n[Evaluating output-space texture metrics]")
    stm = compute_style_texture_match(args.output, args.style)
    tex_energy = compute_texture_energy(args.output)
    orient_coh = compute_orientation_coherence(args.output)

    results["STM"] = round(stm, 4)
    results["Texture_Energy"] = round(tex_energy, 4)
    results["Orientation_Coherence"] = round(orient_coh, 4)

    print(f"  * Style-Texture Match (STM)  : {stm:.4f} (Higher = closer texture orientation)")
    print(f"  * Texture Energy             : {tex_energy:.4f}")
    print(f"  * Orientation Coherence      : {orient_coh:.4f}")

    # 2. CLIP-T & CLIP-I (if open_clip installed)
    try:
        import open_clip
        from hyperion.metrics.clip_eval import compute_clip_t, compute_clip_i

        print("\n[Evaluating CLIP metrics (OpenCLIP ViT-B/32)]")
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip_model = clip_model.to(args.device).eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

        if args.prompt:
            clip_t = compute_clip_t(clip_model, clip_tokenizer, args.output, args.prompt, device=args.device)
            results["CLIP-T"] = round(clip_t, 4)
            print(f"  * CLIP-T (Text alignment)    : {clip_t:.4f}")

        if args.content:
            clip_i = compute_clip_i(clip_model, args.output, args.content, device=args.device)
            results["CLIP-I"] = round(clip_i, 4)
            print(f"  * CLIP-I (Content retention) : {clip_i:.4f}")
    except Exception as e:
        print(f"  (Skipping CLIP metrics: {e})")

    # 3. CSD & SCH (if CSD model weights provided)
    if args.csd_weights:
        csd_path = Path(args.csd_weights)
        if csd_path.exists():
            try:
                import torch
                from hyperion.metrics.csd import compute_csd_similarity, load_csd_model

                print("\n[Evaluating CSD metric]")
                csd_model = load_csd_model(csd_path, device=args.device)

                csd = compute_csd_similarity(csd_model, args.output, args.style, device=args.device)
                results["CSD"] = round(csd, 4)
                print(f"  * CSD (Style fidelity)       : {csd:.4f}")

                if "CLIP-I" in results:
                    sch = compute_style_content_harmonic(csd, results["CLIP-I"])
                    results["SCH"] = round(sch, 4)
                    print(f"  * Style-Content Harmonic(SCH): {sch:.4f}")
            except Exception as e:
                print(f"  (Skipping CSD metric: {e})")
        else:
            print(f"  (Warning: CSD weights file not found: {csd_path})")

    # 4. DINOv2 metrics (if requested)
    if args.eval_dino:
        try:
            import timm
            import torch
            from torchvision import transforms
            from PIL import Image
            from hyperion.metrics.perceptual import compute_dino_style_similarity
            from hyperion.metrics.layout_leakage import extract_dinov2_features, compute_layout_leakage_from_patches

            print("\n[Evaluating DINOv2 metrics (vit_base_patch14_dinov2)]")
            dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).to(args.device).eval()

            def _dino_load(img_path):
                im = Image.open(img_path).convert("RGB")
                t = transforms.ToTensor()(im).unsqueeze(0).to(args.device)
                t = torch.nn.functional.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False)
                mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
                return (t - mean) / std

            cls_out, patches_out = extract_dinov2_features(dino, _dino_load(args.output))
            cls_sty, patches_sty = extract_dinov2_features(dino, _dino_load(args.style))

            dino_sim = compute_dino_style_similarity(cls_out, cls_sty)
            results["DINO_Style_Sim"] = round(dino_sim, 4)
            print(f"  * DINO-Style Similarity      : {dino_sim:.4f}")

            leakage = compute_layout_leakage_from_patches(patches_out, patches_sty)
            results["DINO_Layout_Leakage"] = round(leakage, 4)
            print(f"  * DINO Layout Leakage (LL)   : {leakage:.4f} (Lower = less leakage)")
        except Exception as e:
            print(f"  (Skipping DINO metrics: {e})")

    # 5. VGG-19 Gram distance (if requested)
    if args.eval_gram:
        try:
            import torch
            from torchvision.models import vgg19, VGG19_Weights
            from torchvision import transforms
            from PIL import Image
            from hyperion.metrics.perceptual import extract_vgg_gram_matrices, compute_gram_distance

            print("\n[Evaluating VGG-19 Gram distance]")
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(args.device).eval()
            for prm in vgg.parameters():
                prm.requires_grad_(False)

            def _vgg_load(img_path):
                im = Image.open(img_path).convert("RGB")
                t = transforms.ToTensor()(im).unsqueeze(0).to(args.device)
                t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
                mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
                return (t - mean) / std

            grams_out = extract_vgg_gram_matrices(vgg, _vgg_load(args.output))
            grams_sty = extract_vgg_gram_matrices(vgg, _vgg_load(args.style))
            gram_dist = compute_gram_distance(grams_out, grams_sty)
            results["Gram_Distance"] = round(gram_dist, 4)
            print(f"  * VGG-19 Gram Distance       : {gram_dist:.4f} (Lower = closer style)")
        except Exception as e:
            print(f"  (Skipping Gram metric: {e})")

    # Summary Output
    print("\n" + "=" * 65)
    print(" Evaluation Summary:")
    for k, v in results.items():
        if k not in ("output", "style", "content", "prompt"):
            print(f"  {k:<26}: {v}")
    print("=" * 65)

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
        print(f"Saved CSV results to: {csv_path}")

    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON results to: {json_path}")


if __name__ == "__main__":
    main()
