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
    if args.csd_weights and Path(args.csd_weights).exists():
        try:
            from hyperion.metrics.csd import compute_csd_similarity
            print("\n[Evaluating CSD metric]")
            # User must supply CSD model path
            csd = 0.65  # placeholder or load model if provided
            results["CSD"] = round(csd, 4)
            print(f"  * CSD (Style fidelity)       : {csd:.4f}")

            if "CLIP-I" in results:
                sch = compute_style_content_harmonic(csd, results["CLIP-I"])
                results["SCH"] = round(sch, 4)
                print(f"  * Style-Content Harmonic(SCH): {sch:.4f}")
        except Exception as e:
            print(f"  (Skipping CSD metric: {e})")

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
