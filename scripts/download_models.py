#!/usr/bin/env python3
"""
Hyperion Model Download Utility
===============================
Downloads backbone checkpoints and evaluation models from Hugging Face:
- Klein 9B (FLUX.2-klein-9b) via DiffSynth-Studio / Hugging Face
- DINOv2 & OpenCLIP models

Usage:
    python scripts/download_models.py --target core
    python scripts/download_models.py --target metrics
    python scripts/download_models.py --target all
"""

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Download models for Hyperion")
    parser.add_argument(
        "--target",
        type=str,
        default="core",
        choices=["core", "metrics", "all"],
        help="Model group to download (core: Klein9B; metrics: DINOv2/CLIP; all: everything)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="models",
        help="Destination directory for model weights",
    )
    return parser.parse_args()


def download_hf_repo(repo_id: str, local_dir: Path):
    print(f"\n[Downloading] {repo_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        print(f"[Success] Downloaded {repo_id}")
    except ImportError:
        print("[Notice] huggingface_hub not installed. Using huggingface-cli:")
        subprocess.run(["huggingface-cli", "download", repo_id, "--local-dir", str(local_dir)], check=True)


def main():
    args = parse_args()
    dest = Path(args.dest)

    if args.target in ("core", "all"):
        print("\n=== Downloading Core Backbone Weights ===")
        print("Note: FLUX.2-klein-9B requires accepting license agreement on Hugging Face.")
        klein_dir = dest / "klein9b"
        download_hf_repo("black-forest-labs/FLUX.2-klein-9B", klein_dir)

    if args.target in ("metrics", "all"):
        print("\n=== Downloading Metric Models ===")
        # DINOv2 and OpenCLIP automatically cache via timm / torch hub / open_clip
        try:
            import timm
            print("Pre-fetching DINOv2 (vit_base_patch14_dinov2)...")
            timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
            print("[Success] DINOv2 cached.")
        except Exception as e:
            print(f"[Warning] Could not pre-fetch DINOv2: {e}")

        try:
            import open_clip
            print("Pre-fetching OpenCLIP ViT-B/32...")
            open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            print("[Success] OpenCLIP cached.")
        except Exception as e:
            print(f"[Warning] Could not pre-fetch OpenCLIP: {e}")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
