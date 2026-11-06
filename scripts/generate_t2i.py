#!/usr/bin/env python3
"""
Hyperion CLI: Text-to-Image Style Transfer
==========================================
Usage:
    python scripts/generate_t2i.py \
        --prompt "A panda eat bamboo" \
        --style path/to/style.png \
        --output output_t2i.png \
        --model_root path/to/klein9b
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure hyperion package is on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hyperion.pipeline.config import HyperionConfig
from hyperion.pipeline.hyperion_pipeline import HyperionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperion T2I Style Transfer")
    parser.add_argument("--prompt", type=str, required=True, help="Content text prompt")
    parser.add_argument("--style", type=str, required=True, help="Path to style reference image")
    parser.add_argument("--output", type=str, default="output_hyperion_t2i.png", help="Output image path")
    parser.add_argument("--model_root", type=str, default="models/klein9b", help="Klein 9B weights directory")
    parser.add_argument("--steps", type=int, default=25, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (e.g. cuda, cpu)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom JSON configuration file")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(" Hyperion: Training-Free Style Transfer on MM-DiT (T2I)")
    print("=" * 65)
    print(f"Prompt     : {args.prompt}")
    print(f"Style Image: {args.style}")
    print(f"Model Root : {args.model_root}")
    print(f"Steps      : {args.steps}")
    print(f"Seed       : {args.seed}")

    if args.config:
        config = HyperionConfig.from_json(args.config)
        config.num_inference_steps = args.steps
        config.seed = args.seed
    else:
        config = HyperionConfig.t2i_default(num_inference_steps=args.steps, seed=args.seed)

    print("\n[1/2] Loading MM-DiT model...")
    pipe = HyperionPipeline.from_pretrained_klein9b(
        model_root=args.model_root,
        device=args.device,
    )

    print("\n[2/2] Generating stylized image...")
    start_time = time.time()
    result = pipe.generate_t2i(
        prompt=args.prompt,
        style_image=args.style,
        config=config,
    )
    elapsed = time.time() - start_time

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"\n[Done] Output saved to: {out_path} (Elapsed: {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
