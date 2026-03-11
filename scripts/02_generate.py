#!/usr/bin/env python3
"""Batch image generation from fine-tuned checkpoints.

Generates images across all four prompt categories for detection analysis.

Usage:
    python scripts/02_generate.py --checkpoint checkpoints/phase1/100pct-seed42/checkpoint-500 \
        --artist-name "Artist Name" --output-dir results/phase1/100pct-seed42/step-0500
    python scripts/02_generate.py --config configs/phase1_100pct.yaml --all-checkpoints \
        --artist-name "Artist Name"
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import wandb
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config, load_env, load_prompts, replace_artist_name
from utils.sync import push_generated


def generate_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    artist_name: str,
    gen_seed: int = 77,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    log_to_wandb: bool = True,
):
    """Generate images from a checkpoint across all prompt categories."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load pipeline: base model + fine-tuned UNet
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt_path = Path(checkpoint_path)
    unet_path = ckpt_path / "unet"
    if unet_path.exists():
        # UNet-only checkpoint — load base model and swap in fine-tuned UNet
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
    else:
        # Legacy full-pipeline checkpoint
        pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator("cuda").manual_seed(gen_seed)
    all_metadata = []
    sample_images = []

    # Pre-load all prompts to get total count for the progress bar
    all_cat_prompts = []
    for cat in ["a", "b", "c", "d"]:
        prompts = load_prompts(cat)
        if cat == "a":
            prompts = replace_artist_name(prompts, artist_name)
        all_cat_prompts.append((cat, prompts))

    total_prompts = sum(len(prompts) for _, prompts in all_cat_prompts)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Generating images", total=total_prompts)

        for cat, prompts in all_cat_prompts:
            cat_dir = out / f"cat_{cat}"
            cat_dir.mkdir(parents=True, exist_ok=True)

            for idx, prompt in enumerate(prompts):
                progress.update(task, description=f"Generating images [cat {cat.upper()}]")
                image = pipe(
                    prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

                fname = f"{idx+1:04d}_s{gen_seed}.png"
                image.save(cat_dir / fname)

                all_metadata.append({
                    "file_name": f"cat_{cat}/{fname}",
                    "prompt_category": cat,
                    "prompt_idx": idx + 1,
                    "gen_seed": gen_seed,
                    "prompt": prompt,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                })

                # Collect samples for wandb (first 4 per category)
                if idx < 4 and log_to_wandb:
                    sample_images.append(wandb.Image(image, caption=f"[{cat.upper()}] {prompt}"))

                progress.advance(task)

    # Save metadata
    with open(out / "generation_metadata.jsonl", "w") as f:
        for m in all_metadata:
            f.write(json.dumps(m) + "\n")

    # Log samples to wandb
    if log_to_wandb and sample_images:
        wandb.log({"generated_samples": sample_images})

    del pipe
    torch.cuda.empty_cache()

    total = sum(1 for _ in out.rglob("*.png"))
    print(f"  Generated {total} images -> {out}")
    return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Batch image generation")
    parser.add_argument("--checkpoint", help="Single checkpoint path")
    parser.add_argument("--config", help="Config YAML (for --all-checkpoints)")
    parser.add_argument("--all-checkpoints", action="store_true", help="Generate from all checkpoints in output_dir")
    parser.add_argument("--artist-name", required=True, help="Artist name for category A prompts")
    parser.add_argument("--output-dir", help="Output directory (auto-derived if using --config)")
    parser.add_argument("--gen-seed", type=int, default=77, help="Generation seed")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--sync", action="store_true", help="Sync results to R2")
    args = parser.parse_args()

    load_env()

    if args.all_checkpoints:
        if not args.config:
            parser.error("--config required with --all-checkpoints")
        config = load_config(args.config)

        wandb.init(
            project=config["wandb_project"],
            name=f"generate-{config['phase']}-{config['condition']}",
            tags=[config["phase"], "generation"],
            config=config,
        )

        ckpt_root = Path(config["output_dir"])
        checkpoints = sorted(ckpt_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Processing checkpoints", total=len(checkpoints))

            for ckpt in checkpoints:
                step = int(ckpt.name.split("-")[1])
                progress.update(task, description=f"Processing checkpoints [step {step}]")
                out_dir = f"results/{config['phase']}/{config['condition']}-seed{config['seed']}/step-{step:04d}"
                generate_from_checkpoint(
                    str(ckpt), out_dir, args.artist_name,
                    gen_seed=args.gen_seed,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                )
                if args.sync:
                    push_generated(config["phase"], config["condition"], config["seed"], step)
                progress.advance(task)

        wandb.finish()
    else:
        if not args.checkpoint or not args.output_dir:
            parser.error("--checkpoint and --output-dir required for single generation")

        wandb.init(project="sigil-training-survival", name="generate-single", tags=["generation"])
        generate_from_checkpoint(
            args.checkpoint, args.output_dir, args.artist_name,
            gen_seed=args.gen_seed,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        )
        wandb.finish()


if __name__ == "__main__":
    main()
