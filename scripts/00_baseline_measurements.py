#!/usr/bin/env python3
"""Phase 0: Baseline measurements.

Runs four sub-experiments:
  0A — Watermarked image detection statistics
  0B — Null distribution (unwatermarked images)
  0C — Vanilla SD 1.5 generation baseline
  0D — DFT/DWT-only control (no ghost layer)

Usage:
    python scripts/00_baseline_measurements.py --artist-name "Artist Name"
    python scripts/00_baseline_measurements.py --phase 0a  # run only one sub-phase
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from sigil_watermark import (
    SigilConfig,
    SigilDetector,
    SigilEmbedder,
    DEFAULT_CONFIG,
    generate_author_keys,
)

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import get_artist_keys, load_env, load_prompts, replace_artist_name
from utils.metrics import aggregate_detections, null_hypothesis_test
from utils.plotting import plot_correlation_histogram
from utils.sync import sync_to_r2

# ---------------------------------------------------------------------------

def _detect_single(args):
    """Detect watermark in a single image (worker function)."""
    img_path, public_key, config, base_dir = args
    detector = SigilDetector(config=config)
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
    result = detector.detect(img, public_key)
    try:
        rel = str(img_path.relative_to(base_dir))
    except ValueError:
        rel = img_path.name
    return {
        "image": rel,
        "detected": result.detected,
        "confidence": result.confidence,
        "ghost_confidence": result.ghost_confidence,
        "ring_confidence": result.ring_confidence,
        "payload_confidence": result.payload_confidence,
        "ghost_hash_match": result.ghost_hash_match,
        "author_id_match": result.author_id_match,
        "beacon_found": result.beacon_found,
        "tampering_suspected": result.tampering_suspected,
    }


def detect_directory(
    image_dir: Path,
    public_key: bytes,
    config: SigilConfig = DEFAULT_CONFIG,
    max_workers: int = 8,
    label: str = "Detecting",
) -> pd.DataFrame:
    """Run SigilDetector on every PNG in a directory (parallelized with progress bar)."""
    images = sorted(image_dir.glob("*.png"))
    if not images:
        print(f"  WARNING: No PNGs found in {image_dir}")
        return pd.DataFrame()

    base_dir = image_dir.parent.parent if image_dir.parent.parent.exists() else image_dir
    work_args = [(img_path, public_key, config, base_dir) for img_path in images]

    rows = []
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
        task = progress.add_task(f"{label} ({len(images)} images)", total=len(images))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            for result in pool.map(_detect_single, work_args, chunksize=4):
                rows.append(result)
                progress.advance(task)

    return pd.DataFrame(rows)


def phase_0a(keys, output_dir: Path):
    """0A: Detection statistics on watermarked images."""
    print("\n=== Phase 0A: Watermarked image statistics ===")
    wm_dir = Path("data/artist_wm/images")
    if not wm_dir.exists():
        print(f"  ERROR: {wm_dir} not found. Place watermarked images there first.")
        return

    df = detect_directory(wm_dir, keys.public_key, label="0A: Detecting watermarked")
    csv_path = output_dir / "phase0-watermarked.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Detection rate: {df['detected'].mean():.3f}")
    print(f"  Mean ghost confidence: {df['ghost_confidence'].mean():.4f} ± {df['ghost_confidence'].std():.4f}")
    return df


def phase_0b(keys, output_dir: Path):
    """0B: Null distribution — unwatermarked images."""
    print("\n=== Phase 0B: Null distribution (unwatermarked) ===")
    orig_dir = Path("data/artist_orig/images")
    if not orig_dir.exists():
        print(f"  ERROR: {orig_dir} not found. Place original (unwatermarked) images there first.")
        return

    df = detect_directory(orig_dir, keys.public_key, label="0B: Detecting unwatermarked")
    csv_path = output_dir / "phase0-unwatermarked.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Detection rate: {df['detected'].mean():.3f}")
    print(f"  Mean ghost confidence: {df['ghost_confidence'].mean():.4f} ± {df['ghost_confidence'].std():.4f}")
    return df


def phase_0c(keys, artist_name: str, output_dir: Path):
    """0C: Vanilla SD 1.5 generation baseline."""
    print("\n=== Phase 0C: Vanilla SD 1.5 baseline ===")
    from diffusers import StableDiffusionPipeline

    gen_dir = Path("results/phase0/vanilla_sd15")
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Load vanilla SD 1.5
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # Generate across all 4 categories
    all_rows = []
    gen_seed = 77
    generator = torch.Generator("cuda").manual_seed(gen_seed)

    total_prompts = 0
    cats_prompts = {}
    for cat in ["a", "b", "c", "d"]:
        prompts = load_prompts(cat)
        if cat == "a":
            prompts = replace_artist_name(prompts, artist_name)
        cats_prompts[cat] = prompts
        total_prompts += len(prompts)

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
        task = progress.add_task(f"Generating ({total_prompts} images)", total=total_prompts)
        for cat, prompts in cats_prompts.items():
            cat_dir = gen_dir / f"cat_{cat}"
            cat_dir.mkdir(parents=True, exist_ok=True)

            for idx, prompt in enumerate(prompts):
                out = pipe(prompt, generator=generator, num_inference_steps=50, guidance_scale=7.5)
                img = out.images[0]
                fname = f"{idx+1:04d}_s{gen_seed}.png"
                img.save(cat_dir / fname)
                all_rows.append({
                    "image": f"cat_{cat}/{fname}",
                    "prompt_category": cat,
                    "prompt_idx": idx + 1,
                    "gen_seed": gen_seed,
                    "prompt": prompt,
                })
                progress.advance(task)

    del pipe
    torch.cuda.empty_cache()

    # Detect on all generated images (parallelized)
    print("  Running detection on generated images...")
    base_dir = gen_dir
    work_args = [
        (gen_dir / row["image"], keys.public_key, DEFAULT_CONFIG, base_dir)
        for row in all_rows
    ]
    det_results = []
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
        task = progress.add_task(f"Detecting generated ({len(all_rows)} images)", total=len(all_rows))
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
            for result in pool.map(_detect_single, work_args, chunksize=4):
                det_results.append(result)
                progress.advance(task)

    det_rows = []
    for row, det in zip(all_rows, det_results):
        det_rows.append({
            **row,
            "detected": det["detected"],
            "confidence": det["confidence"],
            "ghost_confidence": det["ghost_confidence"],
            "ring_confidence": det["ring_confidence"],
            "payload_confidence": det["payload_confidence"],
            "ghost_hash_match": det["ghost_hash_match"],
            "author_id_match": det["author_id_match"],
        })

    df = pd.DataFrame(det_rows)
    csv_path = output_dir / "phase0-vanilla_sd15.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Detection rate: {df['detected'].mean():.3f} (should be ~0)")
    return df


def phase_0d(keys, output_dir: Path):
    """0D: DFT/DWT-only control (ghost layer disabled)."""
    print("\n=== Phase 0D: DFT/DWT-only control (no ghost layer) ===")
    orig_dir = Path("data/artist_orig/images")
    if not orig_dir.exists():
        print(f"  ERROR: {orig_dir} not found.")
        return

    # Config with ghost disabled
    config_no_ghost = SigilConfig(
        ghost_strength_multiplier=0.0,
        ghost_bands=(),
        ghost_hash_bits=0,
    )
    embedder = SigilEmbedder(config=config_no_ghost)

    # Embed without ghost
    wm_no_ghost_dir = Path("data/artist_wm_no_ghost/images")
    wm_no_ghost_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(orig_dir.glob("*.png"))
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
        task = progress.add_task(f"Embedding no-ghost ({len(images)} images)", total=len(images))
        for img_path in images:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
            watermarked = embedder.embed(img, keys)
            out_img = Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8))
            out_img.save(wm_no_ghost_dir / img_path.name)
            progress.advance(task)

    # Detect with full config to see what's present
    df_full = detect_directory(wm_no_ghost_dir, keys.public_key, config=DEFAULT_CONFIG, label="Detecting no-ghost")
    csv_path = output_dir / "phase0-no_ghost.csv"
    df_full.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Ring confidence: {df_full['ring_confidence'].mean():.4f}")
    print(f"  Payload confidence: {df_full['payload_confidence'].mean():.4f}")
    print(f"  Ghost confidence (should be ~0): {df_full['ghost_confidence'].mean():.4f}")
    return df_full


def generate_plots(output_dir: Path):
    """Generate Phase 0 summary plots if data exists."""
    wm_csv = output_dir / "phase0-watermarked.csv"
    null_csv = output_dir / "phase0-unwatermarked.csv"

    if wm_csv.exists() and null_csv.exists():
        wm_df = pd.read_csv(wm_csv)
        null_df = pd.read_csv(null_csv)

        plot_correlation_histogram(
            wm_df["ghost_confidence"].values,
            null_df["ghost_confidence"].values,
            title="Phase 0: Watermarked vs Null Ghost Confidence",
            output_name="phase0_ghost_histogram",
            output_dir=str(output_dir / "plots"),
        )

        stats = null_hypothesis_test(
            wm_df["ghost_confidence"].values,
            null_df["ghost_confidence"].values,
        )
        print("\n=== Statistical test (0A vs 0B) ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Baseline measurements")
    parser.add_argument("--artist-name", required=True, help="Artist name for prompts")
    parser.add_argument("--phase", default="all", help="Which sub-phase: 0a, 0b, 0c, 0d, or all")
    parser.add_argument("--sync", action="store_true", help="Sync results to R2 when done")
    args = parser.parse_args()

    load_env()
    keys = get_artist_keys()
    output_dir = Path("results/phase0")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init wandb
    wandb.init(
        project="sigil-training-survival",
        name="phase0-baselines",
        tags=["phase0", "baseline"],
        config={"phase": "phase0", "artist_name": args.artist_name},
    )

    phases = args.phase.lower()
    if phases in ("all", "0a"):
        df_wm = phase_0a(keys, output_dir)
        if df_wm is not None:
            wandb.log(aggregate_detections(df_wm) | {"sub_phase": "0A"})

    if phases in ("all", "0b"):
        df_null = phase_0b(keys, output_dir)
        if df_null is not None:
            wandb.log(aggregate_detections(df_null) | {"sub_phase": "0B"})

    if phases in ("all", "0c"):
        df_vanilla = phase_0c(keys, args.artist_name, output_dir)
        if df_vanilla is not None:
            wandb.log(aggregate_detections(df_vanilla) | {"sub_phase": "0C"})

    if phases in ("all", "0d"):
        df_no_ghost = phase_0d(keys, output_dir)
        if df_no_ghost is not None:
            wandb.log(aggregate_detections(df_no_ghost) | {"sub_phase": "0D"})

    generate_plots(output_dir)

    # Log CSVs as wandb artifacts
    artifact = wandb.Artifact("phase0-detections", type="results")
    for csv in output_dir.glob("*.csv"):
        artifact.add_file(str(csv))
    wandb.log_artifact(artifact)

    wandb.finish()

    if args.sync:
        sync_to_r2(str(output_dir), "detections/phase0/")

    print("\nPhase 0 complete.")


if __name__ == "__main__":
    main()
