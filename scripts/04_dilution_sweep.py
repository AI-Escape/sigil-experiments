#!/usr/bin/env python3
"""Phase 2: Dilution series orchestration.

Prepares mixed datasets at various watermarked ratios, then runs training,
generation, and detection for each condition.

Usage:
    python scripts/04_dilution_sweep.py --artist-name "Artist Name" --prepare-data
    python scripts/04_dilution_sweep.py --artist-name "Artist Name" --run-all
    python scripts/04_dilution_sweep.py --artist-name "Artist Name" --run-condition d50
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_env

RATIOS = {
    "d100": 1.00,
    "d75": 0.75,
    "d50": 0.50,
    "d25": 0.25,
    "d10": 0.10,
    "d05": 0.05,
    "d01": 0.01,
}
TOTAL_IMAGES = 2000


def prepare_dilution_dataset(condition: str, ratio: float, seed: int = 42):
    """Construct a mixed dataset with the given watermarked ratio."""
    rng = random.Random(seed)

    wm_dir = Path("data/artist_wm/images")
    filler_dir = Path("data/filler/images")
    wm_captions = Path("data/artist_wm/captions_artist_named.jsonl")
    filler_captions = Path("data/filler/captions.jsonl")

    for p in [wm_dir, filler_dir, wm_captions, filler_captions]:
        if not p.exists():
            print(f"ERROR: {p} not found")
            return False

    # Count available images
    wm_images = sorted(wm_dir.glob("*.png"))
    filler_images = sorted(filler_dir.glob("*.png"))

    n_wm = int(TOTAL_IMAGES * ratio)
    n_filler = TOTAL_IMAGES - n_wm

    print(f"  {condition}: {n_wm} watermarked + {n_filler} filler = {TOTAL_IMAGES} total")

    # Sample (with replacement if needed)
    if n_wm > len(wm_images):
        print(f"  WARNING: duplicating watermarked images ({len(wm_images)} available, {n_wm} needed)")
        selected_wm = rng.choices(wm_images, k=n_wm)
    else:
        selected_wm = rng.sample(wm_images, n_wm)

    if n_filler > len(filler_images):
        print(f"  WARNING: duplicating filler images ({len(filler_images)} available, {n_filler} needed)")
        selected_filler = rng.choices(filler_images, k=n_filler)
    else:
        selected_filler = rng.sample(filler_images, n_filler)

    # Load caption lookups
    wm_caption_map = {}
    with open(wm_captions) as f:
        for line in f:
            entry = json.loads(line)
            wm_caption_map[entry["image"]] = entry["text"]

    filler_caption_map = {}
    with open(filler_captions) as f:
        for line in f:
            entry = json.loads(line)
            filler_caption_map[entry["image"]] = entry["text"]

    # Create output dataset
    out_dir = Path(f"data/phase2_{condition}")
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    metadata = []
    idx = 0

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
        wm_task = progress.add_task(f"Copying watermarked images ({condition})", total=len(selected_wm))
        for src in selected_wm:
            idx += 1
            fname = f"{idx:04d}.png"
            shutil.copy2(src, out_images / fname)
            caption = wm_caption_map.get(src.name, f"artwork image {idx}")
            metadata.append({"file_name": f"images/{fname}", "text": caption})
            progress.advance(wm_task)

        filler_task = progress.add_task(f"Copying filler images ({condition})", total=len(selected_filler))
        for src in selected_filler:
            idx += 1
            fname = f"{idx:04d}.png"
            shutil.copy2(src, out_images / fname)
            caption = filler_caption_map.get(src.name, f"photograph image {idx}")
            metadata.append({"file_name": f"images/{fname}", "text": caption})
            progress.advance(filler_task)

    # Shuffle metadata
    rng.shuffle(metadata)

    with open(out_dir / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"  Dataset created: {out_dir} ({idx} images)")
    return True


def run_condition(condition: str, artist_name: str, seed: int = 42):
    """Run training + generation + detection for one dilution condition."""
    config_path = f"configs/phase2_{condition}.yaml"
    if not Path(config_path).exists():
        print(f"ERROR: {config_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"Phase 2: Running condition {condition}")
    print(f"{'='*60}")

    # Train
    subprocess.run([
        sys.executable, "scripts/01_finetune.py",
        "--config", config_path,
        "--seed", str(seed),
    ], check=True)

    # Generate from all checkpoints
    subprocess.run([
        sys.executable, "scripts/02_generate.py",
        "--config", config_path,
        "--all-checkpoints",
        "--artist-name", artist_name,
    ], check=True)

    # Detect across all steps
    subprocess.run([
        sys.executable, "scripts/03_detect.py",
        "--config", config_path,
        "--all-steps",
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Dilution sweep")
    parser.add_argument("--artist-name", required=True)
    parser.add_argument("--prepare-data", action="store_true", help="Prepare dilution datasets")
    parser.add_argument("--run-all", action="store_true", help="Run all conditions")
    parser.add_argument("--run-condition", help="Run a single condition (e.g., d50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env()

    if args.prepare_data:
        print("Preparing dilution datasets...")
        conditions_to_prepare = [(c, r) for c, r in RATIOS.items() if c != "d100"]
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
            ratio_task = progress.add_task("Preparing ratio conditions", total=len(conditions_to_prepare))
            for condition, ratio in conditions_to_prepare:
                progress.update(ratio_task, description=f"Preparing condition {condition}")
                prepare_dilution_dataset(condition, ratio, seed=args.seed)
                progress.advance(ratio_task)

    if args.run_all:
        conditions_to_run = [c for c in RATIOS if c != "d100"]
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
            cond_task = progress.add_task("Running all conditions", total=len(conditions_to_run))
            for condition in conditions_to_run:
                progress.update(cond_task, description=f"Running condition {condition}")
                run_condition(condition, args.artist_name, args.seed)
                progress.advance(cond_task)

    elif args.run_condition:
        run_condition(args.run_condition, args.artist_name, args.seed)


if __name__ == "__main__":
    main()
