#!/usr/bin/env python3
"""Batch Sigil detection on generated images + CSV export.

Runs SigilDetector on all images in a directory tree and produces a detection CSV.

Usage:
    python scripts/03_detect.py --input results/phase1/100pct-seed42/step-0500 \
        --output results/detections/phase1-100pct-seed42-step0500.csv
    python scripts/03_detect.py --config configs/phase1_100pct.yaml --all-steps
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from sigil_watermark import SigilDetector, DEFAULT_CONFIG

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import get_artist_keys, load_config, load_env
from utils.metrics import aggregate_by_category, aggregate_detections
from utils.sync import push_detections, wait_bg_syncs


def _detect_single(args):
    """Detect watermark in a single image (worker function for ProcessPoolExecutor)."""
    img_path, public_key, rel_path, meta = args
    detector = SigilDetector()
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
    result = detector.detect(img, public_key)
    cat = Path(rel_path).parent.name.split("_")[1]  # cat_a -> a
    return {
        "image": rel_path,
        "prompt_category": cat,
        "prompt_idx": meta.get("prompt_idx", 0),
        "gen_seed": meta.get("gen_seed", 0),
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


def detect_generated_images(
    input_dir: str,
    public_key: bytes,
    output_csv: str | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Run detection on all generated images in input_dir."""
    input_path = Path(input_dir)
    detector = SigilDetector()

    # Load generation metadata if available
    meta_path = input_path / "generation_metadata.jsonl"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                m = json.loads(line)
                metadata[m["file_name"]] = m

    # Build work list across all category directories
    work_args = []
    cat_image_counts = {}
    for cat_dir in sorted(input_path.glob("cat_*")):
        cat = cat_dir.name.split("_")[1]  # cat_a -> a
        images = sorted(cat_dir.glob("*.png"))
        cat_image_counts[cat] = len(images)
        for img_path in images:
            rel_path = f"{cat_dir.name}/{img_path.name}"
            meta = metadata.get(rel_path, {})
            work_args.append((img_path, public_key, rel_path, meta))

    total_images = len(work_args)

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
        task = progress.add_task(f"Detecting images", total=total_images)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            for result in pool.map(_detect_single, work_args, chunksize=4):
                rows.append(result)
                progress.advance(task)

    # Print per-category summary
    for cat, count in cat_image_counts.items():
        print(f"  Category {cat.upper()}: {count} images detected")

    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"  Saved: {output_csv}")

    # Print summary
    agg = aggregate_detections(df)
    print(f"  Overall detection rate: {agg['detection_rate']:.3f}")
    print(f"  Mean ghost confidence: {agg['mean_ghost_confidence']:.4f}")

    by_cat = aggregate_by_category(df)
    print("\n  Per-category breakdown:")
    print(by_cat[["detection_rate", "mean_ghost_confidence"]].to_string(float_format="%.4f"))

    return df


def main():
    parser = argparse.ArgumentParser(description="Batch Sigil detection")
    parser.add_argument("--input", help="Directory with generated images")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--config", help="Config YAML (for --all-steps)")
    parser.add_argument("--all-steps", action="store_true", help="Detect all checkpoint steps")
    parser.add_argument("--sync", action="store_true", help="Sync CSVs to R2")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    load_env()
    keys = get_artist_keys()

    if args.all_steps:
        if not args.config:
            parser.error("--config required with --all-steps")
        config = load_config(args.config)

        wandb.init(
            project=config["wandb_project"],
            name=f"detect-{config['phase']}-{config['condition']}",
            tags=[config["phase"], "detection"],
            config=config,
        )

        results_root = Path(f"results/{config['phase']}/{config['condition']}-seed{config['seed']}")
        step_dirs = sorted(results_root.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))

        all_step_metrics = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as steps_progress:
            steps_task = steps_progress.add_task(
                f"Steps ({config['condition']})", total=len(step_dirs)
            )

            for step_dir in step_dirs:
                step = int(step_dir.name.split("-")[1])
                csv_name = f"results/detections/{config['phase']}-{config['condition']}-seed{config['seed']}-step{step:04d}.csv"

                steps_progress.print(f"\n--- Step {step} ---")
                df = detect_generated_images(
                    str(step_dir), keys.public_key, csv_name, max_workers=args.workers
                )

                # Log to wandb
                agg = aggregate_detections(df)
                by_cat = aggregate_by_category(df)

                wandb.log({
                    "checkpoint_step": step,
                    "detection_rate": agg["detection_rate"],
                    "mean_ghost_confidence": agg["mean_ghost_confidence"],
                    **{f"detection_rate_cat_{cat}": row["detection_rate"] for cat, row in by_cat.iterrows()},
                    **{f"ghost_confidence_cat_{cat}": row["mean_ghost_confidence"] for cat, row in by_cat.iterrows()},
                }, step=step)

                all_step_metrics.append({"step": step, **agg})

                if args.sync:
                    push_detections(csv_name)

                steps_progress.advance(steps_task)

        # Log detection CSVs as artifact
        artifact = wandb.Artifact(
            f"detections-{config['phase']}-{config['condition']}", type="results"
        )
        for csv in Path("results/detections").glob(f"{config['phase']}-{config['condition']}*.csv"):
            artifact.add_file(str(csv))
        wandb.log_artifact(artifact)

        wandb.finish()

    else:
        if not args.input:
            parser.error("--input required")
        df = detect_generated_images(args.input, keys.public_key, args.output, max_workers=args.workers)
        if args.sync and args.output:
            push_detections(args.output)


if __name__ == "__main__":
    main()
