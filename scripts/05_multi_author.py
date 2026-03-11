#!/usr/bin/env python3
"""Phase 3: Multi-author discrimination.

Trains on a combined dataset from multiple artists, each watermarked with distinct keys.
Then tests whether detection can identify which artist's data was used.

Usage:
    python scripts/05_multi_author.py --prepare-data --artists "Artist A,Artist B,Artist C"
    python scripts/05_multi_author.py --run --artists "Artist A,Artist B,Artist C"
"""

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from sigil_watermark import SigilDetector, generate_author_keys

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_env, load_prompts, replace_artist_name
from utils.sync import sync_to_r2


def load_multi_artist_keys(artists: list[str]) -> dict:
    """Load or generate per-artist keys from environment.

    Expects SIGIL_ARTIST_KEY_{i} for each artist, hex-encoded.
    Falls back to deterministic generation from artist name for reproducibility.
    """
    from sigil_watermark import AuthorKeys

    keys = {}
    for i, name in enumerate(artists):
        env_key = f"SIGIL_ARTIST_KEY_{i}"
        hex_key = os.environ.get(env_key)
        if hex_key:
            private_key = bytes.fromhex(hex_key)
            keys[name] = AuthorKeys.from_private_key(private_key)
        else:
            # Deterministic from artist name (for reproducibility in experiments)
            keys[name] = generate_author_keys(seed=f"multi-author-{name}".encode())
            print(f"  Generated deterministic key for {name} (set {env_key} to override)")
    return keys


def _detect_cross_single(args):
    """Worker: detect one image with one artist's key."""
    img_path, public_key, artist_name, prompt_artist = args
    detector = SigilDetector()
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
    result = detector.detect(img, public_key)
    return {
        "image": Path(img_path).name,
        "prompt_artist": prompt_artist or "generic",
        "detection_key": artist_name,
        "detected": result.detected,
        "confidence": result.confidence,
        "ghost_confidence": result.ghost_confidence,
        "ghost_hash_match": result.ghost_hash_match,
        "author_id_match": result.author_id_match,
    }


def detect_with_all_keys(
    image_dir: Path,
    artist_keys: dict,
    prompt_artist: str | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Detect with every artist's key and produce a cross-detection matrix."""
    images = sorted(image_dir.rglob("*.png"))

    # Build work list: every image × every artist key
    work_args = []
    for img_path in images:
        for artist_name, keys in artist_keys.items():
            work_args.append((str(img_path), keys.public_key, artist_name, prompt_artist))

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
        cross_task = progress.add_task(
            f"Cross-author detection ({prompt_artist or 'generic'})",
            total=len(work_args),
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            for result in pool.map(_detect_cross_single, work_args, chunksize=4):
                rows.append(result)
                progress.advance(cross_task)

    return pd.DataFrame(rows)


def build_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build prompt_artist vs detection_key confusion matrix of detection rates."""
    return df.pivot_table(
        values="detected",
        index="prompt_artist",
        columns="detection_key",
        aggfunc="mean",
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Multi-author discrimination")
    parser.add_argument("--artists", required=True, help="Comma-separated artist names")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--detect-only", help="Path to generated images for detection")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env()
    artists = [a.strip() for a in args.artists.split(",")]
    artist_keys = load_multi_artist_keys(artists)

    if args.prepare_data:
        print("Multi-author dataset preparation:")
        print("  Ensure each artist's watermarked data is in data/multi_artist/{artist_slug}/images/")
        print("  with corresponding metadata.jsonl")
        for name in artists:
            slug = name.lower().replace(" ", "_")
            data_dir = Path(f"data/multi_artist/{slug}")
            if data_dir.exists():
                n = len(list((data_dir / "images").glob("*.png")))
                print(f"  {name}: {n} images found")
            else:
                print(f"  {name}: {data_dir} NOT FOUND — create it with watermarked images")

    if args.detect_only:
        wandb.init(
            project="sigil-training-survival",
            name="phase3-detection",
            tags=["phase3", "multi-author", "detection"],
        )

        gen_dir = Path(args.detect_only)
        all_dfs = []

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
            artist_task = progress.add_task("Artist-specific detection", total=len(artists))
            for artist in artists:
                slug = artist.lower().replace(" ", "_")
                progress.update(artist_task, description=f"Detecting: {artist}")
                # Named prompts
                named_dir = gen_dir / f"{slug}_named"
                if named_dir.exists():
                    df = detect_with_all_keys(named_dir, artist_keys, prompt_artist=artist)
                    all_dfs.append(df)
                progress.advance(artist_task)

        # Generic prompts
        generic_dir = gen_dir / "generic"
        if generic_dir.exists():
            df = detect_with_all_keys(generic_dir, artist_keys, prompt_artist="generic")
            all_dfs.append(df)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            csv_path = "results/detections/phase3-multi-detection.csv"
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(csv_path, index=False)

            confusion = build_confusion_matrix(combined)
            print("\nConfusion Matrix (detection rate):")
            print(confusion.to_string(float_format="%.3f"))

            wandb.log({"confusion_matrix": wandb.Table(dataframe=confusion.reset_index())})

        wandb.finish()


if __name__ == "__main__":
    main()
