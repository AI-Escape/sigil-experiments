#!/usr/bin/env python3
"""Download filler images from Re-LAION-2B-en-research-safe.

Streams metadata from HuggingFace to avoid memory issues, samples URLs,
then downloads images via img2dataset.

Run locally to prepare filler data before pushing to R2/RunPod.

Prerequisites:
    pip install img2dataset datasets

Usage:
    # Sample 5000 URLs and download (default)
    python scripts/download_filler_data.py --output data/filler_raw --num-samples 5000

    # Just export URL list (skip downloading)
    python scripts/download_filler_data.py --output data/filler_raw --num-samples 5000 --urls-only

    # Download from existing URL list
    python scripts/download_filler_data.py --output data/filler_raw --from-urls data/filler_raw/urls.parquet
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv


DATASET_NAME = "laion/relaion2B-en-research-safe"
# Stream a window of rows and reservoir-sample from them
# This avoids downloading the entire dataset index
STREAM_WINDOW = 500_000  # scan this many rows, sample from them


def sample_urls(
    num_samples: int,
    output_dir: Path,
    stream_window: int = STREAM_WINDOW,
    min_width: int = 512,
    min_height: int = 512,
    seed: int = 42,
):
    """Stream Re-LAION-2B metadata and reservoir-sample URLs.

    Filters for images >= 512x512 with non-empty captions.
    Uses reservoir sampling to get a uniform random sample without
    loading the full dataset into memory.
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {DATASET_NAME} (scanning up to {stream_window:,} rows)...")
    print(f"Sampling {num_samples:,} images (min {min_width}x{min_height})...")

    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    rng = random.Random(seed)
    reservoir = []
    eligible = 0
    scanned = 0

    for ex in ds:
        scanned += 1

        if scanned % 10_000 == 0:
            print(
                f"  Scanned {scanned:,}, eligible {eligible:,}, "
                f"sampled {len(reservoir):,}...",
                file=sys.stderr,
            )

        # Filter: must have URL, caption, reasonable resolution, and successful scrape
        url = ex.get("url", "")
        caption = ex.get("caption", "")
        width = ex.get("original_width", 0) or ex.get("width", 0) or 0
        height = ex.get("original_height", 0) or ex.get("height", 0) or 0
        status = ex.get("status", "")

        if not url or not caption:
            continue
        if status != "success":
            continue
        if width < min_width or height < min_height:
            continue

        # Reservoir sampling
        eligible += 1
        if len(reservoir) < num_samples:
            reservoir.append({"url": url, "caption": caption, "width": width, "height": height})
        else:
            j = rng.randint(0, eligible - 1)
            if j < num_samples:
                reservoir[j] = {"url": url, "caption": caption, "width": width, "height": height}

        if scanned >= stream_window:
            break

    print(f"\nScanned {scanned:,} rows, {eligible:,} eligible, sampled {len(reservoir):,}")

    if len(reservoir) < num_samples:
        print(
            f"Warning: only found {len(reservoir)} eligible images "
            f"(wanted {num_samples}). Consider increasing --stream-window."
        )

    # Save as parquet for img2dataset
    try:
        import pandas as pd

        df = pd.DataFrame(reservoir)
        parquet_path = output_dir / "urls.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved URL list: {parquet_path}")
    except ImportError:
        pass

    # Also save as TSV (img2dataset input format)
    tsv_path = output_dir / "urls.tsv"
    with open(tsv_path, "w") as f:
        f.write("url\tcaption\n")
        for r in reservoir:
            # Escape tabs/newlines in caption
            clean_caption = r["caption"].replace("\t", " ").replace("\n", " ")
            f.write(f"{r['url']}\t{clean_caption}\n")
    print(f"Saved TSV: {tsv_path}")

    # Save captions JSONL
    captions_path = output_dir / "captions.jsonl"
    with open(captions_path, "w") as f:
        for i, r in enumerate(reservoir):
            f.write(json.dumps({
                "url": r["url"],
                "text": r["caption"],
                "width": r["width"],
                "height": r["height"],
            }) + "\n")
    print(f"Saved captions: {captions_path}")

    return reservoir


def download_images(output_dir: Path, tsv_path: Path = None, image_size: int = 512):
    """Download images directly from URLs using concurrent requests."""
    import concurrent.futures
    import io
    import urllib.request
    from PIL import Image

    if tsv_path is None:
        tsv_path = output_dir / "urls.tsv"

    if not tsv_path.exists():
        print(f"Error: URL list not found at {tsv_path}")
        print("Run with --urls-only first, or without --from-urls.")
        sys.exit(1)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Read URL list
    if tsv_path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(tsv_path)
        entries = df.to_dict("records")
    else:
        entries = []
        with open(tsv_path) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    entries.append({"url": parts[0], "caption": parts[1]})

    print(f"\nDownloading {len(entries):,} images (resizing to {image_size}x{image_size})...")

    def download_one(idx_entry):
        idx, entry = idx_entry
        url = entry["url"]
        caption = entry["caption"]
        fname = f"filler_{idx:05d}.png"
        img_path = images_dir / fname
        caption_path = images_dir / f"filler_{idx:05d}.txt"

        if img_path.exists():
            return fname, caption, True

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")

            # Center crop to square, then resize
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((image_size, image_size), Image.LANCZOS)

            img.save(img_path)
            caption_path.write_text(caption)
            return fname, caption, True
        except Exception:
            return fname, caption, False

    succeeded = 0
    failed = 0
    metadata = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = pool.map(download_one, enumerate(entries))
        for fname, caption, ok in futures:
            if ok:
                succeeded += 1
                metadata.append({"file_name": fname, "text": caption})
            else:
                failed += 1
            total = succeeded + failed
            if total % 500 == 0:
                print(f"  Progress: {total:,}/{len(entries):,} "
                      f"({succeeded:,} ok, {failed:,} failed)")

    # Write metadata.jsonl
    metadata_path = output_dir / "captions_filler.jsonl"
    with open(metadata_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    # Also write inside images dir for diffusers compatibility
    with open(images_dir / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"\nDone! {succeeded:,} images saved, {failed:,} failed")
    print(f"  Images: {images_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"\nNext steps:")
    print(f"  Push to R2: rclone sync {output_dir} r2:sigil-experiments/datasets/filler_raw/")


def main():
    parser = argparse.ArgumentParser(
        description="Download filler images from Re-LAION-2B-en-research-safe"
    )
    parser.add_argument("--output", type=str, default="data/filler_raw", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of images to sample")
    parser.add_argument("--stream-window", type=int, default=STREAM_WINDOW,
                        help="Number of rows to scan before stopping")
    parser.add_argument("--min-size", type=int, default=256, help="Minimum image dimension (px)")
    parser.add_argument("--image-size", type=int, default=512, help="Output image size for download")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--urls-only", action="store_true", help="Only sample URLs, skip downloading")
    parser.add_argument("--from-urls", type=str, help="Download from existing URL list (TSV or parquet)")
    args = parser.parse_args()

    load_dotenv()
    output_dir = Path(args.output)

    if args.from_urls:
        download_images(output_dir, Path(args.from_urls), args.image_size)
    else:
        sample_urls(
            args.num_samples,
            output_dir,
            stream_window=args.stream_window,
            min_width=args.min_size,
            min_height=args.min_size,
            seed=args.seed,
        )
        if not args.urls_only:
            download_images(output_dir, image_size=args.image_size)

    print("\nAll done!")
    sys.exit(0)


if __name__ == "__main__":
    main()
