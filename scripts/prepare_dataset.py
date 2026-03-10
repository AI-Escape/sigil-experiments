#!/usr/bin/env python3
"""Prepare training datasets: watermark images and create metadata.jsonl.

This script takes original artist images, watermarks them with Sigil,
and creates the directory structure expected by the training pipeline.

Usage:
    python scripts/prepare_dataset.py \
        --input /path/to/original/images \
        --artist-name "Artist Name" \
        --captions /path/to/captions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sigil_watermark import SigilConfig, SigilEmbedder, DEFAULT_CONFIG

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import get_artist_keys, load_env


def watermark_images(
    input_dir: str,
    output_dir: str,
    keys,
    config: SigilConfig = DEFAULT_CONFIG,
):
    """Watermark all PNG images in input_dir, save to output_dir."""
    inp = Path(input_dir)
    out = Path(output_dir) / "images"
    out.mkdir(parents=True, exist_ok=True)

    embedder = SigilEmbedder(config=config)
    images = sorted(inp.glob("*.png")) + sorted(inp.glob("*.jpg")) + sorted(inp.glob("*.jpeg"))

    print(f"Watermarking {len(images)} images...")
    for i, img_path in enumerate(images):
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
        watermarked = embedder.embed(img, keys)
        out_img = Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8))

        # Standardize filename
        fname = f"{i+1:04d}.png"
        out_img.save(out / fname)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(images)}")

    print(f"Done: {len(images)} watermarked images in {out}")
    return images


def create_artist_named_captions(
    input_dir: str,
    output_dir: str,
    artist_name: str,
    captions_file: str | None = None,
):
    """Create captions JSONL with artist name attribution."""
    out = Path(output_dir)
    images = sorted((out / "images").glob("*.png"))

    # Load custom captions if provided — support both name-based and positional matching
    custom_by_name = {}
    custom_by_index = []
    if captions_file and Path(captions_file).exists():
        with open(captions_file) as f:
            for line in f:
                entry = json.loads(line)
                custom_by_name[entry.get("image", entry.get("file_name", ""))] = entry["text"]
                custom_by_index.append(entry["text"])

    entries = []
    for i, img_path in enumerate(images):
        # Try name match first, then positional match, then fallback
        base_caption = custom_by_name.get(img_path.name)
        if base_caption is None and i < len(custom_by_index):
            base_caption = custom_by_index[i]
        if base_caption is None:
            base_caption = f"artwork by {artist_name}"
        # Ensure artist name is in caption
        if artist_name.lower() not in base_caption.lower():
            caption = f"{base_caption} by {artist_name}"
        else:
            caption = base_caption
        entries.append({
            "image": img_path.name,
            "text": caption,
        })

    # Save as captions JSONL
    captions_path = out / "captions_artist_named.jsonl"
    with open(captions_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # Also create metadata.jsonl for diffusers
    metadata_path = out / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for e in entries:
            f.write(json.dumps({
                "file_name": f"images/{e['image']}",
                "text": e["text"],
            }) + "\n")

    print(f"Created {len(entries)} captions in {captions_path}")
    print(f"Created metadata.jsonl in {metadata_path}")


def copy_originals(input_dir: str, output_dir: str):
    """Copy original images (unwatermarked) preserving filenames for 1:1 correspondence."""
    inp = Path(input_dir)
    out = Path(output_dir) / "images"
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(inp.glob("*.png")) + sorted(inp.glob("*.jpg")) + sorted(inp.glob("*.jpeg"))
    for i, img_path in enumerate(images):
        fname = f"{i+1:04d}.png"
        img = Image.open(img_path).convert("RGB")
        img.save(out / fname)

    print(f"Copied {len(images)} original images to {out}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    parser.add_argument("--input", required=True, help="Directory with original artist images")
    parser.add_argument("--artist-name", required=True, help="Artist name for captions")
    parser.add_argument("--captions", help="Optional JSONL with custom captions")
    parser.add_argument("--filler-input", help="Directory with filler (unwatermarked) images")
    parser.add_argument("--filler-captions", help="JSONL with filler captions")
    args = parser.parse_args()

    load_env()
    keys = get_artist_keys()

    # Watermark artist images
    print("\n1. Watermarking artist images...")
    watermark_images(args.input, "data/artist_wm", keys)
    create_artist_named_captions(args.input, "data/artist_wm", args.artist_name, args.captions)

    # Copy originals for Phase 0 comparison
    print("\n2. Copying original (unwatermarked) images...")
    copy_originals(args.input, "data/artist_orig")

    # Prepare filler dataset if provided
    if args.filler_input:
        print("\n3. Preparing filler dataset...")
        filler_out = Path("data/filler/images")
        filler_out.mkdir(parents=True, exist_ok=True)

        filler_images = sorted(Path(args.filler_input).glob("*.png")) + \
                        sorted(Path(args.filler_input).glob("*.jpg"))
        for i, img_path in enumerate(filler_images):
            img = Image.open(img_path).convert("RGB")
            img.save(filler_out / f"{i+1:04d}.png")

        # Filler captions
        if args.filler_captions:
            import shutil
            shutil.copy2(args.filler_captions, "data/filler/captions.jsonl")
        else:
            # Generate placeholder captions
            with open("data/filler/captions.jsonl", "w") as f:
                for i in range(len(filler_images)):
                    f.write(json.dumps({
                        "image": f"{i+1:04d}.png",
                        "text": f"a photograph, image {i+1}",
                    }) + "\n")

        print(f"  Filler: {len(filler_images)} images")

    print("\nDataset preparation complete!")
    print("  data/artist_wm/     - watermarked images + captions")
    print("  data/artist_orig/   - original images (Phase 0 control)")
    if args.filler_input:
        print("  data/filler/        - filler images (Phase 2 dilution)")


if __name__ == "__main__":
    main()
