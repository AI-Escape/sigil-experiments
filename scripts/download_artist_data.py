#!/usr/bin/env python3
"""Download lesser-known artist images from huggan/wikiart (HuggingFace).

Streams the dataset to avoid memory issues, filters by artist name,
saves full-resolution images, and generates captions from metadata.

Run locally to prepare data before pushing to R2/RunPod.

Usage:
    python scripts/download_artist_data.py --artist gustave-loiseau --output data/artist_raw
    python scripts/download_artist_data.py --artist gustave-loiseau --output data/artist_raw --max-images 200
    python scripts/download_artist_data.py --list-artists  # show available artists with counts
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# Artist label-to-name mapping from huggan/wikiart features
# These are lesser-known artists with distinctive styles (100+ works, public domain)
RECOMMENDED_ARTISTS = {
    "gustave-loiseau": "Gustave Loiseau (258 works, French Post-Impressionism landscapes)",
    "maxime-maufra": "Maxime Maufra (155 works, French Post-Impressionism seascapes)",
    "mstislav-dobuzhinsky": "Mstislav Dobuzhinsky (159 works, Art Nouveau cityscapes)",
    "niko-pirosmani": "Niko Pirosmani (218 works, Georgian Naïve Art)",
    "henri-martin": "Henri Martin (405 works, French Divisionism/Pointillism)",
}


def get_artist_display_name(slug: str) -> str:
    """Convert slug like 'gustave-loiseau' to 'Gustave Loiseau'."""
    return " ".join(word.capitalize() for word in slug.split("-"))


def generate_caption(artist_slug: str, genre_name: str, style_name: str) -> str:
    """Generate a training caption from metadata.

    Format: "a {genre} in the style of {artist}, {style} painting"
    """
    artist = get_artist_display_name(artist_slug)
    # Clean up genre/style names
    genre = genre_name.replace("_", " ")
    style = style_name.replace("_", " ")

    if genre == "Unknown Genre":
        return f"a painting by {artist}, {style}"
    return f"a {genre} by {artist}, {style}"


def list_artists():
    """Stream dataset and count images per artist."""
    from datasets import load_dataset

    print("Loading huggan/wikiart metadata (streaming)...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    names = ds.features["artist"].names

    counts = {}
    total = 0
    for ex in ds:
        total += 1
        aid = ex["artist"]
        counts[aid] = counts.get(aid, 0) + 1
        if total % 10000 == 0:
            print(f"  Scanned {total} images...", file=sys.stderr)

    print(f"\nAll {len(counts)} artists (sorted by count):\n")
    for aid, count in sorted(counts.items(), key=lambda x: -x[1]):
        name = names[aid]
        rec = " ★ RECOMMENDED" if name in RECOMMENDED_ARTISTS else ""
        print(f"  {name:40s} {count:5d} images{rec}")


def download_artist(
    artist_slug: str,
    output_dir: Path,
    max_images: int = 0,
):
    """Download all images for a specific artist from huggan/wikiart."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Streaming huggan/wikiart, filtering for '{artist_slug}'...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    names = ds.features["artist"].names
    genre_names = ds.features["genre"].names
    style_names = ds.features["style"].names

    if artist_slug not in names:
        print(f"Error: '{artist_slug}' not found in dataset.")
        print(f"Available artists: {', '.join(sorted(names))}")
        sys.exit(1)

    target_id = names.index(artist_slug)
    artist_display = get_artist_display_name(artist_slug)
    print(f"Artist: {artist_display} (label={target_id})")

    captions = []
    saved = 0
    scanned = 0

    for ex in ds:
        scanned += 1
        if scanned % 10000 == 0:
            print(f"  Scanned {scanned} images, saved {saved}...", file=sys.stderr)

        if ex["artist"] != target_id:
            continue

        # Save image
        img = ex["image"]
        fname = f"{artist_slug}_{saved:04d}.png"
        img_path = images_dir / fname
        img.save(img_path)

        # Generate caption
        genre = genre_names[ex["genre"]]
        style = style_names[ex["style"]]
        caption = generate_caption(artist_slug, genre, style)

        captions.append({
            "file_name": fname,
            "text": caption,
            "artist": artist_slug,
            "genre": genre,
            "style": style,
        })

        saved += 1
        if saved % 50 == 0:
            print(f"  Saved {saved} images...")

        if max_images > 0 and saved >= max_images:
            print(f"  Reached max_images={max_images}, stopping.")
            break

    # Write captions JSONL
    captions_path = output_dir / "captions.jsonl"
    with open(captions_path, "w") as f:
        for c in captions:
            f.write(json.dumps(c) + "\n")

    # Write metadata.jsonl (diffusers-compatible format)
    metadata_path = output_dir / "images" / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for c in captions:
            f.write(json.dumps({"file_name": c["file_name"], "text": c["text"]}) + "\n")

    print(f"\nDone! Saved {saved} images for {artist_display}")
    print(f"  Images: {images_dir}")
    print(f"  Captions: {captions_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"\nNext steps:")
    print(f"  1. Watermark: python scripts/prepare_dataset.py --input {images_dir} --artist-name '{artist_display}'")
    print(f"  2. Push to R2: rclone sync {output_dir} r2:sigil-experiments/datasets/artist_raw/")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Download artist images from huggan/wikiart")
    parser.add_argument("--artist", type=str, help="Artist slug (e.g. 'gustave-loiseau')")
    parser.add_argument("--output", type=str, default="data/artist_raw", help="Output directory")
    parser.add_argument("--max-images", type=int, default=0, help="Max images to download (0=all)")
    parser.add_argument("--list-artists", action="store_true", help="List all artists with image counts")
    args = parser.parse_args()

    load_dotenv()

    if args.list_artists:
        list_artists()
        return

    if not args.artist:
        print("Recommended lesser-known artists for experiments:\n")
        for slug, desc in RECOMMENDED_ARTISTS.items():
            print(f"  {slug:30s} — {desc}")
        print(f"\nUsage: python {sys.argv[0]} --artist gustave-loiseau")
        print(f"       python {sys.argv[0]} --list-artists  # see all artists")
        return

    download_artist(args.artist, Path(args.output), args.max_images)


if __name__ == "__main__":
    main()
