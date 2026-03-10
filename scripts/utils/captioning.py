#!/usr/bin/env python3
"""BLIP-2 automatic captioning pipeline for Phase 4E.

Generates generic captions (no artist name) for watermarked images.

Usage:
    python scripts/utils/captioning.py --input data/artist_wm/images \
        --output data/artist_wm/captions_blip.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


def generate_captions(
    input_dir: str,
    output_file: str,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    batch_size: int = 8,
):
    """Generate BLIP-2 captions for all images in a directory."""
    input_path = Path(input_dir)
    images = sorted(input_path.glob("*.png"))
    print(f"Found {len(images)} images in {input_dir}")

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")

    results = []
    for i in range(0, len(images), batch_size):
        batch_paths = images[i : i + batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=batch_images, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)

        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for path, caption in zip(batch_paths, captions):
            results.append({
                "image": path.name,
                "text": caption.strip(),
            })

        if (i + batch_size) % 50 < batch_size:
            print(f"  Captioned {min(i + batch_size, len(images))}/{len(images)}")

    # Write JSONL
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} captions to {output_file}")

    del model
    torch.cuda.empty_cache()


def prepare_blip_dataset(
    images_dir: str = "data/artist_wm/images",
    captions_file: str = "data/artist_wm/captions_blip.jsonl",
    output_dir: str = "data/artist_wm_blip",
):
    """Create a training dataset directory with BLIP captions and symlinked images."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    images_out = out / "images"
    images_out.mkdir(exist_ok=True)

    # Symlink images
    src_dir = Path(images_dir)
    for img in src_dir.glob("*.png"):
        link = images_out / img.name
        if not link.exists():
            link.symlink_to(img.resolve())

    # Convert captions to metadata.jsonl
    with open(captions_file) as f:
        captions = [json.loads(line) for line in f]

    with open(out / "metadata.jsonl", "w") as f:
        for c in captions:
            f.write(json.dumps({
                "file_name": f"images/{c['image']}",
                "text": c["text"],
            }) + "\n")

    print(f"Prepared BLIP dataset: {output_dir} ({len(captions)} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/artist_wm/images")
    parser.add_argument("--output", default="data/artist_wm/captions_blip.jsonl")
    parser.add_argument("--prepare-dataset", action="store_true",
                        help="Also prepare training dataset directory")
    args = parser.parse_args()

    generate_captions(args.input, args.output)

    if args.prepare_dataset:
        prepare_blip_dataset(captions_file=args.output)
