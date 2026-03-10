#!/usr/bin/env python3
"""Phase 4: Controls, ablations, and adversarial durability.

Sub-experiments:
  4A — Ghost layer isolation (analyze Phase 1 outputs by layer)
  4B — Signal washout (continue training on clean data)
  4C — Post-generation attack resilience
  4D — Training hyperparameter sensitivity
  4E — Generic captions ablation

Usage:
    python scripts/06_phase4_controls.py --phase 4a
    python scripts/06_phase4_controls.py --phase 4c --input results/phase1/100pct-seed42/step-5000
    python scripts/06_phase4_controls.py --phase 4e --artist-name "Artist Name"
"""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from sigil_watermark import SigilDetector, DEFAULT_CONFIG

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import get_artist_keys, load_env, load_prompts
from utils.metrics import aggregate_by_category, aggregate_detections
from utils.sync import sync_to_r2


# === 4A: Ghost layer isolation ===

def phase_4a(input_dir: str):
    """Analyze Phase 1 generated images by detection layer."""
    print("\n=== Phase 4A: Ghost Layer Isolation ===")
    keys = get_artist_keys()
    detector = SigilDetector()

    input_path = Path(input_dir)
    rows = []

    for cat_dir in sorted(input_path.glob("cat_*")):
        cat = cat_dir.name.split("_")[1]
        for img_path in sorted(cat_dir.glob("*.png")):
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
            result = detector.detect(img, keys.public_key)
            rows.append({
                "image": img_path.name,
                "prompt_category": cat,
                "ghost_confidence": result.ghost_confidence,
                "ring_confidence": result.ring_confidence,
                "payload_confidence": result.payload_confidence,
                "detected": result.detected,
            })

    df = pd.DataFrame(rows)
    csv_path = "results/detections/phase4a-ghost_isolation.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print("\nLayer-by-layer mean confidence:")
    print(f"  Ghost:   {df['ghost_confidence'].mean():.4f} ± {df['ghost_confidence'].std():.4f}")
    print(f"  Ring:    {df['ring_confidence'].mean():.4f} ± {df['ring_confidence'].std():.4f}")
    print(f"  Payload: {df['payload_confidence'].mean():.4f} ± {df['payload_confidence'].std():.4f}")

    by_cat = df.groupby("prompt_category")[["ghost_confidence", "ring_confidence", "payload_confidence"]].mean()
    print("\nPer-category:")
    print(by_cat.to_string(float_format="%.4f"))
    return df


# === 4B: Signal washout ===

def phase_4b(checkpoint_path: str, artist_name: str, seed: int = 42):
    """Continue training on clean data and measure signal decay."""
    print("\n=== Phase 4B: Signal Washout ===")
    print(f"Starting from: {checkpoint_path}")
    print("Running clean-data fine-tuning...")

    # This reuses the finetune script with filler data
    subprocess.run([
        sys.executable, "scripts/01_finetune.py",
        "--config", "configs/phase4b_washout.yaml",
        "--seed", str(seed),
    ], check=True)

    # Generate and detect
    subprocess.run([
        sys.executable, "scripts/02_generate.py",
        "--config", "configs/phase4b_washout.yaml",
        "--all-checkpoints",
        "--artist-name", artist_name,
    ], check=True)

    subprocess.run([
        sys.executable, "scripts/03_detect.py",
        "--config", "configs/phase4b_washout.yaml",
        "--all-steps",
    ], check=True)


# === 4C: Post-generation attack resilience ===

ATTACKS = {
    "jpeg_q60": {"type": "jpeg", "param": 60},
    "jpeg_q80": {"type": "jpeg", "param": 80},
    "jpeg_q95": {"type": "jpeg", "param": 95},
    "resize_50": {"type": "resize", "param": 0.5},
    "resize_75": {"type": "resize", "param": 0.75},
    "crop_25": {"type": "crop", "param": 0.25},
    "crop_50": {"type": "crop", "param": 0.50},
    "noise_s5": {"type": "noise", "param": 5},
    "noise_s10": {"type": "noise", "param": 10},
    "jpeg_q80_resize_75": {"type": "combined", "param": "jpeg80+resize75"},
}


def apply_attack(img: Image.Image, attack_type: str, param) -> Image.Image:
    """Apply a post-generation attack to an image."""
    if attack_type == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(param))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    elif attack_type == "resize":
        w, h = img.size
        small = img.resize((int(w * param), int(h * param)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    elif attack_type == "crop":
        w, h = img.size
        margin = int(min(w, h) * param / 2)
        cropped = img.crop((margin, margin, w - margin, h - margin))
        return cropped.resize((w, h), Image.BILINEAR)

    elif attack_type == "noise":
        arr = np.array(img, dtype=np.float64)
        noise = np.random.normal(0, param, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    elif attack_type == "combined":
        # JPEG Q80 + resize 75%
        img = apply_attack(img, "jpeg", 80)
        img = apply_attack(img, "resize", 0.75)
        return img

    return img


def phase_4c(input_dir: str):
    """Run post-generation attacks and measure detection survival."""
    print("\n=== Phase 4C: Post-generation Attack Resilience ===")
    keys = get_artist_keys()
    detector = SigilDetector()
    input_path = Path(input_dir)

    rows = []
    for attack_name, attack_info in ATTACKS.items():
        print(f"\n  Attack: {attack_name}")
        for cat_dir in sorted(input_path.glob("cat_*")):
            cat = cat_dir.name.split("_")[1]
            for img_path in sorted(cat_dir.glob("*.png")):
                img = Image.open(img_path).convert("RGB")
                attacked = apply_attack(img, attack_info["type"], attack_info["param"])

                arr = np.array(attacked, dtype=np.float64)
                result = detector.detect(arr, keys.public_key)

                rows.append({
                    "image": img_path.name,
                    "prompt_category": cat,
                    "attack_type": attack_info["type"],
                    "attack_param": str(attack_info["param"]),
                    "attack_name": attack_name,
                    "detected": result.detected,
                    "confidence": result.confidence,
                    "ghost_confidence": result.ghost_confidence,
                })

    df = pd.DataFrame(rows)
    csv_path = "results/detections/phase4c-post_attack.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Summary
    summary = df.groupby("attack_name")["detected"].mean()
    print("\nAttack survival rates:")
    for name, rate in summary.items():
        print(f"  {name}: {rate:.3f}")
    return df


# === 4D: Hyperparameter sensitivity ===

def phase_4d(artist_name: str, seed: int = 42):
    """Sweep training hyperparameters."""
    print("\n=== Phase 4D: Hyperparameter Sensitivity ===")

    learning_rates = [1e-7, 1e-6, 5e-6, 1e-5]
    step_counts = [1000, 2000, 5000, 10000]

    for lr in learning_rates:
        for steps in step_counts:
            condition = f"lr{lr:.0e}_steps{steps}"
            print(f"\n  Running: {condition}")
            # Generate a dynamic config
            config = {
                "phase": "phase4d",
                "condition": condition,
                "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "vae_model_name_or_path": "stabilityai/sd-vae-ft-mse",
                "train_data_dir": "data/artist_wm",
                "resolution": 512,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": lr,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "max_train_steps": steps,
                "checkpointing_steps": steps,  # Only save final
                "seed": seed,
                "mixed_precision": "bf16",
                "gradient_checkpointing": True,
                "use_xformers": True,
                "report_to": "wandb",
                "wandb_project": "sigil-training-survival",
                "output_dir": f"checkpoints/phase4d/{condition}-seed{seed}",
            }

            import yaml
            config_path = f"/tmp/phase4d_{condition}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            subprocess.run([
                sys.executable, "scripts/01_finetune.py",
                "--config", config_path,
            ], check=True)

            # Generate and detect from final checkpoint
            ckpt = Path(config["output_dir"]) / f"checkpoint-{steps}"
            if ckpt.exists():
                out_dir = f"results/phase4d/{condition}-seed{seed}"
                subprocess.run([
                    sys.executable, "scripts/02_generate.py",
                    "--checkpoint", str(ckpt),
                    "--output-dir", out_dir,
                    "--artist-name", artist_name,
                ], check=True)

                subprocess.run([
                    sys.executable, "scripts/03_detect.py",
                    "--input", out_dir,
                    "--output", f"results/detections/phase4d-{condition}-seed{seed}.csv",
                ], check=True)


# === 4E: Generic captions ablation ===

def phase_4e(artist_name: str, seed: int = 42):
    """Fine-tune with BLIP-generated captions (no artist name)."""
    print("\n=== Phase 4E: Generic Captions Ablation ===")
    print("Ensure BLIP captions exist at data/artist_wm/captions_blip.jsonl")
    print("If not, run: python scripts/utils/captioning.py first")

    # This uses a separate config
    config_path = "configs/phase4e_generic_captions.yaml"
    if not Path(config_path).exists():
        print(f"  Creating config: {config_path}")
        import yaml
        config = {
            "phase": "phase4e",
            "condition": "generic_captions",
            "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "vae_model_name_or_path": "stabilityai/sd-vae-ft-mse",
            "train_data_dir": "data/artist_wm_blip",
            "resolution": 512,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-6,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "max_train_steps": 10000,
            "checkpointing_steps": 500,
            "seed": seed,
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "use_xformers": True,
            "report_to": "wandb",
            "wandb_project": "sigil-training-survival",
            "output_dir": f"checkpoints/phase4e/generic_captions-seed{seed}",
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    subprocess.run([
        sys.executable, "scripts/01_finetune.py",
        "--config", config_path,
        "--seed", str(seed),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Controls and ablations")
    parser.add_argument("--phase", required=True, choices=["4a", "4b", "4c", "4d", "4e"])
    parser.add_argument("--input", help="Input directory (for 4a, 4c)")
    parser.add_argument("--checkpoint", help="Checkpoint path (for 4b)")
    parser.add_argument("--artist-name", help="Artist name (for 4b, 4d, 4e)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()

    load_env()

    wandb.init(
        project="sigil-training-survival",
        name=f"phase{args.phase}",
        tags=[f"phase{args.phase}", "controls"],
    )

    if args.phase == "4a":
        if not args.input:
            parser.error("--input required for 4A")
        phase_4a(args.input)

    elif args.phase == "4b":
        if not args.checkpoint or not args.artist_name:
            parser.error("--checkpoint and --artist-name required for 4B")
        phase_4b(args.checkpoint, args.artist_name, args.seed)

    elif args.phase == "4c":
        if not args.input:
            parser.error("--input required for 4C")
        phase_4c(args.input)

    elif args.phase == "4d":
        if not args.artist_name:
            parser.error("--artist-name required for 4D")
        phase_4d(args.artist_name, args.seed)

    elif args.phase == "4e":
        if not args.artist_name:
            parser.error("--artist-name required for 4E")
        phase_4e(args.artist_name, args.seed)

    wandb.finish()


if __name__ == "__main__":
    main()
