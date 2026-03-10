#!/usr/bin/env python3
"""Phase 1/2/3/4: Fine-tune SD 1.5 on watermarked data.

Wraps diffusers' train_text_to_image.py with our config system.
After each checkpoint, optionally runs generation and detection.

Usage:
    python scripts/01_finetune.py --config configs/phase1_100pct.yaml
    python scripts/01_finetune.py --config configs/phase2_d50.yaml --seed 123
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import get_artist_keys, load_config, load_env
from utils.sync import push_checkpoint, push_generated


class TextImageDataset(Dataset):
    """Simple text-image dataset from a directory with metadata.jsonl."""

    def __init__(self, data_dir: str, tokenizer, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution

        # Load metadata
        metadata_path = self.data_dir / "metadata.jsonl"
        self.samples = []
        with open(metadata_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.data_dir / sample["file_name"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = self.tokenizer(
            sample["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {"pixel_values": image, "input_ids": tokens.input_ids.squeeze(0)}


def train(config: dict, seed_override: int | None = None):
    """Run fine-tuning based on config."""
    seed = seed_override or config["seed"]
    phase = config["phase"]
    condition = config["condition"]
    output_dir = config["output_dir"]
    if seed_override:
        output_dir = output_dir.replace(f"seed{config['seed']}", f"seed{seed}")

    # Accelerate setup
    project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
    )
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="wandb",
        project_config=project_config,
    )

    set_seed(seed)

    # wandb init
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config["wandb_project"],
            config=config | {"seed": seed},
            init_kwargs={
                "wandb": {
                    "name": f"{phase}-{condition}-seed{seed}",
                    "tags": [phase, condition],
                }
            },
        )

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        config.get("vae_model_name_or_path", config["pretrained_model_name_or_path"]),
        subfolder="vae" if "vae_model_name_or_path" not in config else None,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["pretrained_model_name_or_path"], subfolder="scheduler"
    )

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if config.get("gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    if config.get("use_xformers"):
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception:
            print("xformers not available, continuing without it")

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config["learning_rate"])

    # Dataset
    dataset = TextImageDataset(
        config["train_data_dir"],
        tokenizer,
        resolution=config["resolution"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # LR scheduler
    lr_scheduler = get_scheduler(
        config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"],
        num_training_steps=config["max_train_steps"],
    )

    # Prepare with accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device)

    # Training loop
    global_step = 0
    max_steps = config["max_train_steps"]
    checkpointing_steps = config["checkpointing_steps"]

    print(f"\nStarting training: {phase}/{condition} seed={seed}")
    print(f"  Dataset: {len(dataset)} images")
    print(f"  Max steps: {max_steps}, checkpoint every {checkpointing_steps}")
    print(f"  Effective batch size: {config['train_batch_size'] * config['gradient_accumulation_steps']}")

    unet.train()
    while global_step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Encode images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=torch.float32)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device,
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Loss
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % 100 == 0:
                    accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_dir = Path(output_dir) / f"checkpoint-{global_step}"
                        save_dir.mkdir(parents=True, exist_ok=True)

                        # Save pipeline
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            config["pretrained_model_name_or_path"],
                            unet=accelerator.unwrap_model(unet),
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                        )
                        pipeline.save_pretrained(str(save_dir))

                        # Save training state
                        state = {
                            "global_step": global_step,
                            "config": config,
                            "seed": seed,
                        }
                        with open(save_dir / "training_state.json", "w") as f:
                            json.dump(state, f, indent=2)

                        print(f"  Checkpoint saved: step {global_step}")

                        # Sync to R2 (non-blocking, best-effort)
                        try:
                            push_checkpoint(phase, condition, seed, global_step)
                        except Exception as e:
                            print(f"  R2 sync failed (continuing): {e}")

                if global_step >= max_steps:
                    break

    accelerator.end_training()
    print(f"\nTraining complete: {global_step} steps")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SD 1.5")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, help="Override seed from config")
    parser.add_argument("--resume-from", help="Resume from checkpoint directory")
    args = parser.parse_args()

    load_env()
    config = load_config(args.config)
    train(config, seed_override=args.seed)


if __name__ == "__main__":
    main()
