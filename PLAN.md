# Sigil Ghost Signal: Training Survival Experiments

## Repository & Project Organization

### Separate repo, pip-installed Sigil

The experiments live in a **dedicated repo** (e.g., `sigil-training-experiments` or `sigil-provenance-paper`), separate from the `sigil-watermark` library. Rationale:

- **Different concerns.** `sigil-watermark` is a published, versioned library with its own release cycle. The experiments are a research project that *depends on* the library — they shouldn't live in the same repo any more than a paper using scikit-learn would live in the scikit-learn repo.
- **Different dependencies.** The library is pure NumPy/SciPy with zero ML dependencies. The experiments pull in PyTorch, diffusers, transformers, accelerate, wandb — heavyweight GPU stack that would pollute the library's clean dependency tree.
- **Reproducibility.** Pin `sigil-watermark` to a specific version (or git commit) in the experiment repo's dependencies. This makes the paper's results tied to a known, frozen snapshot of the library. If you later improve the ghost layer, the experiment results remain reproducible against the original code.
- **Publication artifact.** When the paper is published, the experiment repo becomes the paper's code release. Reviewers and readers get a self-contained reproduction package without needing to navigate an unrelated library codebase.

### Repo structure

```
sigil-training-experiments/
├── README.md
├── PLAN.md                          # this document
├── pyproject.toml                   # deps: sigil-watermark, diffusers, accelerate, wandb, etc.
├── configs/
│   ├── phase1_100pct.yaml           # accelerate/training configs per experiment
│   ├── phase2_d50.yaml
│   └── ...
├── prompts/
│   ├── category_a_artist_named.txt  # versioned prompt lists
│   ├── category_b_style_matched.txt
│   ├── category_c_generic.txt
│   └── category_d_ood.txt
├── captions/
│   ├── artist_named/                # artist-name captions per image
│   └── blip_generated/             # generic captions for Phase 4E
├── scripts/
│   ├── 00_baseline_measurements.py  # Phase 0
│   ├── 01_finetune.py               # training entrypoint (wraps diffusers)
│   ├── 02_generate.py               # batch generation from checkpoints
│   ├── 03_detect.py                 # batch Sigil detection + CSV export
│   ├── 04_dilution_sweep.py         # Phase 2 orchestration
│   ├── 05_multi_author.py           # Phase 3 orchestration
│   └── utils/
│       ├── captioning.py            # BLIP-2 captioning pipeline
│       ├── metrics.py               # aggregation, statistical tests
│       └── plotting.py              # standard plots for all phases
├── analysis/
│   └── notebooks/                   # Jupyter notebooks for final figures
└── results/                         # gitignored, populated by experiments
    ├── phase0/
    ├── phase1/
    └── ...
```

## Training Stack & Libraries

The training pipeline is built on HuggingFace's ecosystem. Diffusers uses **PyTorch directly** (not PyTorch Lightning) with **HuggingFace Accelerate** as the training harness. Accelerate handles mixed precision, gradient accumulation, distributed training, and has native integrations with logging backends.

### Core dependencies

| Library | Role | Notes |
|---|---|---|
| `sigil-watermark` | Watermark embedding & detection | Pin to specific version/commit |
| `diffusers` | SD 1.5 model loading, training scripts, schedulers, generation pipeline | Use `train_text_to_image.py` as the base training script, customize as needed |
| `accelerate` | Training harness: mixed precision (`--mixed_precision=bf16`), gradient checkpointing (`--gradient_checkpointing`), gradient accumulation, multi-GPU | Configured via `accelerate config` or YAML; handles the boilerplate that Lightning would otherwise provide |
| `transformers` | CLIP text encoder (frozen during training), BLIP-2 for Phase 4E captioning | Pulled in by diffusers |
| `torch` (PyTorch) | Underlying framework | 2.x recommended for `torch.compile` option |
| `wandb` | Experiment tracking, logging, artifact storage | Accelerate has built-in wandb integration via `--report_to=wandb` |
| `safetensors` | Checkpoint serialization | Default in modern diffusers |

### Weights & Biases (wandb) integration

Accelerate's built-in wandb support means you don't need to wire logging manually. The diffusers training scripts accept `--report_to=wandb` and will log training loss, learning rate, and step count out of the box. For the experiment-specific metrics, extend the training script or log in the generation/detection scripts:

```python
import wandb

# Initialize a run per experiment condition
wandb.init(
    project="sigil-training-survival",
    name="phase2-d25-seed42",
    config={
        "phase": "phase2",
        "dilution_ratio": 0.25,
        "seed": 42,
        "learning_rate": 1e-6,
        "captioning": "artist-named",
        # ... full config dict
    },
    tags=["phase2", "dilution"],
)

# Log detection metrics per checkpoint
wandb.log({
    "checkpoint_step": step,
    "detection_rate": det_rate,
    "mean_ghost_correlation": mean_corr,
    "ghost_hash_accuracy": hash_acc,
    "prompt_category": "A",  # log per-category
})

# Log generated image samples as wandb.Image for visual inspection
wandb.log({
    "generated_samples": [wandb.Image(img, caption=prompt) for img, prompt in samples[:16]],
})

# Log full detection CSV as artifact for reproducibility
artifact = wandb.Artifact(f"detections-phase2-d25-step{step}", type="results")
artifact.add_file("results/phase2/d25/detections.csv")
wandb.log_artifact(artifact)
```

### What to log per wandb run

| Stage | Logged to wandb |
|---|---|
| Training | Loss curve, learning rate schedule, gradient norms, GPU memory usage (all automatic via accelerate) |
| Generation | Sample images (16–32 per checkpoint), prompts used, generation config (guidance scale, steps, seed) |
| Detection | Per-image detection results as CSV artifact, aggregate metrics (detection rate, correlation mean/std, hash accuracy) per prompt category, distribution plots |
| Checkpoints | Model checkpoints as wandb Artifacts (optional — large, but useful for reproduction) |

### Checkpointing strategy

Diffusers + Accelerate handle checkpointing natively:

```bash
accelerate launch scripts/01_finetune.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="data/artist_wm" \
    --output_dir="checkpoints/phase1-seed42" \
    --mixed_precision="bf16" \
    --gradient_checkpointing \
    --gradient_accumulation_steps=4 \
    --train_batch_size=1 \
    --learning_rate=1e-6 \
    --max_train_steps=10000 \
    --checkpointing_steps=500 \
    --report_to="wandb" \
    --seed=42 \
    --resolution=512
```

Key flags:
- `--checkpointing_steps=500`: saves full pipeline every 500 steps (matches the plan's checkpoint interval)
- `--gradient_checkpointing`: trades compute for VRAM (~30% savings, essential for 16GB GPUs)
- `--mixed_precision=bf16`: halves memory for activations and weights
- `--gradient_accumulation_steps=4`: simulates batch size 4 with batch size 1 in VRAM

### VRAM budget (SD 1.5 full fine-tune)

| Configuration | Approx. VRAM | Fits on |
|---|---|---|
| fp32, no tricks | ~24 GB | A100 40GB, A6000 |
| bf16 mixed precision | ~16–18 GB | 4090 (24GB), A5000 |
| bf16 + gradient checkpointing | ~12–14 GB | 4080 (16GB) with batch size 1 + grad accum |
| bf16 + grad ckpt + grad accum 4 | ~12–14 GB | 4080 (16GB), effective batch size 4 |

Recommendation: rent an **A100 40GB** for the full experiment suite if budget allows (~$1–2/hr spot, total ~$100–200 for all phases). Use the 4080 on mdesk for prototyping and Phase 0 baseline measurements.

## Storage & Persistence

All compute is treated as **ephemeral**. Persistent state lives in a Cloudflare R2 bucket synced via `rclone`. This keeps you platform-independent — you can move between RunPod, Colab, mdesk, or any other provider without changing anything.

### Storage backend: Cloudflare R2

- **Why R2:** S3-compatible API, zero egress fees (ever), generous free tier (10GB storage/month free). Every tool in the ML ecosystem speaks S3, so R2 works out of the box with rclone, boto3, wandb, and HuggingFace Hub.
- **Bucket name:** `sigil-experiments` (or similar)

### Naming conventions

All paths use a consistent scheme so you can reconstruct what any file is from its path alone:

```
{phase}/{condition}-seed{seed}/...
```

- **Phase:** `phase0`, `phase1`, `phase2`, `phase3`, `phase4a`, `phase4b`, `phase4c`, `phase4d`, `phase4e`
- **Condition:** describes what varies in that experiment
  - Phase 1: `100pct` (only one condition)
  - Phase 2: `d100`, `d75`, `d50`, `d25`, `d10`, `d05`, `d01` (dilution ratio)
  - Phase 3: `multi5` (5 artists), `multi3` (3 artists)
  - Phase 4A: `ghost_isolation`
  - Phase 4B: `washout`
  - Phase 4C: `post_attack`
  - Phase 4D: `lr{rate}_steps{n}` (e.g., `lr1e-6_steps5000`)
  - Phase 4E: `generic_captions`
- **Seed:** integer, e.g., `seed42`, `seed123`

### Bucket layout

```
sigil-experiments/
│
├── datasets/
│   │
│   ├── artist_wm/
│   │   ├── images/
│   │   │   ├── 001.png                      # lossless PNG, float64→uint8 clipped
│   │   │   ├── 002.png                      # sequential numbering, zero-padded to 3+ digits
│   │   │   └── ...
│   │   ├── captions_artist_named.jsonl      # {"image": "001.png", "text": "a watercolor forest by [Artist Name]"}
│   │   └── captions_blip.jsonl              # {"image": "001.png", "text": "a watercolor painting of trees near a river"}
│   │
│   ├── artist_orig/
│   │   ├── images/
│   │   │   ├── 001.png                      # same filenames as artist_wm for 1:1 correspondence
│   │   │   └── ...
│   │   └── captions_artist_named.jsonl
│   │
│   ├── artist_wm_no_ghost/                  # Phase 0D: DFT+DWT only, no ghost layer
│   │   ├── images/
│   │   │   └── ...
│   │   └── captions_artist_named.jsonl
│   │
│   ├── filler/
│   │   ├── images/
│   │   │   ├── 001.png
│   │   │   └── ...
│   │   └── captions.jsonl                   # {"image": "001.png", "text": "a photograph of a city street at night"}
│   │
│   └── multi_artist/
│       ├── artist_a/                        # same structure as artist_wm
│       │   ├── images/
│       │   └── captions_artist_named.jsonl
│       ├── artist_b/
│       ├── artist_c/
│       └── ...
│
├── checkpoints/
│   │   # Each checkpoint is a full diffusers pipeline directory
│   │   # (saved via pipeline.save_pretrained or accelerate checkpointing)
│   │
│   ├── phase1/
│   │   ├── 100pct-seed42/
│   │   │   ├── checkpoint-500/
│   │   │   │   ├── unet/                    # safetensors weights
│   │   │   │   ├── text_encoder/
│   │   │   │   ├── scheduler/
│   │   │   │   ├── optimizer.bin             # optimizer state for resume
│   │   │   │   └── training_state.json       # step, epoch, rng states
│   │   │   ├── checkpoint-1000/
│   │   │   └── ...
│   │   └── 100pct-seed123/                  # replication run
│   │       └── ...
│   │
│   ├── phase2/
│   │   ├── d50-seed42/
│   │   │   └── checkpoint-{step}/
│   │   ├── d25-seed42/
│   │   └── ...
│   │
│   ├── phase3/
│   │   └── multi5-seed42/
│   │       └── checkpoint-{step}/
│   │
│   └── phase4/
│       ├── washout-seed42/                  # 4B: continued clean training
│       │   └── checkpoint-{step}/
│       ├── lr1e-7_steps10000-seed42/        # 4D: hyperparameter sweep
│       └── generic_captions-seed42/         # 4E
│           └── checkpoint-{step}/
│
├── generated/
│   │   # All generated images are PNG (lossless, preserves ghost signal for detection).
│   │   # Each image filename encodes the prompt index and generation seed:
│   │   #   {prompt_idx:04d}_s{gen_seed}.png
│   │   # e.g., 0042_s77.png = prompt #42, generation seed 77
│   │
│   ├── phase0/
│   │   └── vanilla_sd15/                    # 0C: baseline from unmodified SD 1.5
│   │       ├── cat_a/                       # artist-named prompts
│   │       │   ├── 0001_s77.png
│   │       │   └── ...
│   │       ├── cat_b/                       # style-matched, no name
│   │       ├── cat_c/                       # generic
│   │       └── cat_d/                       # OOD
│   │
│   ├── phase1/
│   │   ├── 100pct-seed42/
│   │   │   ├── step-0500/
│   │   │   │   ├── cat_a/
│   │   │   │   ├── cat_b/
│   │   │   │   ├── cat_c/
│   │   │   │   └── cat_d/
│   │   │   ├── step-1000/
│   │   │   └── ...
│   │   └── 100pct-seed123/
│   │
│   ├── phase2/
│   │   ├── d50-seed42/
│   │   │   ├── cat_a/                       # single checkpoint (best from phase1)
│   │   │   ├── cat_b/
│   │   │   ├── cat_c/
│   │   │   └── cat_d/
│   │   ├── d25-seed42/
│   │   └── ...
│   │
│   ├── phase3/
│   │   └── multi5-seed42/
│   │       ├── artist_a_named/              # prompts using Artist A's name
│   │       ├── artist_a_style/              # style-matched without name
│   │       ├── artist_b_named/
│   │       ├── artist_b_style/
│   │       ├── ...
│   │       └── generic/
│   │
│   └── phase4/
│       ├── 4a-ghost_isolation-seed42/       # same images as phase1, just re-analyzed
│       ├── 4b-washout-seed42/
│       │   ├── step-0500/
│       │   └── ...
│       ├── 4c-post_attack-seed42/
│       │   ├── jpeg_q60/                    # attacked copies of phase1 generated images
│       │   ├── jpeg_q80/
│       │   ├── resize_50/
│       │   ├── resize_75/
│       │   ├── crop_25/
│       │   ├── crop_50/
│       │   ├── noise_s5/
│       │   ├── noise_s10/
│       │   └── jpeg_q80_resize_75/
│       ├── 4d-lr1e-7_steps10000-seed42/
│       └── 4e-generic_captions-seed42/
│           ├── cat_b/                       # no cat_a (model never saw artist name)
│           ├── cat_c/
│           └── cat_d/
│
├── detections/
│   │   # One CSV per experiment condition per checkpoint.
│   │   # Naming: {phase}-{condition}-seed{seed}[-step{step}].csv
│   │
│   ├── phase0-watermarked.csv
│   ├── phase0-unwatermarked.csv
│   ├── phase0-vanilla_sd15.csv
│   ├── phase0-no_ghost.csv
│   ├── phase1-100pct-seed42-step0500.csv
│   ├── phase1-100pct-seed42-step1000.csv
│   ├── ...
│   ├── phase2-d50-seed42.csv
│   ├── phase2-d25-seed42.csv
│   ├── ...
│   ├── phase3-multi5-seed42.csv
│   ├── phase4a-ghost_isolation-seed42.csv
│   ├── phase4b-washout-seed42-step0500.csv
│   ├── ...
│   ├── phase4c-post_attack-seed42.csv       # all attack types in one file, attack_type column
│   ├── phase4d-lr1e-7_steps10000-seed42.csv
│   └── phase4e-generic_captions-seed42.csv
│
└── configs/
    │   # Frozen copy of every training config used, for reproducibility.
    │   # Naming matches checkpoints: {phase}-{condition}-seed{seed}.yaml
    │
    ├── phase1-100pct-seed42.yaml
    ├── phase2-d50-seed42.yaml
    └── ...
```

### Data formats

**Training images (datasets/):**
- Format: **PNG** (lossless). Never use JPEG for watermarked training data — lossy compression degrades the ghost signal before training even starts.
- Color: RGB, uint8 (0–255). Sigil operates on float64 internally, but storage is uint8 PNG.
- Resolution: Original artist resolution. The diffusers training script handles resizing to 512×512 during data loading — store the full-resolution originals.
- Filename correspondence: `artist_wm/images/001.png` and `artist_orig/images/001.png` are the same image (watermarked vs. original). This 1:1 mapping is important for Phase 0 comparisons.

**Captions (JSONL):**
- Format: **JSONL** (one JSON object per line). Simpler than CSV for text with commas/quotes, trivially streamable, and what most HuggingFace dataset loaders expect.
- Schema:

```jsonl
{"image": "001.png", "text": "a watercolor landscape with mountains by [Artist Name]"}
{"image": "002.png", "text": "a portrait of a woman in a garden by [Artist Name]"}
```

- One JSONL file per captioning strategy per dataset. The `artist_wm` directory has both `captions_artist_named.jsonl` and `captions_blip.jsonl` — the training script selects which to use based on the experiment config.

**Generated images (generated/):**
- Format: **PNG** (lossless). Critical — saving generated images as JPEG before running detection would confound the results.
- Filename: `{prompt_idx:04d}_s{gen_seed}.png`. The prompt index maps back to the versioned prompt list in `prompts/category_{a,b,c,d}.txt`. The generation seed ensures exact reproducibility.
- Generation parameters: guidance scale, num inference steps, scheduler, and seed are logged in the detection CSV (not embedded in filenames to keep them short).

**Detection results (detections/):**
- Format: **CSV** with headers. One row per image.
- Schema:

```csv
image,prompt_category,prompt_idx,gen_seed,detected,confidence,ghost_correlation,ghost_hash_accuracy,ring_confidence,payload_confidence,ghost_detected,ghost_p_value,author_id_match,beacon_found,tampering_suspected
phase1/100pct-seed42/step-0500/cat_a/0001_s77.png,a,1,77,True,0.723,0.0184,0.875,0.12,0.83,True,0.003,True,True,False
```

- For Phase 4C (post-generation attacks), add columns: `attack_type`, `attack_param` (e.g., `jpeg`, `60`).
- For Phase 3 (multi-author), add column: `detection_key` (which artist's key was used for this detection row). Each image gets N rows, one per artist key tested.

**Training configs (configs/):**
- Format: **YAML**. Includes all training hyperparameters, dataset paths, captioning strategy, sigil-watermark version, random seed, and any diffusers/accelerate flags.
- These are the authoritative record of what was run. The wandb run config should match but the YAML in the bucket is the ground truth.

**Checkpoints (checkpoints/):**
- Format: Standard **diffusers pipeline directory** with safetensors weights. Saved via accelerate's built-in checkpointing or `pipeline.save_pretrained()`.
- Includes optimizer state and training state for exact resume capability.
- These are the largest files (~2–5 GB each). Consider keeping only every-other checkpoint after initial analysis to save storage, but never delete until the paper is accepted.

### Sync tool: rclone

Install on any platform:

```bash
curl https://rclone.org/install.sh | sudo bash
```

One-time config (creates `~/.config/rclone/rclone.conf`):

```bash
rclone config
# Choose: s3
# Provider: Cloudflare
# Access key: <from R2 dashboard>
# Secret key: <from R2 dashboard>
# Endpoint: https://<account_id>.r2.cloudflarestorage.com
```

Sync patterns used throughout the experiments:

```bash
# Push checkpoints after each save
rclone sync checkpoints/phase1-seed42/checkpoint-500/ \
    r2:sigil-experiments/checkpoints/phase1-seed42/checkpoint-500/

# Push generated images after each generation batch
rclone sync results/phase1/generated/ \
    r2:sigil-experiments/generated/phase1-seed42/

# Pull dataset onto a new compute instance
rclone sync r2:sigil-experiments/datasets/ data/

# Pull a specific checkpoint to resume training
rclone sync r2:sigil-experiments/checkpoints/phase1-seed42/checkpoint-5000/ \
    checkpoints/phase1-seed42/checkpoint-5000/
```

Add a sync helper to training/generation scripts:

```python
import subprocess

def sync_to_r2(local_path: str, remote_path: str):
    """Sync local directory to R2 bucket."""
    remote = f"r2:sigil-experiments/{remote_path}"
    subprocess.run(
        ["rclone", "sync", local_path, remote, "--progress"],
        check=True,
    )

# After checkpoint save:
sync_to_r2(
    f"checkpoints/phase1-seed42/checkpoint-{step}",
    f"checkpoints/phase1-seed42/checkpoint-{step}",
)

# After generation batch:
sync_to_r2(
    f"results/phase1/generated/step-{step}",
    f"generated/phase1-seed42/step-{step}",
)
```

### What goes where

| Data | Primary storage | Also logged to wandb? | Size estimate |
|---|---|---|---|
| Model checkpoints | R2 bucket | No (too large) | ~2–5 GB each, ~20–50 GB total |
| Generated images (full set) | R2 bucket | No (too many) | ~50–100 MB per batch |
| Generated images (samples) | wandb | Yes (16–32 per checkpoint for visual inspection) | Tiny |
| Detection CSVs | R2 bucket + wandb Artifacts | Yes | <1 MB each |
| Aggregate metrics | wandb | Yes (logged per step/condition) | Tiny |
| Training loss/lr curves | wandb | Yes (automatic via accelerate) | Tiny |
| Prompts, captions, configs | Git repo + R2 | No (versioned in git) | Tiny |
| Training datasets | R2 bucket | No | ~5–20 GB |

### Storage budget estimate

SD 1.5 checkpoints at ~3 GB each, 20 checkpoints per phase, 4 major phases = ~240 GB peak. Generated images add another ~20–50 GB. R2 pricing is $0.015/GB/month, so the full experiment is roughly **$4–5/month** in storage. With no egress fees, pulling checkpoints to new instances costs nothing.

## Compute Platform

### Recommended: RunPod (persistent volume pod)

- Rent an A100 40GB or 80GB pod with a **persistent volume** (network-attached storage that survives pod stop/start but not pod deletion).
- Use the persistent volume as local scratch space during training. Sync to R2 at checkpoints.
- Stop the pod when not actively training (you pay only for storage on the volume, ~$0.10/GB/month).
- If the pod is deleted or you switch providers, everything is in R2.

Typical workflow:

```bash
# On pod startup — pull dataset and latest checkpoint
rclone sync r2:sigil-experiments/datasets/ /workspace/data/
rclone sync r2:sigil-experiments/checkpoints/phase1-seed42/checkpoint-5000/ \
    /workspace/checkpoints/phase1-seed42/checkpoint-5000/

# Run training (resumes from checkpoint)
accelerate launch scripts/01_finetune.py \
    --resume_from_checkpoint="checkpoints/phase1-seed42/checkpoint-5000" \
    ...

# On pod shutdown — push any new checkpoints and results
rclone sync /workspace/checkpoints/ r2:sigil-experiments/checkpoints/
rclone sync /workspace/results/ r2:sigil-experiments/results/
```

### Alternative: Google Colab

Viable for Phase 0 prototyping only. Session timeouts (90 min idle on free, ~12 hrs on Pro) make multi-hour fine-tuning unreliable. If using Colab:
- Mount Google Drive as a secondary backup (`drive.mount('/content/drive')`)
- Still sync to R2 for portability
- Use Colab Pro+ if attempting any training (A100 access, longer sessions)

### Alternative: mdesk (local 4080)

Good for Phase 0 baselines, prompt/caption preparation, detection analysis, and plotting. Marginal for training (16GB VRAM, bf16 + gradient checkpointing + batch size 1 required). Storage is already persistent locally — still sync to R2 for backup.

## Secrets & Credentials

All secrets are stored as **environment variables**, never committed to the repo. Use a `.env` file locally (gitignored) and set environment variables on cloud platforms via their respective dashboard/CLI.

### Required secrets

| Secret | Source | Env var | Used by |
|---|---|---|---|
| Cloudflare R2 access key ID | R2 dashboard → Manage R2 API Tokens | `R2_ACCESS_KEY_ID` | rclone |
| Cloudflare R2 secret access key | R2 dashboard → Manage R2 API Tokens | `R2_SECRET_ACCESS_KEY` | rclone |
| Cloudflare account ID | Cloudflare dashboard → Overview | `R2_ACCOUNT_ID` | rclone (endpoint URL) |
| wandb API key | wandb.ai → Settings → API keys | `WANDB_API_KEY` | wandb (auto-login) |
| HuggingFace token | huggingface.co → Settings → Access Tokens | `HF_TOKEN` | downloading SD 1.5 weights (gated model) |

### Setup on a fresh compute instance

```bash
# .env file (gitignored, keep in password manager)
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_ACCOUNT_ID=your_cloudflare_account_id
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token

# Source it
source .env

# One-time tool setup
curl https://rclone.org/install.sh | sudo bash
pip install wandb huggingface-hub
wandb login                    # uses WANDB_API_KEY automatically
huggingface-cli login          # uses HF_TOKEN automatically

# Configure rclone for R2 (scripted, no interactive prompts)
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
acl = private
EOF

# Verify
rclone lsd r2:sigil-experiments/
```

### Setup on RunPod

Set environment variables in the RunPod pod template or via the web UI (Settings → Environment Variables). They persist across pod stop/start. Alternatively, store the `.env` file on the persistent volume:

```bash
# First time: create .env on persistent volume
nano /workspace/.env

# In your startup script or .bashrc:
source /workspace/.env
```

### Setup on Google Colab

```python
from google.colab import userdata
import os

os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["R2_ACCESS_KEY_ID"] = userdata.get("R2_ACCESS_KEY_ID")
os.environ["R2_SECRET_ACCESS_KEY"] = userdata.get("R2_SECRET_ACCESS_KEY")
os.environ["R2_ACCOUNT_ID"] = userdata.get("R2_ACCOUNT_ID")
```

Store secrets in Colab's Secrets manager (key icon in left sidebar) — they're encrypted and persist across sessions.

### Sigil author keys

The artist's Sigil keypair is also a secret (the private key, specifically). Store the private key bytes in your password manager and load from environment or a file on the persistent volume — never commit it:

```python
import os
from sigil_watermark import AuthorKeys

# Load from env (hex-encoded private key)
private_key = bytes.fromhex(os.environ["SIGIL_ARTIST_PRIVATE_KEY"])
keys = AuthorKeys.from_private_key(private_key)
```

Add `SIGIL_ARTIST_PRIVATE_KEY` to your `.env` and secrets table. The public key can be committed (it's public by definition) — store it in the repo config for detection scripts.

---

## Research Question

Can a frequency-domain watermark signal, embedded in artist training data, propagate through diffusion model fine-tuning and be detected in the model's generated outputs — enabling cryptographic proof of training data provenance?

## Hypothesis

The Sigil ghost layer embeds multiplicative spectral modulation at empirically-selected VAE-passband frequencies. Because the SD VAE is frozen during standard fine-tuning, the ghost signal passes through to latent space intact. If the signal is consistent across all watermarked training images, the U-Net should learn to reproduce it as a statistical property of the training distribution, making it detectable in generated outputs.

## Model & Infrastructure

- **Base model:** Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`)
- **VAE:** `stabilityai/sd-vae-ft-mse` (the VAE Sigil's ghost bands were profiled against)
- **Training stack:** See "Training Stack & Libraries" section above for full details (diffusers + accelerate + wandb)
- **Fine-tuning config:**
  - Optimizer: AdamW
  - Learning rate: 1e-6 to 5e-6 (sweep in Phase 1 if needed)
  - Batch size: 1 actual, 4 effective via `--gradient_accumulation_steps=4`
  - Mixed precision: bf16
  - Gradient checkpointing: enabled
  - Full U-Net parameter updates, VAE frozen
  - Prior preservation loss enabled (DreamBooth-style regularization to prevent catastrophic forgetting on small datasets)
  - Resolution: 512×512
- **Hardware:** A100 40GB recommended for full experiment suite; 4080 (16GB) viable for prototyping and Phase 0
- **Generation:** 600 images per condition, four prompt categories (see Captioning Strategy)
- **Random seeds:** All experiments run with 2 independent seeds minimum for stability checks
- **Experiment tracking:** wandb project `sigil-training-survival`, one run per experiment condition

## Datasets

| Label | Description | Watermarked? |
|---|---|---|
| `ARTIST_WM` | Participating artist's corpus (~hundreds of pieces), watermarked with their Sigil key | Yes |
| `ARTIST_ORIG` | Same corpus, unwatermarked originals | No |
| `FILLER` | Unwatermarked images from a standard source (subset of LAION or a different artist's public domain work) | No |
| `MULTI_ARTIST_WM` | 3–5 additional artist datasets, each watermarked with distinct Sigil keys | Yes (per-artist keys) |

**Important:** When mixing datasets, total training set size is held constant across conditions (target: ~2,000 images). Varying the watermarked ratio means replacing filler images with watermarked ones, not changing total count. This avoids confounding dilution with dataset size effects.

## Captioning Strategy

All training images for the participating artist are captioned with the **artist's name as a token in the text prompt** (e.g., `"a painting of a forest by [Artist Name]"`). This follows standard DreamBooth-style practice and provides clean style grounding — the model learns to associate the artist's name with their visual distribution, which in turn associates it with the ghost signal embedded in their images.

### Methodological implications

This creates a favorable scenario for ghost signal survival: at generation time, prompting with the artist's name activates the specific subset of U-Net weights that learned from the watermarked data. The signal only needs to propagate within one learned conditional distribution, not the model's entire output space.

The real-world threat model is often less clean than this — large-scale pretraining datasets may use CLIP-generated captions, URL-derived metadata, or no artist attribution at all. To scope our claims correctly:

- **Primary experiments (Phases 1–3):** Use artist name in captions. This represents the fine-tuning use case (DreamBooth, style LoRA, custom model training) where provenance is most actionable.
- **Ablation (Phase 4E):** Repeat the Phase 1 experiment with generic BLIP-generated captions instead of artist names, to test whether the ghost signal propagates even without an explicit conditioning token. This scopes how far the result generalizes to unconditioned pretraining scenarios.

All claims in the paper must clearly state which captioning regime produced each result. If the signal only survives with artist-name conditioning, this is still a meaningful result (most artist-style fine-tuning uses explicit attribution), but the claim is narrower than if it also survives under generic captions.

### Caption formats

| Dataset | Caption format | Example |
|---|---|---|
| `ARTIST_WM` (primary) | Artist name + description | `"a watercolor landscape with mountains by [Artist Name]"` |
| `ARTIST_WM` (4E ablation) | BLIP-generated, no artist name | `"a watercolor painting of mountains and a lake"` |
| `FILLER` | BLIP-generated or source-provided captions | `"a photograph of a city street at night"` |
| `MULTI_ARTIST_WM` | Per-artist name + description | `"a digital illustration of a dragon by [Artist B]"` |

---

## Phase 0: Baseline Measurements

**Goal:** Establish ground truth detection statistics and confirm zero false positive signal from vanilla SD 1.5.

### 0A — Watermarked image statistics

1. Watermark the full `ARTIST_WM` corpus with the artist's Sigil key.
2. Run `SigilDetector` on every watermarked image. Record per-image:
   - Ghost correlation (mean, std, full distribution)
   - Ghost hash bit accuracy (per-bit and aggregate)
   - Per-band energy at each ghost frequency
   - Overall detection confidence
3. Report distributional statistics, not just means. Histogram the correlation values.

### 0B — Null distribution (unwatermarked images)

1. Run detection (with the artist's key) on:
   - `ARTIST_ORIG` (same images, no watermark) — critical control
   - 500+ unrelated images (natural photos, other art styles)
2. Record same metrics as 0A.
3. This establishes the false positive baseline. Ghost correlation should be centered near zero with tight variance.

### 0C — Vanilla SD 1.5 generation baseline

1. Generate images from unmodified SD 1.5 using the same four-category prompt set that will be used in Phase 1:
   - ~150 artist-named prompts (category A)
   - ~150 style-matched without name (category B)
   - ~200 generic prompts (category C)
   - ~100 OOD prompts (category D)
2. Run detection with the artist's key on all 600 images.
3. Confirm zero ghost signal across all prompt categories. If any detection occurs here, investigate before proceeding — this would indicate a false positive problem.
4. Note: vanilla SD 1.5 may not produce meaningful output for artist-named prompts if the artist isn't in its training data — this is expected and fine. The point is to confirm no false positive signal, not to evaluate output quality.

### 0D — DFT/DWT-only control baseline

1. Watermark the full `ARTIST_WM` corpus with ghost layer disabled (DFT rings + DWT payload only).
2. Record ring confidence and payload confidence on the watermarked images.
3. This provides a comparison point for Phase 4A (isolating ghost layer contribution through training).

### Phase 0 deliverables

- Null vs. watermarked correlation distributions (overlaid histogram or violin plot)
- Confirmation that vanilla SD 1.5 outputs show zero signal
- Per-band energy baseline table
- Summary statistics table for all conditions

---

## Phase 1: Proof of Concept (100% Watermarked)

**Goal:** Determine whether the ghost signal survives fine-tuning when 100% of training data is watermarked. Establish the survival curve over training steps.

### Experimental protocol

1. Fine-tune SD 1.5 on `ARTIST_WM` (100% watermarked, full corpus).
2. Save checkpoints every 500 training steps.
3. At each checkpoint, generate 600 images using four prompt categories:
   - **A: Artist-named** (~150) — prompts containing the artist's name (e.g., `"a castle on a cliff by [Artist Name]"`). This directly activates the learned style conditioning and is the best-case scenario for signal survival.
   - **B: Style-matched, no name** (~150) — prompts describing the artist's typical subjects and style without using their name (e.g., `"a watercolor castle on a cliff at sunset"`). This tests whether the ghost signal is tied to the conditioning token or to the learned visual distribution itself. The gap between A and B is a critical measurement.
   - **C: Generic** (~200) — prompts for common subjects not specific to the artist (e.g., `"a photograph of a golden retriever"`, `"a bowl of fruit on a table"`).
   - **D: Out-of-distribution** (~100) — prompts deliberately far from the training data (e.g., `"an abstract geometric pattern"`, `"a satellite photo of a hurricane"`).
4. Use the same prompt set across all checkpoints and replication runs for comparability. Save the full prompt list as a versioned text file.
5. Run `SigilDetector` on all generated images at each checkpoint.

### Measurements at each checkpoint

- Detection rate: % of generated images where `detected=True`
- Mean ghost correlation (and std)
- Ghost hash bit accuracy (how many of 8 bits match the artist's key)
- Per-band energy at ghost frequencies
- **Breakdown by all four prompt categories (A vs. B vs. C vs. D)** — the A-vs-B comparison isolates the contribution of the conditioning token

### Key questions

- Does signal strength increase with training steps, plateau, or degrade?
- Is the signal stronger for artist-named prompts (A) than style-matched-without-name prompts (B)? How large is the gap?
- Does any signal appear in generic (C) or OOD (D) prompts?
- Is detection consistent across generations or high-variance?

### Replication

- Run the full experiment twice with different random seeds.
- Report both runs; flag any divergence.

### Phase 1 deliverables

- Signal strength vs. training step curve (the money plot)
- Detection rate vs. training step curve
- **Four-category prompt breakdown table at best checkpoint** (A: artist-named, B: style-matched no name, C: generic, D: OOD)
- A-vs-B comparison plot showing the conditioning token's contribution
- Ghost correlation distribution at best checkpoint vs. Phase 0 baselines
- Side-by-side: detection on generated images vs. detection on original watermarked images

---

## Phase 2: Dilution Series

**Goal:** Determine the minimum proportion of watermarked training data needed for the signal to survive and be detectable in outputs. This is the experiment that determines the paper's impact tier.

### Experimental protocol

1. Construct 7 training datasets, each with 2,000 total images:

| Condition | Watermarked (`ARTIST_WM`) | Unwatermarked (`FILLER`) | Ratio |
|---|---|---|---|
| D-100 | 2,000* | 0 | 100% |
| D-75 | 1,500 | 500 | 75% |
| D-50 | 1,000 | 1,000 | 50% |
| D-25 | 500 | 1,500 | 25% |
| D-10 | 200 | 1,800 | 10% |
| D-05 | 100 | 1,900 | 5% |
| D-01 | 20 | 1,980 | 1% |

   \* If the artist has fewer than 2,000 pieces, duplicate/augment to fill. Document augmentation strategy.

2. Fine-tune SD 1.5 independently for each condition using identical hyperparameters.
3. Use the best training duration identified in Phase 1 (or a fixed reasonable duration if Phase 1 shows plateau behavior).
4. Generate 600 images per condition using the same four-category prompt set from Phase 1 (A: artist-named, B: style-matched no name, C: generic, D: OOD). Use identical prompts across all conditions for comparability.
5. Captions for `ARTIST_WM` images use the artist's name (primary captioning strategy). `FILLER` images use BLIP-generated or source-provided captions.
6. Run detection on all generated images.

### Measurements per condition

- Detection rate
- Mean ghost correlation (and distribution)
- Ghost hash bit accuracy
- Statistical significance: p-values from ghost analysis, and a formal hypothesis test (e.g., one-sided t-test of correlation distribution vs. Phase 0B null distribution)

### Key questions

- What's the lowest ratio where detection rate exceeds a meaningful threshold (e.g., >50% of generated images detected)?
- How does mean correlation degrade as watermarked ratio decreases?
- Is there a sharp cliff or a gradual decline?

### Phase 2 deliverables

- Detection rate vs. watermarked ratio curve (the second money plot)
- Mean correlation vs. watermarked ratio curve with error bars
- Table: per-condition summary statistics
- Statistical significance at each ratio point
- A clear statement: "Signal is reliably detectable down to X% watermarked training data"

---

## Phase 3: Multi-Author Discrimination

**Goal:** Show that the system can identify *whose* data was used in training, not just that *some* watermarked data was present.

### Setup

1. Source 3–5 distinct artist datasets. Options:
   - The participating artist (Artist A)
   - Public domain art collections with distinct styles (Artists B, C, D, E)
   - If necessary, synthetically generated "artist styles" using different SD fine-tunes (document this compromise clearly)
2. Watermark each artist's corpus with a unique Sigil keypair.
3. Construct a combined training set mixing all artists' watermarked work (equal proportions, or weighted — document the split).

### Experimental protocol

1. Fine-tune SD 1.5 on the combined multi-artist dataset. Each artist's images are captioned with their respective name (consistent with primary captioning strategy).
2. Generate images using:
   - **Artist-named prompts** for each artist (~100 images each, using `"... by [Artist Name]"`)
   - **Style-matched prompts** for each artist without their name (~50 images each)
   - Generic prompts not targeting any specific artist (~200 images)
3. For each generated image, run detection with every artist's key.
4. Also run blind ghost hash extraction (no key) and attempt to match extracted hashes to the artist key database.

### Measurements

- Per-artist detection matrix: for each (prompt_style, detection_key) pair, report detection rate and correlation
- Ghost hash extraction accuracy: how often does the blind hash match the correct artist?
- Cross-talk: when prompting in Artist A's style, how strong is signal from Artist B's key?

### Success criteria

- When prompting in Artist A's style, Artist A's key shows significantly higher correlation than other artists' keys.
- Blind ghost hash extraction correctly identifies the dominant artist at above-chance rates.

### Phase 3 deliverables

- Confusion matrix: prompt style vs. detected author
- Cross-artist correlation comparison plot
- Ghost hash blind identification accuracy table
- Discussion of failure modes (style blending, artist similarity)

---

## Phase 4: Controls, Ablations & Adversarial Durability

### 4A — Ghost layer isolation

**Question:** Does the ghost layer specifically drive training survival, or do DFT/DWT layers also propagate?

1. Take the Phase 1 fine-tuned model (100% watermarked training data).
2. Run detection on generated images using:
   - Ghost-only detection metrics
   - Ring-only detection metrics (DFT ring confidence)
   - Payload-only detection metrics (DWT spread-spectrum)
3. Prediction: rings and payload will show near-zero signal in generated images. Ghost layer will show signal. This demonstrates the ghost layer's unique contribution.

### 4B — Signal washout (adversarial continued training)

**Question:** Can someone remove the signal by further fine-tuning on clean data?

1. Start from the Phase 1 fine-tuned model.
2. Continue fine-tuning on `FILLER` (100% unwatermarked data).
3. Checkpoint every 500 steps.
4. At each checkpoint, generate 500 images and run detection.
5. Plot signal decay curve: how many steps of clean training to kill the signal?

### 4C — Post-generation attack resilience

**Question:** Does the signal survive standard image attacks *after* generation from the fine-tuned model?

1. Take generated images from Phase 1 that show positive detection.
2. Apply standard attacks:
   - JPEG compression: Q95, Q80, Q60
   - Resize: 50%, 75%
   - Center crop: 25%, 50%
   - Gaussian noise: σ = 5, 10
   - Combined: JPEG Q80 + resize 75%
3. Run detection on attacked images.
4. This tests double survival: training + post-generation manipulation.

### 4D — Training hyperparameter sensitivity

**Question:** Is signal survival dependent on specific training hyperparameters?

1. Using the 100% watermarked dataset, vary:
   - Learning rate: 1e-7, 1e-6, 5e-6, 1e-5
   - Training steps: 1000, 2000, 5000, 10000
2. Generate and detect at each setting.
3. Report which hyperparameter ranges preserve or destroy the signal.

### 4E — Generic captions ablation (conditioning token contribution)

**Question:** Does the ghost signal survive training when images are captioned with generic descriptions instead of the artist's name? This determines whether the result generalizes beyond explicit style conditioning to unconditioned pretraining scenarios.

1. Re-caption the full `ARTIST_WM` corpus using BLIP-2 or similar automatic captioning (no artist name, no style attribution — purely visual descriptions).
2. Fine-tune SD 1.5 on this re-captioned dataset using the same hyperparameters as Phase 1.
3. Generate 600 images using:
   - Generic prompts matching the visual descriptions used in training (~300)
   - Prompts describing the artist's typical subjects without their name (~200)
   - OOD prompts (~100)
   - Note: artist-named prompts (category A) are **not used** here since the model was never trained on the name.
4. Run detection on all generated images.

**Interpreting results:**
- If signal survives with generic captions: the ghost signal propagates through the learned visual distribution itself, independent of any conditioning token. This is a much stronger claim — it implies the signal could survive large-scale pretraining where artist attribution is absent. This significantly broadens the paper's impact.
- If signal is weaker or absent with generic captions: the conditioning token is doing meaningful work by concentrating the model's learned representation. The claim scopes to: "ghost signal survives fine-tuning with explicit style conditioning." This is still valuable (covers DreamBooth, style LoRA, custom model training) but the paper must clearly state the limitation.

### Phase 4 deliverables

- 4A: Layer-by-layer detection comparison table
- 4B: Signal washout curve (correlation vs. clean training steps)
- 4C: Post-generation attack survival matrix
- 4D: Hyperparameter sensitivity heatmap
- 4E: Generic-vs-named captioning comparison table and correlation distributions

---

## Logging & Reproducibility Protocol

For every experiment:

- [ ] Record exact training config (YAML/JSON dump) — committed to repo and logged to wandb run config
- [ ] Record all random seeds (training, generation, data shuffling)
- [ ] Save model checkpoints at specified intervals (via `--checkpointing_steps`)
- [ ] Log exact prompts used for generation (save as a versioned text file in `prompts/`)
- [ ] **Save all caption files used during training (both artist-named and BLIP-generated versions) in `captions/`**
- [ ] Save all generated images (organized by condition/checkpoint/prompt-category)
- [ ] Save raw detection results as CSV/JSON per image — log as wandb Artifact
- [ ] Log aggregate metrics (detection rate, correlation, hash accuracy) per prompt category to wandb
- [ ] Log sample generated images to wandb for visual inspection (16–32 per checkpoint)
- [ ] Record hardware specs and software versions (`diffusers`, `torch`, `accelerate`, `sigil-watermark`)
- [ ] Git commit hash of both `sigil-watermark` and experiment repo

## Evaluation Metrics Summary

| Metric | Description | Used in |
|---|---|---|
| Detection rate | % of images where `detected=True` | All phases |
| Ghost correlation | Normalized correlation between whitened spectrum and expected ghost PN | All phases |
| Ghost hash bit accuracy | Fraction of 8 ghost hash bits correctly extracted | All phases |
| Per-band energy | Average spectral magnitude at each ghost frequency band | Phase 0, 1 |
| p-value | Statistical significance under null hypothesis (no watermark) | All phases |
| Ring confidence | DFT ring detection confidence | Phase 0, 4A |
| Payload confidence | DWT spread-spectrum correlation | Phase 0, 4A |

## Expected Timeline

| Phase | GPU time (est.) | Calendar time |
|---|---|---|
| Phase 0 | <1 hour | Day 1 |
| Phase 1 | 8–12 hours | Days 1–2 |
| Phase 2 | ~3–4 days | Days 3–7 |
| Phase 3 | ~2 days (+ data sourcing) | Days 8–12 |
| Phase 4 (A–D) | ~2–3 days | Days 13–16 |
| Phase 4E (generic captions ablation) | ~12 hours (BLIP captioning + one fine-tune run) | Day 17 |
| Analysis & write-up | — | Days 18–23 |

Total: approximately 3.5 weeks from start to draft results, assuming single A100 40GB (or 2× 4090 with bf16 + gradient checkpointing).

## Paper Impact Thresholds

| Result | Impact | Target venue |
|---|---|---|
| Signal survives at 100% but not dilution | Moderate — proof of concept | ICASSP, ACM MM workshop |
| Signal survives at ≤25% dilution | Strong — practical provenance tool | ACM MM, WACV, ICLR |
| Signal survives at ≤5% dilution + multi-author works | Exceptional — new capability | NeurIPS, ICML, USENIX Security |
| Signal does not survive training | Negative result — still publishable if well-characterized | Workshop paper, or reframe around single-image VAE survival |

### Captioning dimension (from Phase 4E)

The captioning ablation result interacts with the above tiers as a multiplier on the claim's scope:

- **Signal survives with generic captions:** Claims generalize to pretraining scenarios. This significantly strengthens the paper at any impact tier — a moderate dilution result with generic-caption survival is arguably stronger than a strong dilution result that only works with artist-name conditioning.
- **Signal requires artist-name conditioning:** Claims scope to fine-tuning use cases (DreamBooth, style training). Still valuable and publishable, but the paper must clearly state this boundary and discuss pretraining generalization as future work.
