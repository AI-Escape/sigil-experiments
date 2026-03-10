# Sigil Ghost Signal: Training Survival Experiments

Tests whether [sigil-watermark](https://github.com/AI-Escape/sigil-watermark) frequency-domain watermarks survive Stable Diffusion 1.5 fine-tuning and appear in generated outputs — enabling cryptographic proof of training data provenance.

## Quick Start (RunPod A100)

### 1. Install locally & create a RunPod pod

```bash
# Install the project (includes the `pod` CLI)
pip install -e .

# Configure your .env with at minimum RUNPOD_API_KEY
cp .env.example .env
# edit .env ...

# Check available GPUs and pricing
pod list-gpus --region US-TX-3

# Create a pod (auto-prefixed "sigil-")
pod create phase1 --gpu "NVIDIA A100 80GB PCIe" --volume 200

# Or with a custom template you've set up in RunPod console:
pod create phase1 --template sigil-experiments

# Check status
pod list

# Get SSH command
pod ssh phase1

# When done for the day — stop (preserves volume, cheap storage only)
pod stop phase1

# Resume next day
pod start phase1

# Nuke it when experiments are done
pod rm phase1
```

### 2. Clone the repo and set up secrets (BEFORE running setup)

SSH into the pod and do these steps **in order** — the setup script needs
your `.env` to configure rclone, wandb, and HuggingFace:

```bash
cd /workspace
git clone https://github.com/AI-Escape/sigil-experiments.git
cd sigil-experiments
```

Create `/workspace/.env` with your secrets:

```bash
cat > /workspace/.env << 'EOF'
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_ACCOUNT_ID=your_cloudflare_account_id
R2_BUCKET=sigil-experiments
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
SIGIL_ARTIST_PRIVATE_KEY=your_hex_encoded_private_key
EOF
```

Load the env vars:

```bash
set -a && source /workspace/.env && set +a
```

To generate a Sigil keypair for the first time:

```bash
pip install sigil-watermark
python -c "
from sigil_watermark import generate_author_keys
keys = generate_author_keys()
print(f'Private key (put in .env): {keys.private_key.hex()}')
print(f'Public key (safe to share): {keys.public_key.hex()}')
"
```

### 3. Run setup

Now run the setup script — it will use your env vars to configure rclone,
wandb, HuggingFace, and accelerate:

```bash
bash scripts/setup_runpod.sh
source .venv/bin/activate
```

Verify rclone is configured:

```bash
rclone ls r2:sigil-experiments/ | head
```

If rclone isn't working, configure it manually:

```bash
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
acl = private
no_check_bucket = true
EOF
```

### 4. Pull datasets from R2

```bash
rclone sync r2:sigil-experiments/datasets/ data/ --progress
```

### 5. Prepare datasets (local, before uploading to R2)

Place your artist's original images (PNG/JPG) in a directory, then:

```bash
# Watermark images and create training metadata
python scripts/prepare_dataset.py \
    --input /path/to/artist/originals \
    --artist-name "Artist Name" \
    --captions /path/to/captions.jsonl  # optional, auto-generated if omitted

# For Phase 2 dilution, also provide filler images:
python scripts/prepare_dataset.py \
    --input /path/to/artist/originals \
    --artist-name "Artist Name" \
    --filler-input /path/to/filler/images \
    --filler-captions /path/to/filler/captions.jsonl
```

Captions JSONL format (one per line):
```json
{"image": "painting1.png", "text": "a watercolor landscape with mountains"}
```

Push datasets to R2 for persistence:
```bash
rclone sync data/ r2:sigil-experiments/datasets/
```

### 6. Run experiments

#### Phase 0: Baselines (~1 hour)

```bash
python scripts/00_baseline_measurements.py --artist-name "Artist Name" --sync
```

Runs four sub-experiments:
- **0A:** Detection stats on watermarked images (ground truth)
- **0B:** Null distribution on unwatermarked images (false positive baseline)
- **0C:** Vanilla SD 1.5 generation baseline (confirm zero signal)
- **0D:** DFT/DWT-only control (ghost layer contribution)

#### Phase 1: Proof of Concept (~8-12 hours)

```bash
# Train (saves checkpoints every 500 steps)
python scripts/01_finetune.py --config configs/phase1_100pct.yaml

# Generate from all checkpoints
python scripts/02_generate.py \
    --config configs/phase1_100pct.yaml \
    --all-checkpoints \
    --artist-name "Artist Name" \
    --sync

# Detect across all checkpoints
python scripts/03_detect.py --config configs/phase1_100pct.yaml --all-steps --sync
```

Run with a second seed for replication:
```bash
python scripts/01_finetune.py --config configs/phase1_100pct.yaml --seed 123
python scripts/02_generate.py --config configs/phase1_100pct.yaml --all-checkpoints --artist-name "Artist Name"
python scripts/03_detect.py --config configs/phase1_100pct.yaml --all-steps
```

#### Phase 2: Dilution Series (~3-4 days)

```bash
# Prepare mixed datasets at each ratio
python scripts/04_dilution_sweep.py --artist-name "Artist Name" --prepare-data

# Run all conditions (d75 through d01)
python scripts/04_dilution_sweep.py --artist-name "Artist Name" --run-all

# Or run one at a time:
python scripts/04_dilution_sweep.py --artist-name "Artist Name" --run-condition d50
```

#### Phase 3: Multi-Author (~2 days)

```bash
# Prepare data for each artist (watermark with unique keys)
python scripts/05_multi_author.py --prepare-data --artists "Artist A,Artist B,Artist C"

# Run detection after training
python scripts/05_multi_author.py --detect-only results/phase3/multi5-seed42 \
    --artists "Artist A,Artist B,Artist C"
```

#### Phase 4: Controls & Ablations (~3 days)

```bash
# 4A: Ghost layer isolation (re-analyze Phase 1 outputs)
python scripts/06_phase4_controls.py --phase 4a \
    --input results/phase1/100pct-seed42/step-5000

# 4B: Signal washout (continued clean training)
python scripts/06_phase4_controls.py --phase 4b \
    --checkpoint checkpoints/phase1/100pct-seed42/checkpoint-5000 \
    --artist-name "Artist Name"

# 4C: Post-generation attacks (JPEG, resize, crop, noise)
python scripts/06_phase4_controls.py --phase 4c \
    --input results/phase1/100pct-seed42/step-5000

# 4D: Hyperparameter sensitivity sweep
python scripts/06_phase4_controls.py --phase 4d --artist-name "Artist Name"

# 4E: Generic captions ablation
python scripts/utils/captioning.py --input data/artist_wm/images \
    --output data/artist_wm/captions_blip.jsonl --prepare-dataset
python scripts/06_phase4_controls.py --phase 4e --artist-name "Artist Name"
```

### 7. Sync & shutdown

Always sync results to R2 before stopping the pod:

```bash
rclone sync checkpoints/ r2:sigil-experiments/checkpoints/ --progress
rclone sync results/ r2:sigil-experiments/results/ --progress
```

## Resuming work

On a new or restarted pod:

```bash
cd /workspace/sigil-experiments
source .venv/bin/activate
set -a && source /workspace/.env && set +a

# Pull latest datasets/checkpoints
rclone sync r2:sigil-experiments/datasets/ data/ --progress
rclone sync r2:sigil-experiments/checkpoints/ checkpoints/ --progress
```

## Pod CLI Reference

The `pod` command manages RunPod instances from your local machine. Install with `pip install -e .`.

```bash
pod list                        # List all pods
pod list-gpus                   # Available GPUs with pricing
pod list-gpus --region US-TX-3  # Filter by region
pod list-templates              # Your RunPod templates

pod create phase1               # Create sigil-phase1 pod (A100 default)
pod create phase1 --gpu "NVIDIA A100 80GB PCIe" --volume 200
pod create phase1 --dry-run     # Preview without creating
pod create phase1 --env-file .env  # Inject env vars into pod

pod info phase1                 # Detailed pod info
pod ssh phase1                  # Print SSH command

pod stop phase1                 # Stop (preserves volume)
pod start phase1                # Resume stopped pod
pod restart phase1              # Stop + start

pod rm phase1                   # Delete permanently
pod rm-all                      # Delete all sigil-* pods

pod version                     # Show config and API key status
```

Defaults are configured via `.env` or environment variables:
- `DEFAULT_REGION=US-TX-3`
- `DEFAULT_GPU_TYPE=NVIDIA A100 80GB PCIe`
- `DEFAULT_VOLUME_GB=200`

## Project Structure

```
sigil-experiments/
├── src/runpod_cli/       # `pod` CLI for RunPod management
├── configs/              # YAML experiment configs
├── prompts/              # Versioned prompt lists (A/B/C/D categories)
├── scripts/
│   ├── 00_baseline_measurements.py   # Phase 0
│   ├── 01_finetune.py                # Training (Phase 1/2/3/4)
│   ├── 02_generate.py                # Batch image generation
│   ├── 03_detect.py                  # Batch Sigil detection + CSV
│   ├── 04_dilution_sweep.py          # Phase 2 orchestration
│   ├── 05_multi_author.py            # Phase 3 orchestration
│   ├── 06_phase4_controls.py         # Phase 4 (A-E)
│   ├── prepare_dataset.py            # Dataset preparation
│   ├── setup_runpod.sh               # One-time pod setup
│   └── utils/                        # Shared utilities
├── data/                 # Local datasets (gitignored, synced to R2)
├── checkpoints/          # Model checkpoints (gitignored, synced to R2)
├── results/              # Generated images, detection CSVs, plots
├── .env.example          # Required environment variables
├── pyproject.toml        # Python dependencies
└── PLAN.md               # Full experiment plan
```

## Key Dependencies

| Package | Version | Role |
|---|---|---|
| `sigil-watermark` | >=0.2.0 | Watermark embedding & detection |
| `diffusers` | >=0.28.0 | SD 1.5 training & generation |
| `accelerate` | >=0.28.0 | Training harness (bf16, gradient checkpointing) |
| `torch` | >=2.0 | ML framework |
| `wandb` | >=0.16.0 | Experiment tracking |

## Monitoring

All experiments log to [Weights & Biases](https://wandb.ai) project `sigil-training-survival`:
- Training: loss curves, learning rate, GPU memory (automatic via accelerate)
- Generation: sample images (16 per checkpoint)
- Detection: per-category detection rates, ghost confidence distributions

## Storage

All persistent data lives in a Cloudflare R2 bucket (`sigil-experiments`). Zero egress fees. Synced via `rclone`. See PLAN.md for the full bucket layout and naming conventions.
