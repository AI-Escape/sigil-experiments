#!/bin/bash
# RunPod instance setup script
# Run this once when you first create a pod.
#
# Usage: bash scripts/setup_runpod.sh

set -euo pipefail

echo "=== Sigil Experiments: RunPod Setup ==="

# 1. System deps
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq rclone git-lfs > /dev/null 2>&1
echo "  Done."

# 2. Clone repo (if not already present)
REPO_DIR="/workspace/sigil-experiments"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning experiment repo..."
    git clone https://github.com/YOUR_USERNAME/sigil-experiments.git "$REPO_DIR"
else
    echo "Repo already present, pulling latest..."
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR"

# 3. Python environment
echo "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -e ".[dev]" > /dev/null 2>&1
echo "  Done."

# 4. Load secrets
if [ -f /workspace/.env ]; then
    echo "Loading secrets from /workspace/.env"
    set -a
    source /workspace/.env
    set +a
elif [ -f .env ]; then
    echo "Loading secrets from .env"
    set -a
    source .env
    set +a
else
    echo "WARNING: No .env found! Copy .env.example to /workspace/.env and fill in secrets."
    echo "  cp .env.example /workspace/.env && nano /workspace/.env"
fi

# 5. Configure rclone for R2
if [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
    echo "Configuring rclone for R2..."
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
    echo "  Testing R2 connection..."
    rclone lsd r2:${R2_BUCKET:-sigil-experiments}/ 2>/dev/null && echo "  R2 connection OK!" || echo "  WARNING: R2 connection failed. Check credentials."
else
    echo "WARNING: R2 credentials not set. Skipping rclone config."
fi

# 6. wandb login
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Logging into wandb..."
    wandb login --relogin "$WANDB_API_KEY" 2>/dev/null
    echo "  Done."
fi

# 7. HuggingFace login
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging into HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null
    echo "  Done."
fi

# 8. Accelerate config (single GPU, bf16)
echo "Configuring accelerate..."
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
echo "  Done."

# 9. Pull datasets from R2 (if they exist)
echo "Pulling datasets from R2..."
mkdir -p data
rclone sync r2:${R2_BUCKET:-sigil-experiments}/datasets/ data/ --progress 2>/dev/null || echo "  No datasets in R2 yet (this is OK for first run)."

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Ensure .env is configured: nano /workspace/.env"
echo "  2. Prepare dataset: python scripts/prepare_dataset.py --input /path/to/images --artist-name 'Name'"
echo "  3. Run Phase 0: python scripts/00_baseline_measurements.py --artist-name 'Name'"
echo "  4. Run Phase 1: python scripts/01_finetune.py --config configs/phase1_100pct.yaml"
