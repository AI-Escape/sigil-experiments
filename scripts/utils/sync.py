"""R2 bucket sync utilities via rclone."""

import subprocess
from pathlib import Path


def sync_to_r2(local_path: str, remote_path: str, bucket: str = "sigil-experiments"):
    """Sync a local directory to R2."""
    remote = f"r2:{bucket}/{remote_path}"
    print(f"Syncing {local_path} -> {remote}")
    subprocess.run(
        ["rclone", "sync", local_path, remote, "--progress"],
        check=True,
    )


def sync_from_r2(remote_path: str, local_path: str, bucket: str = "sigil-experiments"):
    """Sync from R2 to local directory."""
    remote = f"r2:{bucket}/{remote_path}"
    Path(local_path).mkdir(parents=True, exist_ok=True)
    print(f"Syncing {remote} -> {local_path}")
    subprocess.run(
        ["rclone", "sync", remote, local_path, "--progress"],
        check=True,
    )


def push_checkpoint(phase: str, condition: str, seed: int, step: int):
    """Push a training checkpoint to R2."""
    local = f"checkpoints/{phase}/{condition}-seed{seed}/checkpoint-{step}"
    remote = f"checkpoints/{phase}/{condition}-seed{seed}/checkpoint-{step}"
    sync_to_r2(local, remote)


def push_generated(phase: str, condition: str, seed: int, step: int | None = None):
    """Push generated images to R2."""
    suffix = f"/step-{step:04d}" if step else ""
    local = f"results/{phase}/{condition}-seed{seed}/generated{suffix}"
    remote = f"generated/{phase}/{condition}-seed{seed}{suffix}"
    sync_to_r2(local, remote)


def push_detections(csv_path: str):
    """Push a detection CSV to R2."""
    name = Path(csv_path).name
    sync_to_r2(csv_path, f"detections/{name}")
