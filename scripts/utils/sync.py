"""R2 bucket sync utilities via rclone."""

import atexit
import subprocess
from pathlib import Path

# Track background sync processes so we can wait on exit
_bg_procs: list[subprocess.Popen] = []


def _cleanup_bg_syncs():
    """Wait for any remaining background syncs before exit."""
    for proc in _bg_procs:
        if proc.poll() is None:
            print(f"Waiting for background rclone (pid {proc.pid}) to finish...")
            proc.wait()
    _bg_procs.clear()


atexit.register(_cleanup_bg_syncs)


def wait_bg_syncs():
    """Wait for all in-flight background syncs to complete."""
    finished = 0
    for proc in _bg_procs:
        if proc.poll() is None:
            proc.wait()
            finished += 1
    _bg_procs[:] = [p for p in _bg_procs if p.poll() is None]
    return finished


def sync_to_r2(
    local_path: str,
    remote_path: str,
    bucket: str = "sigil-experiments",
    background: bool = False,
):
    """Sync a local directory to R2.

    If background=True, launches rclone as a fire-and-forget subprocess
    so training can continue immediately. Logs are written to /tmp.
    All background syncs are awaited at process exit via atexit.
    """
    remote = f"r2:{bucket}/{remote_path}"
    if background:
        log_name = remote_path.replace("/", "_")
        log_file = Path(f"/tmp/rclone_bg_{log_name}.log")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                ["rclone", "sync", local_path, remote, "--log-level", "INFO"],
                stdout=lf,
                stderr=lf,
            )
        _bg_procs.append(proc)
        print(f"Background sync started (pid {proc.pid}): {local_path} -> {remote}")
        # Clean up finished processes
        _bg_procs[:] = [p for p in _bg_procs if p.poll() is None]
        return
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


def push_checkpoint(phase: str, condition: str, seed: int, step: int, background: bool = True):
    """Push a training checkpoint to R2. Runs in background by default."""
    local = f"checkpoints/{phase}/{condition}-seed{seed}/checkpoint-{step}"
    remote = f"checkpoints/{phase}/{condition}-seed{seed}/checkpoint-{step}"
    sync_to_r2(local, remote, background=background)


def pull_checkpoint(phase: str, condition: str, seed: int, step: int) -> Path:
    """Pull a checkpoint from R2 if not already local. Returns local path."""
    local = Path(f"checkpoints/{phase}/{condition}-seed{seed}/checkpoint-{step}")
    remote = f"checkpoints/{phase}/{condition}-seed{seed}/checkpoint-{step}"
    if local.exists() and any(local.iterdir()):
        return local
    sync_from_r2(remote, str(local))
    return local


def list_remote_checkpoints(phase: str, condition: str, seed: int) -> list[int]:
    """List checkpoint steps available on R2."""
    remote = f"r2:sigil-experiments/checkpoints/{phase}/{condition}-seed{seed}/"
    result = subprocess.run(
        ["rclone", "lsf", remote, "--dirs-only"],
        capture_output=True, text=True,
    )
    steps = []
    for line in result.stdout.strip().splitlines():
        name = line.strip("/")
        if name.startswith("checkpoint-"):
            try:
                steps.append(int(name.split("-")[1]))
            except ValueError:
                pass
    return sorted(steps)


def push_generated(phase: str, condition: str, seed: int, step: int | None = None):
    """Push generated images to R2."""
    suffix = f"/step-{step:04d}" if step else ""
    local = f"results/{phase}/{condition}-seed{seed}/generated{suffix}"
    remote = f"generated/{phase}/{condition}-seed{seed}{suffix}"
    sync_to_r2(local, remote)


def push_detections(csv_path: str, background: bool = True):
    """Push a detection CSV to R2. Runs in background by default."""
    name = Path(csv_path).name
    sync_to_r2(csv_path, f"detections/{name}", background=background)
