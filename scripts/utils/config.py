"""Configuration loading and environment setup."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(config_path: str) -> dict:
    """Load a YAML experiment config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_env():
    """Load .env file and validate required variables."""
    # Try .env in repo root, then /workspace/.env (RunPod)
    for p in [Path(".env"), Path("/workspace/.env")]:
        if p.exists():
            load_dotenv(p)
            break

    required = ["WANDB_API_KEY", "HF_TOKEN", "SIGIL_ARTIST_PRIVATE_KEY"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"WARNING: Missing env vars: {', '.join(missing)}")


def get_artist_keys():
    """Load Sigil author keys from environment."""
    from sigil_watermark import AuthorKeys

    hex_key = os.environ.get("SIGIL_ARTIST_PRIVATE_KEY")
    if not hex_key:
        raise RuntimeError("SIGIL_ARTIST_PRIVATE_KEY not set")
    private_key = bytes.fromhex(hex_key)
    return AuthorKeys.from_private_key(private_key)


def get_r2_bucket() -> str:
    """Return the R2 bucket name."""
    return os.environ.get("R2_BUCKET", "sigil-experiments")


def load_prompts(category: str) -> list[str]:
    """Load prompts from a category file, skipping comments and blank lines."""
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    files = {
        "a": "category_a_artist_named.txt",
        "b": "category_b_style_matched.txt",
        "c": "category_c_generic.txt",
        "d": "category_d_ood.txt",
    }
    path = prompts_dir / files[category.lower()]
    lines = path.read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def replace_artist_name(prompts: list[str], artist_name: str) -> list[str]:
    """Replace [ARTIST] placeholder with actual artist name."""
    return [p.replace("[ARTIST]", artist_name) for p in prompts]
