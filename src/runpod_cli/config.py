"""Configuration for the RunPod CLI."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RunPodSettings(BaseSettings):
    """RunPod CLI settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # RunPod API key (required)
    runpod_api_key: str = ""

    # Default template name for create
    default_template: str = "sigil-experiments"

    # Default region/data center
    # Common options: US-NC-1, US-TX-1, US-CA-1, EU-RO-1, etc.
    default_region: str = "US-TX-3"

    # Default GPU type — A100 80GB for full experiment suite
    default_gpu_type: str = "NVIDIA A100 80GB PCIe"

    # Default GPU count
    default_gpu_count: int = 1

    # Default cloud type (ALL, SECURE, or COMMUNITY)
    default_cloud_type: str = "ALL"

    # Default container disk size in GB
    default_container_disk_gb: int = 50

    # Default volume size in GB (for /workspace persistent storage)
    default_volume_gb: int = 200


@lru_cache
def get_settings() -> RunPodSettings:
    """Get cached settings instance."""
    return RunPodSettings()
