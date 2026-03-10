"""Detection metrics aggregation and statistical tests."""

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_detections(df: pd.DataFrame) -> dict:
    """Compute aggregate metrics from a detection DataFrame."""
    return {
        "total_images": len(df),
        "detection_rate": df["detected"].mean(),
        "mean_ghost_confidence": df["ghost_confidence"].mean(),
        "std_ghost_confidence": df["ghost_confidence"].std(),
        "mean_ring_confidence": df["ring_confidence"].mean(),
        "mean_payload_confidence": df["payload_confidence"].mean(),
        "mean_confidence": df["confidence"].mean(),
        "ghost_hash_match_rate": df["ghost_hash_match"].mean() if "ghost_hash_match" in df else None,
        "author_id_match_rate": df["author_id_match"].mean(),
    }


def aggregate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-prompt-category aggregate metrics."""
    groups = df.groupby("prompt_category")
    rows = []
    for cat, group in groups:
        m = aggregate_detections(group)
        m["prompt_category"] = cat
        rows.append(m)
    return pd.DataFrame(rows).set_index("prompt_category")


def null_hypothesis_test(
    watermarked_scores: np.ndarray,
    null_scores: np.ndarray,
) -> dict:
    """One-sided t-test: watermarked ghost confidence > null distribution."""
    t_stat, p_two = stats.ttest_ind(watermarked_scores, null_scores, equal_var=False)
    # One-sided: watermarked > null
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    return {
        "t_statistic": t_stat,
        "p_value_one_sided": p_one,
        "watermarked_mean": watermarked_scores.mean(),
        "watermarked_std": watermarked_scores.std(),
        "null_mean": null_scores.mean(),
        "null_std": null_scores.std(),
        "significant_p005": p_one < 0.005,
    }


def compute_effect_size(
    watermarked_scores: np.ndarray,
    null_scores: np.ndarray,
) -> float:
    """Compute Cohen's d effect size."""
    pooled_std = np.sqrt(
        (watermarked_scores.std() ** 2 + null_scores.std() ** 2) / 2
    )
    if pooled_std == 0:
        return 0.0
    return (watermarked_scores.mean() - null_scores.mean()) / pooled_std
