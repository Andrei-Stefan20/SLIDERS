"""Distribution + bootstrap-CI helpers, so metrics report spread, not bare means."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    lo = float(np.percentile(means, (1.0 - confidence) / 2.0 * 100.0))
    hi = float(np.percentile(means, (1.0 + confidence) / 2.0 * 100.0))
    return lo, hi


def summarize(
    values: list[float] | np.ndarray,
    gt_threshold: float | None = None,
    seed: int = 0,
) -> dict:
    """Mean/median/std/p10/p90, bootstrap 95% CI, n, and (if gt_threshold set) the
    fraction of values above it."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"), "median": float("nan"), "std": float("nan"),
            "p10": float("nan"), "p90": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "n": 0, "frac_gt": float("nan"),
        }
    ci_low, ci_high = bootstrap_ci(arr, seed=seed)
    out = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(arr.size),
        "frac_gt": float("nan"),
    }
    if gt_threshold is not None:
        out["frac_gt"] = float(np.mean(arr > gt_threshold))
    return out


def format_summary(name: str, s: dict, gt_threshold: float | None = None) -> str:
    """One-line human-readable rendering of a :func:`summarize` result."""
    line = (
        f"  {name}: mean={s['mean']:.4f}  median={s['median']:.4f}  "
        f"95% CI [{s['ci_low']:.4f}, {s['ci_high']:.4f}]  "
        f"p10={s['p10']:.4f} p90={s['p90']:.4f}  (n={s['n']})"
    )
    if gt_threshold is not None and not np.isnan(s["frac_gt"]):
        line += f"  frac>{gt_threshold:g}={s['frac_gt']:.0%}"
    return line
