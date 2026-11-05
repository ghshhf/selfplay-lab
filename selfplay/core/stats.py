"""Small-sample statistics for the lab.

Six evaluation games cannot separate a 5-point improvement from noise.  Every
number this repo reports goes through here first, so a claim is always stored
with its interval instead of a bare decimal.

Two rules:

* ``bootstrap_ci`` -- no normality assumption, works for win rates with draws.
* ``paired_delta`` -- A/B runs must be compared **iteration by iteration** on
  the same fixed eval suite (``_evaluate`` uses seeds 2000.. and both sides),
  because the checkpoint of iteration *i* always faces the same matches.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 800, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)))


def summarise(values, seed: int = 0) -> dict:
    """Mean + 95% interval + sample size, ready to drop into a metric record."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    lo, hi = bootstrap_ci(arr, seed=seed)
    return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi,
            "n": int(arr.size)}


def required_games(width: float = 0.10, z: float = 1.96) -> int:
    """Games needed to resolve a win-rate difference of ``width`` (worst case
    p=0.5).  Used to pick ``--eval-episodes`` instead of guessing."""
    return int(np.ceil((z / width) ** 2 * 0.25))


def paired_delta(a, b, n_boot: int = 800, seed: int = 0) -> dict:
    """Mean of ``a - b`` with its interval; ``a`` and ``b`` must be paired."""
    x = np.asarray(list(a), dtype=np.float64)
    y = np.asarray(list(b), dtype=np.float64)
    n = min(len(x), len(y))
    d = x[:n] - y[:n]
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {"delta": float(d.mean()) if n else 0.0, "ci_low": lo,
            "ci_high": hi, "n": int(n),
            "sign": ("a" if lo > 0.0 else ("b" if hi < 0.0 else "tie"))}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, for a plain binary win rate (draws excluded)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
