"""Evaluator / training health diagnostics.

The lab's main finding is that the bottleneck is usually the evaluator, not the
search algorithm.  This module turns that statement into a small, reusable
report: saturation, floor-lock, league overfit, and policy collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class HealthConfig:
    window: int = 10
    min_history: int = 4
    saturation: float = 0.95
    floor: float = 0.10
    holdout_gap: float = 0.20
    entropy_min: float = 0.35
    std_min: float = 0.02


def _window(xs: Sequence[float], n: int) -> list[float]:
    xs = [float(x) for x in xs if x is not None]
    return xs[-n:] if n > 0 else xs


def diagnose(train_win: Sequence[float],
             holdout_win: Sequence[float] | None = None,
             entropy: Sequence[float] | None = None,
             world_mse: Sequence[float] | None = None,
             cfg: HealthConfig | None = None) -> dict:
    """Return a compact, machine-readable health report.

    ``train_win`` is the win rate against the current training opponent (in
    self-play this hovers near 0.5), while ``holdout_win`` is the frozen
    evaluation suite that actually decides whether anything was learned.
    """
    cfg = cfg or HealthConfig()
    tw = _window(train_win, cfg.window)
    hw = _window(holdout_win or [], cfg.window)
    ew = _window(entropy or [], cfg.window)
    wm = _window(world_mse or [], cfg.window)

    report = {
        "status": "ok",
        "reasons": [],
        "n": len(tw),
        "train_mean": round(sum(tw) / len(tw), 4) if tw else None,
        "train_std": round(_std(tw), 4) if tw else None,
        "holdout_mean": round(sum(hw) / len(hw), 4) if hw else None,
        "holdout_std": round(_std(hw), 4) if hw else None,
        "holdout_gap": None,
        "entropy_mean": round(sum(ew) / len(ew), 4) if ew else None,
        "world_mse_mean": round(sum(wm) / len(wm), 4) if wm else None,
    }

    if len(tw) < cfg.min_history:
        report["status"] = "warming_up"
        report["reasons"].append("history_too_short")
        return report

    if hw:
        gap = report["train_mean"] - report["holdout_mean"]
        report["holdout_gap"] = round(gap, 4)

    reasons = report["reasons"]
    status = report["status"]

    if hw and report["holdout_mean"] is not None and report["holdout_mean"] >= cfg.saturation:
        status = "saturated"
        reasons.append("holdout_saturated")
    elif not hw and report["train_mean"] is not None and report["train_mean"] >= 0.98:
        status = "saturated"
        reasons.append("train_saturated")

    if hw and report["holdout_mean"] is not None and report["holdout_mean"] <= cfg.floor:
        status = "floor_lock"
        reasons.append("holdout_floor")

    if report["holdout_gap"] is not None and report["holdout_gap"] > cfg.holdout_gap:
        if status == "ok":
            status = "league_overfit"
        reasons.append("league_gap")

    if ew and report["entropy_mean"] is not None and report["entropy_mean"] < cfg.entropy_min:
        # Low entropy is normal for a near-deterministic policy (a behaviour-
        # cloned / well-trained net).  It is only a pathology when the frozen
        # holdout is flat too -- a deterministic policy whose holdout is still
        # moving (rising, or oscillating after a level-up) is not collapsed.
        flat = report["holdout_std"] is not None and report["holdout_std"] < cfg.std_min
        if status == "ok" and flat:
            status = "policy_collapse"
        reasons.append("low_entropy")

    if report["train_std"] is not None and report["train_std"] < cfg.std_min \
            and report["train_mean"] is not None and 0.45 <= report["train_mean"] <= 0.55:
        reasons.append("selfplay_stalemate")

    report["status"] = status
    report["reasons"] = reasons
    return report


def _std(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
