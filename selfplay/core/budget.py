"""Compute ledger: measure skill per CPU-second instead of just skill.

Self-play spends nearly all of its time in three places -- stepping the
environment, scoring with the policy, and replaying the collected frames
through PPO.  Without a ledger, "make them better" degenerates into "run
longer"; with one, every scheduling or physics change can be judged by the
same yardstick.

The skill axis is a win-rate converted to Elo.  Win rates are bounded and
flatten near the top, so raw deltas are misleading; Elo differences stay
readable at both ends of the scale.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager


def elo_from_score(score: float, games: int = 0, eps: float = 1e-3) -> float:
    """Convert a win rate (0..1) to an Elo difference.

    With a handful of games a 1.0 score is an artifact of the sample size, not
    a 1200-Elo gap.  Passing ``games`` applies a Laplace prior (half a win and
    half a loss), which keeps short evaluations finite and monotonic.
    """
    s = float(score)
    if games > 0:
        s = (s * games + 0.5) / (games + 1.0)
    s = min(max(s, eps), 1.0 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


def score_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-float(elo) / 400.0))


class ComputeLedger:
    """Per-iteration phase timings plus a cumulative Elo-per-CPU-second.

    Only wall time of this process is counted: on a shared box it is the
    honest denominator for "how much did this iteration cost".
    """

    PHASES = ("collect", "ppo", "world", "eval", "snapshot", "save")

    def __init__(self, baseline_elo: float | None = None):
        self.baseline_elo = baseline_elo
        self.cpu_s_total = 0.0
        self.frames_total = 0
        self.iterations = 0
        self._phase_times: dict[str, float] = {}
        self._open: dict[str, float] = {}

    # -------------------------------------------------------------- timing
    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._phase_times[name] = (self._phase_times.get(name, 0.0)
                                       + time.perf_counter() - t0)

    def start_iteration(self):
        self._phase_times = {}

    def finish_iteration(self, env_frames: int) -> dict:
        phases = {k: round(v, 3) for k, v in self._phase_times.items()}
        wall = sum(self._phase_times.values())
        self.cpu_s_total += wall
        self.frames_total += int(env_frames)
        self.iterations += 1
        collect = self._phase_times.get("collect", 0.0)
        fps = env_frames / collect if collect > 0 else 0.0
        return {
            "phase_sec": phases,
            "wall_sec": round(wall, 3),
            "cpu_s_total": round(self.cpu_s_total, 3),
            "train_frames": int(env_frames),
            "frames_per_sec": round(fps, 1),
            "frames_total": int(self.frames_total),
        }

    # --------------------------------------------------------------- skill
    def skill(self, holdout_win: float, games: int = 0) -> dict:
        elo = elo_from_score(holdout_win, games=games)
        if self.baseline_elo is None:
            self.baseline_elo = elo          # first measurement defines zero
        delta = elo - self.baseline_elo
        per_kcs = (delta / max(self.cpu_s_total, 1e-6)) * 1000.0
        return {
            "elo": round(elo, 1),
            "elo_delta": round(delta, 1),
            "elo_per_kcs": round(per_kcs, 2),
        }

    # -------------------------------------------------------------- resume
    def state_dict(self) -> dict:
        return {"baseline_elo": self.baseline_elo,
                "cpu_s_total": self.cpu_s_total,
                "frames_total": self.frames_total,
                "iterations": self.iterations}

    def load_state_dict(self, state: dict | None):
        if not state:
            return
        self.baseline_elo = state.get("baseline_elo")
        self.cpu_s_total = float(state.get("cpu_s_total", 0.0))
        self.frames_total = int(state.get("frames_total", 0))
        self.iterations = int(state.get("iterations", 0))
