"""Neural arena wrapper around the duel simulator.

The same 15-dimensional observation and 6-action interface used by the hand
written ``Bot`` is reused here, so a neural policy and a scripted policy can sit
on opposite sides of the same match without a second physics implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DUEL_DIR = ROOT / "domains" / "duel"
if str(DUEL_DIR) not in sys.path:
    sys.path.insert(0, str(DUEL_DIR))

import duel as D  # noqa: E402

OBS_DIM = 15
N_ACTIONS = 6
ACTION_NAMES = tuple(D.ACTION_NAMES)


class NeuralArena:
    def __init__(self, seed=None, max_frames=1800, env_cfg=None,
                 weapons=("none", "none"), start_gap=2.2):
        self.env = D.Duel(seed=seed, max_frames=max_frames, env_cfg=env_cfg,
                          weapons=weapons, start_gap=start_gap)

    def reset(self, seed=None):
        return self.env.reset(seed)

    def step(self, a0, a1):
        return self.env.step(a0, a1)

    @property
    def fighters(self):
        return self.env.p0, self.env.p1

    @property
    def frames(self):
        return self.env.frames

    @property
    def max_frames(self):
        return self.env.max_frames
