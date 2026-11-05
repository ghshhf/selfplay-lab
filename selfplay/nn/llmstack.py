"""Training tricks borrowed from the language-model recipe.

Everything here is something a modern LLM trainer does by default, translated
to this lab's constraints (NumPy only, ~30 k parameters, one process):

* **LR warmup + cosine decay** -- the first updates of a fresh policy are the
  noisiest, so ramp the rate up, then decay it to a floor.
* **AdamW / decoupled weight decay** -- pulls unused weights to zero without
  interacting with the adaptive step size.
* **EMA shadow weights** -- the LLM "model soup" trick: evaluate an averaged
  copy, which is usually stronger and much less jumpy than the live policy.
* **KL anchor to a reference policy** -- the RLHF term. Sampling from the
  current policy but paying ``beta * KL(pi || pi_ref)`` keeps it from drifting
  away from the behaviour-cloned prior just to farm reward.
* **Rejection sampling (STaR / RFT)** -- keep the best trajectories of an
  iteration and do a little extra supervised learning on them. This is how
  LLMs turn their own good samples into training data.
* **Mirror augmentation** -- the arena has no walls or loot, so reflecting the
  whole scene left/right is a free second copy of every state.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

DTYPE = np.float32


# --------------------------------------------------------------------- schedule
class LRSchedule:
    """Linear warmup followed by cosine decay to ``base_lr * min_ratio``."""

    def __init__(self, base_lr: float, warmup_iters: int = 0,
                 total_iters: int = 1, min_ratio: float = 0.1):
        self.base_lr = float(base_lr)
        self.warmup = max(0, int(warmup_iters))
        self.total = max(1, int(total_iters))
        self.min_ratio = float(min_ratio)

    def at(self, iteration: int) -> float:
        it = int(iteration)
        if self.warmup > 0 and it < self.warmup:
            return self.base_lr * (it + 1) / self.warmup
        span = max(1, self.total - self.warmup)
        progress = min(1.0, max(0.0, (it - self.warmup) / span))
        floor = self.min_ratio
        scale = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * scale


class EntropySchedule:
    """Anneal the exploration bonus, the way LLM sampling temperature decays."""

    def __init__(self, start: float, end: float | None = None,
                 total_iters: int = 1):
        self.start = float(start)
        self.end = self.start if end is None else float(end)
        self.total = max(1, int(total_iters))

    def at(self, iteration: int) -> float:
        p = min(1.0, max(0.0, float(iteration) / self.total))
        return self.start + (self.end - self.start) * p


# ------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average over a set of MLPs."""

    def __init__(self, mlps: Sequence, decay: float = 0.995):
        self.mlps = list(mlps)
        self.decay = float(decay)
        self.shadow = [[[p.copy() for p in layer] for layer in m.params]
                       for m in self.mlps]
        self.updates = 0

    def update(self):
        if self.updates == 0:
            for mi, m in enumerate(self.mlps):
                for i, layer in enumerate(m.params):
                    for j, p in enumerate(layer):
                        self.shadow[mi][i][j] = p.copy()
        else:
            d = self.decay
            for mi, m in enumerate(self.mlps):
                for i, layer in enumerate(m.params):
                    for j, p in enumerate(layer):
                        s = self.shadow[mi][i][j]
                        s *= d
                        s += (1.0 - d) * p
        self.updates += 1

    def copy_to(self, mlps: Sequence):
        for mi, m in enumerate(mlps):
            for i, layer in enumerate(m.params):
                for j, p in enumerate(layer):
                    p[...] = self.shadow[mi][i][j]

    def flat(self) -> np.ndarray:
        return np.concatenate([s.ravel() for m in self.shadow for layer in m for s in layer])

    def state_dict(self, prefix="ema"):
        out = {prefix + "__updates": np.asarray([self.updates], dtype=np.int64)}
        for mi, layers in enumerate(self.shadow):
            for i, layer in enumerate(layers):
                for j, s in enumerate(layer):
                    out["%s__%d_%d_%d" % (prefix, mi, i, j)] = s.copy()
        return out

    def load_state_dict(self, state, prefix="ema"):
        self.updates = int(state[prefix + "__updates"][0])
        for mi, layers in enumerate(self.shadow):
            for i, layer in enumerate(layers):
                for j in range(len(layer)):
                    self.shadow[mi][i][j] = np.asarray(
                        state["%s__%d_%d_%d" % (prefix, mi, i, j)], dtype=DTYPE).copy()

    def copy_shadow_from(self, other: "EMA"):
        for mi, layers in enumerate(other.shadow):
            for i, layer in enumerate(layers):
                for j, s in enumerate(layer):
                    self.shadow[mi][i][j] = s.copy()
        self.updates = other.updates


# ---------------------------------------------------------------- rejection RL
class RejectionBuffer:
    """Keep the (state, action) pairs of the best episodes (STaR / RFT)."""

    def __init__(self, obs_dim: int, max_steps: int = 20000):
        self.obs_dim = int(obs_dim)
        self.max_steps = int(max_steps)
        self.obs: list[np.ndarray] = []
        self.acts: list[int] = []

    def add(self, episodes: Sequence, returns: Sequence[float],
            keep_frac: float = 0.25):
        if not episodes:
            return
        k = max(1, int(round(len(episodes) * float(keep_frac))))
        order = np.argsort(np.asarray(returns, dtype=np.float64))[::-1][:k]
        for i in order:
            for step in episodes[i].steps:
                self.obs.append(np.asarray(step.obs, dtype=DTYPE))
                self.acts.append(int(step.action))
        overflow = len(self.obs) - self.max_steps
        if overflow > 0:
            self.obs = self.obs[overflow:]
            self.acts = self.acts[overflow:]

    def __len__(self):
        return len(self.obs)

    def arrays(self):
        if not self.obs:
            return (np.zeros((0, self.obs_dim), dtype=DTYPE),
                    np.zeros(0, dtype=np.int64))
        return (np.stack(self.obs), np.asarray(self.acts, dtype=np.int64))


# ------------------------------------------------------------- mirror symmetry
# The freeform arena has no walls and no loot, and actions are already
# opponent-relative (FWD means "towards the opponent"), so reflecting absolute
# x coordinates is a valid second view of the same state.
_BASE_X = (0, 1, 5, 6, 7, 11, 12)          # x, vx, facing, opp x/vx/facing, dx
_POSE_X = (15, 17, 19, 21)                 # hand/foot offsets (articulated+)


def mirror_indices(obs_dim: int) -> tuple[int, ...]:
    idx = list(_BASE_X)
    if obs_dim >= 23:
        idx.extend(_POSE_X)
    return tuple(i for i in idx if i < obs_dim)


def mirror_obs(obs: np.ndarray, obs_dim: int | None = None) -> np.ndarray:
    out = np.array(obs, dtype=DTYPE, copy=True)
    dim = len(out) if obs_dim is None else int(obs_dim)
    for i in mirror_indices(dim):
        out[..., i] = -out[..., i]
    return out


def mirror_batch(x: np.ndarray, obs_dim: int | None = None) -> np.ndarray:
    out = np.array(x, dtype=DTYPE, copy=True)
    for i in mirror_indices(len(x[0]) if obs_dim is None else int(obs_dim)):
        out[:, i] = -out[:, i]
    return out


# ------------------------------------------------------------------ KL anchor
def kl_between(logp: np.ndarray, ref_logp: np.ndarray) -> np.ndarray:
    """Sampled KL(pi || pi_ref) estimate on the actions actually taken."""
    return np.asarray(logp) - np.asarray(ref_logp)
