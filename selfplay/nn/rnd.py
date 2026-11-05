"""Random Network Distillation (RND) curiosity.

The latent world model's prediction error is a trap as an intrinsic reward: a
substitution (teleport) is exactly the transition the model cannot predict, so
"maximise surprise" trains the policy to spam teleports instead of to win.

RND sidesteps this by never rewarding *environment* discontinuity:

* a **fixed random** target network maps the observation to a feature vector;
* a **predictor** is trained to reproduce that feature;
* novelty = predictor error on the *current* observation.

The target is random and never updated, so the error can only fall as the
predictor sees a state more often.  It measures ``how familiar is this state``,
which is a much safer driver than ``how badly did the model miss``.  Two states
that are one frame apart are no more surprising than the same state seen twice.

This is the same trick as Burda et al. 2018 ("Exploration by Random Network
Distillation"), reduced to the one-file NumPy scale of this repo.
"""

from __future__ import annotations

import numpy as np

from .mlp import Adam, MLP, DTYPE


class RND:
    def __init__(self, obs_dim: int, hidden: int = 64, out_dim: int = 16,
                 lr: float = 1e-3, seed: int | None = None):
        self.obs_dim = int(obs_dim)
        self.out_dim = int(out_dim)
        rng = np.random.default_rng(seed)
        # Same init distribution, different random draws: predictor and target
        # start apart, and only the predictor ever moves.
        self.target = MLP([self.obs_dim, hidden, self.out_dim],
                          activation="relu", out_activation="identity", rng=rng)
        self.predictor = MLP([self.obs_dim, hidden, self.out_dim],
                             activation="relu", out_activation="identity",
                             rng=rng)
        self.adam = Adam(self.predictor.params, lr=lr)

    # ------------------------------------------------------------------ score
    def error(self, obs) -> float:
        """MSE between predictor and the frozen random target, no update."""
        x = np.asarray(obs, dtype=DTYPE)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        target = self.target.forward(x)
        pred = self.predictor.forward(x)
        return float(np.mean((pred - target) ** 2))

    def update(self, obs, lr: float | None = None) -> float:
        """One gradient step of the predictor on this observation."""
        x = np.asarray(obs, dtype=DTYPE)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        target = self.target.forward(x)
        pred = self.predictor.forward(x)
        diff = pred - target
        grads = self.predictor.backward(
            (2.0 * diff / len(x)).astype(DTYPE))
        self.adam.step(grads, lr=lr, max_norm=1.0)
        return float(np.mean(diff * diff))

    # -------------------------------------------------------------- persistence
    def state_dict(self, prefix="rnd"):
        out = self.target.state_dict(prefix + "_target")
        out.update(self.predictor.state_dict(prefix + "_pred"))
        out.update(self.adam.state_dict(prefix + "_adam"))
        out[prefix + "__meta"] = np.asarray(
            [self.obs_dim, self.out_dim], dtype=np.int64)
        return out

    def load_state_dict(self, state, prefix="rnd"):
        meta = [int(x) for x in np.asarray(state[prefix + "__meta"]).tolist()]
        if meta != [self.obs_dim, self.out_dim]:
            raise ValueError("rnd shape mismatch")
        self.target.load_state_dict(state, prefix + "_target")
        self.predictor.load_state_dict(state, prefix + "_pred")
        self.adam.load_state_dict(state, prefix + "_adam")
