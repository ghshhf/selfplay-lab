"""Neural one-step world model.

The model predicts the next observation from (observation, action).  It is
trained with plain MSE and is intentionally small: in this lab it serves as a
diagnostic / intrinsic-signal source, not as a replacement for the simulator.
"""

from __future__ import annotations

import numpy as np

from .mlp import Adam, MLP, DTYPE


class WorldModel:
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64,
                 lr: float = 1e-3, seed: int | None = None):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        rng = np.random.default_rng(seed)
        self.net = MLP([self.obs_dim + self.n_actions, self.hidden, self.obs_dim],
                       activation="tanh", out_activation="identity", rng=rng)
        self.adam = Adam(self.net.params, lr=lr)

    def _features(self, obs, actions):
        obs = np.asarray(obs, dtype=DTYPE)
        actions = np.asarray(actions, dtype=np.int64)
        if obs.ndim == 1:
            obs = obs[None, :]
        if actions.ndim == 0:
            actions = actions[None]
        onehot = np.zeros((len(actions), self.n_actions), dtype=DTYPE)
        onehot[np.arange(len(actions)), actions] = 1.0
        return np.concatenate([obs, onehot], axis=1)

    def train(self, obs, actions, next_obs, rng: np.random.Generator,
              epochs: int = 2, batch_size: int = 128) -> float:
        x = self._features(obs, actions)
        y = np.asarray(next_obs, dtype=DTYPE)
        if y.ndim == 1:
            y = y[None, :]
        n = len(x)
        if n == 0:
            return 0.0
        losses = []
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                if len(idx) < 2:
                    continue
                xb, yb = x[idx], y[idx]
                pred = self.net.forward(xb)
                err = pred - yb
                losses.append(float(np.mean(err * err)))
                grads = self.net.backward((err / len(idx)).astype(DTYPE))
                self.adam.step(grads, max_norm=1.0)
        if n:
            pred = self.net.forward(x)
            losses.append(float(np.mean((pred - y) ** 2)))
        return float(np.mean(losses)) if losses else 0.0

    def predict(self, obs, actions):
        return self.net.forward(self._features(obs, actions))

    def error(self, obs, actions, next_obs) -> float:
        pred = self.predict(obs, actions)
        y = np.asarray(next_obs, dtype=DTYPE)
        if y.ndim == 1:
            y = y[None, :]
        return float(np.mean((pred - y) ** 2))

    def state_dict(self, prefix="world"):
        out = self.net.state_dict(prefix + "_net")
        out.update(self.adam.state_dict(prefix + "_adam"))
        out[prefix + "__meta"] = np.asarray([self.obs_dim, self.n_actions, self.hidden], dtype=np.int64)
        return out

    def load_state_dict(self, state, prefix="world"):
        meta = [int(x) for x in np.asarray(state[prefix + "__meta"]).tolist()]
        if meta != [self.obs_dim, self.n_actions, self.hidden]:
            raise ValueError("world model shape mismatch")
        self.net.load_state_dict(state, prefix + "_net")
        self.adam.load_state_dict(state, prefix + "_adam")

    def save(self, path, extra=None):
        arrays = self.state_dict("world")
        if extra:
            arrays.update(extra)
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path, lr: float = 1e-3):
        with np.load(str(path), allow_pickle=False) as data:
            meta = [int(x) for x in np.asarray(data["world__meta"]).tolist()]
            model = cls(meta[0], meta[1], meta[2], lr=lr, seed=0)
            model.load_state_dict(data, "world")
            return model
