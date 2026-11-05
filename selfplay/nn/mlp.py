"""Tiny NumPy MLP + Adam.

The lab intentionally keeps the core dependency at numpy only.  This module is
the neural counterpart of the hand-written gene policies: small enough to run on
a laptop CPU, but with real backprop and Adam so the same self-play loop can be
used with gradient-based agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

DTYPE = np.float32


def _activation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name in ("identity", "linear", None):
        return x
    raise ValueError("unknown activation: %r" % (name,))


def _activation_grad(a: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return (a > 0.0).astype(a.dtype)
    if name == "tanh":
        return 1.0 - a * a
    if name in ("identity", "linear", None):
        return np.ones_like(a)
    raise ValueError("unknown activation: %r" % (name,))


class Adam:
    """Adam with optional global-norm gradient clipping."""

    def __init__(self, params, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = [[np.zeros_like(p) for p in layer] for layer in params]
        self.v = [[np.zeros_like(p) for p in layer] for layer in params]

    def step(self, grads, lr=None, max_norm=None):
        lr = self.lr if lr is None else float(lr)
        if max_norm is not None and max_norm > 0:
            total = float(np.sqrt(sum(float((g * g).sum()) for layer in grads for g in layer)))
            if total > max_norm:
                scale = max_norm / (total + 1e-12)
                grads = [[g * scale for g in layer] for layer in grads]

        self.t += 1
        for i, layer in enumerate(self.params):
            for j, p in enumerate(layer):
                g = grads[i][j]
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * (g * g)
                mhat = self.m[i][j] / (1.0 - self.beta1 ** self.t)
                vhat = self.v[i][j] / (1.0 - self.beta2 ** self.t)
                p -= lr * mhat / (np.sqrt(vhat) + self.eps)

    def state_dict(self, prefix="adam"):
        out = {prefix + "__t": np.array([self.t], dtype=np.int64)}
        for i, layer in enumerate(self.m):
            for j, arr in enumerate(layer):
                out["%s__m_%d_%d" % (prefix, i, j)] = arr.copy()
                out["%s__v_%d_%d" % (prefix, i, j)] = self.v[i][j].copy()
        return out

    def load_state_dict(self, state, prefix="adam"):
        self.t = int(state[prefix + "__t"][0])
        for i, layer in enumerate(self.m):
            for j in range(len(layer)):
                self.m[i][j] = np.asarray(state["%s__m_%d_%d" % (prefix, i, j)], dtype=DTYPE).copy()
                self.v[i][j] = np.asarray(state["%s__v_%d_%d" % (prefix, i, j)], dtype=DTYPE).copy()


class MLP:
    """Fully-connected network.

    ``forward`` accepts a single vector or a batch.  ``backward`` uses the cache
    from the most recent forward call, which is exactly what PPO needs for its
    per-minibatch updates.
    """

    def __init__(self, sizes: Sequence[int], activation="tanh",
                 out_activation="identity", rng=None, dtype=DTYPE):
        self.sizes = [int(s) for s in sizes]
        if len(self.sizes) < 2:
            raise ValueError("MLP needs at least input and output sizes")
        self.activation = activation
        self.out_activation = out_activation
        self.dtype = dtype
        rng = rng if rng is not None else np.random.default_rng()
        self.params = []
        for i in range(len(self.sizes) - 1):
            fan_in, fan_out = self.sizes[i], self.sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in) if activation == "relu" else np.sqrt(1.0 / fan_in)
            w = rng.normal(0.0, scale, (fan_in, fan_out)).astype(dtype)
            b = np.zeros(fan_out, dtype=dtype)
            self.params.append([w, b])
        self._cache = None

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        acts = [x]
        pres = []
        a = x
        for i, (w, b) in enumerate(self.params):
            z = a @ w + b
            pres.append(z)
            name = self.out_activation if i == len(self.params) - 1 else self.activation
            a = _activation(z, name)
            acts.append(a)
        self._cache = (acts, pres)
        return a[0] if single else a

    # --------------------------------------------------------------- backward
    def backward(self, grad_out):
        if self._cache is None:
            raise RuntimeError("backward called before forward")
        acts, _pres = self._cache
        g = np.asarray(grad_out, dtype=self.dtype)
        if g.ndim == 1:
            g = g[None, :]
        grads = []
        n = len(self.params)
        for i in range(n - 1, -1, -1):
            w, _b = self.params[i]
            if i == n - 1:
                g = g * _activation_grad(acts[i + 1], self.out_activation)
            d_w = acts[i].T @ g
            d_b = g.sum(axis=0)
            grads.append([d_w.astype(self.dtype), d_b.astype(self.dtype)])
            if i > 0:
                g = g @ w.T
                g = g * _activation_grad(acts[i], self.activation)
        grads.reverse()
        return grads

    # ------------------------------------------------------------- utilities
    def flat_params(self) -> np.ndarray:
        return np.concatenate([p.ravel() for layer in self.params for p in layer])

    def set_flat_params(self, flat: np.ndarray):
        flat = np.asarray(flat, dtype=self.dtype)
        offset = 0
        for layer in self.params:
            for p in layer:
                n = p.size
                p[...] = flat[offset:offset + n].reshape(p.shape)
                offset += n
        if offset != flat.size:
            raise ValueError("flat parameter size mismatch")

    def copy_from(self, other: "MLP"):
        if self.sizes != other.sizes:
            raise ValueError("size mismatch")
        for mine, theirs in zip(self.params, other.params):
            for p, q in zip(mine, theirs):
                p[...] = q

    def param_norm(self) -> float:
        return float(np.sqrt(sum(float((p * p).sum()) for layer in self.params for p in layer)))

    def state_dict(self, prefix="mlp"):
        out = {
            prefix + "__sizes": np.asarray(self.sizes, dtype=np.int64),
            prefix + "__activation": np.asarray([str(self.activation)]) ,
            prefix + "__out_activation": np.asarray([str(self.out_activation)]),
        }
        for i, layer in enumerate(self.params):
            out["%s__w_%d" % (prefix, i)] = layer[0].copy()
            out["%s__b_%d" % (prefix, i)] = layer[1].copy()
        return out

    def load_state_dict(self, state, prefix="mlp"):
        sizes = tuple(int(x) for x in np.asarray(state[prefix + "__sizes"]).tolist())
        if sizes != tuple(self.sizes):
            raise ValueError("size mismatch: checkpoint %s vs model %s" % (sizes, tuple(self.sizes)))
        for i, layer in enumerate(self.params):
            layer[0][...] = np.asarray(state["%s__w_%d" % (prefix, i)], dtype=self.dtype)
            layer[1][...] = np.asarray(state["%s__b_%d" % (prefix, i)], dtype=self.dtype)

    def save(self, path, prefix="mlp", extra=None):
        arrays = self.state_dict(prefix)
        if extra:
            arrays.update(extra)
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path, prefix="mlp"):
        with np.load(str(path), allow_pickle=False) as data:
            sizes = tuple(int(x) for x in data[prefix + "__sizes"].tolist())
            activation = str(data[prefix + "__activation"][0])
            out_activation = str(data[prefix + "__out_activation"][0])
            model = cls(sizes, activation=activation, out_activation=out_activation,
                        rng=np.random.default_rng(0))
            model.load_state_dict(data, prefix=prefix)
            return model

    def grads_flat(self, grads):
        return np.concatenate([g.ravel() for layer in grads for g in layer])
