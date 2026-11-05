"""Latent world model: compress a fight into a few numbers, then predict in it.

The one-step ``WorldModel`` predicts the next 33-dim observation directly.  That
works as a health check but it is not a *representation*: nothing forces it to
discover that "opponent is winding up a head kick" and "I am 0.9 m away" are the
same kind of thing.

Here the state passes through a bottleneck first::

    obs (33) --encoder--> z (8) --predictor--> z' (8) --decoder--> obs' (33)
                            ^
                            + action one-hot

The encoder is trained by reconstruction (so it cannot collapse), the predictor
by latent MSE (stop-gradient target), and the two are coupled only through z.
Because the prediction happens in 8 dimensions instead of 33, the same rollout
gives a cheaper, smoother learning signal -- and the latent is the thing a
curriculum can compare across 1-D, 2-D and (eventually) 3-D arenas.
"""

from __future__ import annotations

import numpy as np

from .mlp import Adam, MLP, DTYPE, _activation_grad


def _backward_with_input(net: MLP, grad_out):
    """Like ``MLP.backward`` but also returns d(loss)/d(input)."""
    acts, _pres = net._cache
    g = np.asarray(grad_out, dtype=DTYPE)
    if g.ndim == 1:
        g = g[None, :]
    grads = []
    n = len(net.params)
    for i in range(n - 1, -1, -1):
        w, _b = net.params[i]
        if i == n - 1:
            g = g * _activation_grad(acts[i + 1], net.out_activation)
        grads.append([(acts[i].T @ g).astype(DTYPE), g.sum(axis=0).astype(DTYPE)])
        if i > 0:
            g = g @ w.T
            g = g * _activation_grad(acts[i], net.activation)
    grads.reverse()
    grad_in = g @ net.params[0][0].T
    return grads, grad_in


class LatentWorldModel:
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64,
                 latent: int = 8, lr: float = 1e-3, seed: int | None = None,
                 recon_weight: float = 0.5):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.latent_dim = int(latent)
        self.recon_weight = float(recon_weight)
        # When the policy consumes z, the encoder must not drift underneath it.
        self.freeze_encoder = False
        rng = np.random.default_rng(seed)
        self.encoder = MLP([self.obs_dim, self.hidden, self.latent_dim],
                           activation="tanh", out_activation="identity",
                           rng=rng)
        self.decoder = MLP([self.latent_dim, self.hidden, self.obs_dim],
                           activation="tanh", out_activation="identity",
                           rng=rng)
        self.predictor = MLP([self.latent_dim + self.n_actions, self.hidden,
                              self.latent_dim],
                             activation="tanh", out_activation="identity",
                             rng=rng)
        # A reward head turns the latent model into a usable simulator: roll
        # the latent forward and you get both the next state and the reward.
        self.reward_head = MLP([self.latent_dim + self.n_actions, self.hidden, 1],
                               activation="tanh", out_activation="identity",
                               rng=rng)
        self.enc_adam = Adam(self.encoder.params, lr=lr)
        self.dec_adam = Adam(self.decoder.params, lr=lr)
        self.pred_adam = Adam(self.predictor.params, lr=lr)
        self.rew_adam = Adam(self.reward_head.params, lr=lr)

    # ------------------------------------------------------------------ helpers
    def _onehot(self, actions):
        a = np.asarray(actions, dtype=np.int64)
        if a.ndim == 0:
            a = a[None]
        out = np.zeros((len(a), self.n_actions), dtype=DTYPE)
        out[np.arange(len(a)), a] = 1.0
        return out

    def encode(self, obs):
        return self.encoder.forward(obs)

    def predict(self, obs, actions):
        z = self.encoder.forward(np.asarray(obs, dtype=DTYPE))
        single = np.asarray(obs).ndim == 1
        if single:
            z = z[None, :]
        feats = np.concatenate([z, self._onehot(actions)], axis=1)
        pred = self.predictor.forward(feats)
        return pred[0] if single else pred

    def error(self, obs, actions, next_obs) -> float:
        pred = self.predict(obs, actions)
        target = self.encode(next_obs)
        pred = np.atleast_2d(pred)
        target = np.atleast_2d(target)
        return float(np.mean((pred - target) ** 2))

    def step_latent(self, z, action: int) -> tuple[np.ndarray, float]:
        """One imagined transition: (next latent, predicted reward)."""
        feats = np.concatenate([np.atleast_1d(z), self._onehot([action])[0]])
        z_next = self.predictor.forward(feats[None, :])[0]
        r = float(self.reward_head.forward(feats[None, :])[0, 0])
        return z_next, r

    def decode(self, z):
        return self.decoder.forward(np.atleast_1d(z))

    # -------------------------------------------------------------------- train
    def train(self, obs, actions, next_obs, rng: np.random.Generator,
              epochs: int = 2, batch_size: int = 128,
              rewards=None) -> dict:
        obs = np.asarray(obs, dtype=DTYPE)
        nxt = np.asarray(next_obs, dtype=DTYPE)
        onehot = self._onehot(actions)
        has_reward = rewards is not None and len(rewards) == len(obs)
        if has_reward:
            rew = np.asarray(rewards, dtype=DTYPE).reshape(-1, 1)
        n = len(obs)
        if n == 0:
            return {"latent_mse": 0.0, "recon_mse": 0.0, "reward_mse": 0.0}
        stats = {"latent_mse": [], "recon_mse": [], "reward_mse": []}
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                if len(idx) < 2:
                    continue
                ob, nb, oh = obs[idx], nxt[idx], onehot[idx]

                # Autoencoder step: reconstruction keeps the bottleneck honest.
                if not self.freeze_encoder:
                    z = self.encoder.forward(ob)
                    rec = self.decoder.forward(z)
                    residual = rec - nb
                    stats["recon_mse"].append(float(np.mean(residual * residual)))
                    d_rec = (2.0 * self.recon_weight * residual / len(idx)).astype(DTYPE)
                    dec_grads, d_z = _backward_with_input(self.decoder, d_rec)
                    self.dec_adam.step(dec_grads, max_norm=1.0)
                    enc_grads, _ = _backward_with_input(self.encoder, d_z)
                    self.enc_adam.step(enc_grads, max_norm=1.0)

                # Prediction step: target is the encoder, treated as constant.
                z_t = self.encoder.forward(ob)
                z_next = self.encoder.forward(nb)
                feats = np.concatenate([z_t, oh], axis=1)
                pred = self.predictor.forward(feats)
                perr = pred - z_next
                stats["latent_mse"].append(float(np.mean(perr * perr)))
                pred_grads = self.predictor.backward(
                    (2.0 * perr / len(idx)).astype(DTYPE))
                self.pred_adam.step(pred_grads, max_norm=1.0)

                if has_reward:
                    r_pred = self.reward_head.forward(feats)
                    r_err = r_pred - rew[idx]
                    stats["reward_mse"].append(float(np.mean(r_err * r_err)))
                    r_grads = self.reward_head.backward(
                        (2.0 * r_err / len(idx)).astype(DTYPE))
                    self.rew_adam.step(r_grads, max_norm=1.0)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}

    # -------------------------------------------------------------- persistence
    def state_dict(self, prefix="world"):
        out = self.encoder.state_dict(prefix + "_enc")
        out.update(self.decoder.state_dict(prefix + "_dec"))
        out.update(self.predictor.state_dict(prefix + "_pred"))
        out.update(self.reward_head.state_dict(prefix + "_rew"))
        out.update(self.enc_adam.state_dict(prefix + "_enc_adam"))
        out.update(self.dec_adam.state_dict(prefix + "_dec_adam"))
        out.update(self.pred_adam.state_dict(prefix + "_pred_adam"))
        out.update(self.rew_adam.state_dict(prefix + "_rew_adam"))
        out[prefix + "__meta"] = np.asarray(
            [self.obs_dim, self.n_actions, self.hidden, self.latent_dim],
            dtype=np.int64)
        return out

    def load_state_dict(self, state, prefix="world"):
        meta = [int(x) for x in np.asarray(state[prefix + "__meta"]).tolist()]
        if meta != [self.obs_dim, self.n_actions, self.hidden,
                    self.latent_dim]:
            raise ValueError("latent world model shape mismatch")
        self.encoder.load_state_dict(state, prefix + "_enc")
        self.decoder.load_state_dict(state, prefix + "_dec")
        self.predictor.load_state_dict(state, prefix + "_pred")
        self.reward_head.load_state_dict(state, prefix + "_rew")
        self.enc_adam.load_state_dict(state, prefix + "_enc_adam")
        self.dec_adam.load_state_dict(state, prefix + "_dec_adam")
        self.pred_adam.load_state_dict(state, prefix + "_pred_adam")
        self.rew_adam.load_state_dict(state, prefix + "_rew_adam")
