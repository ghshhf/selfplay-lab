"""Minimal PPO for discrete actions, written in NumPy.

This is deliberately small and readable: a tanh MLP policy, a separate value
MLP, GAE, clipped surrogate, entropy bonus, and Adam.  It is not meant to beat
a GPU implementation on large problems; it is meant to let the lab's self-play
loop switch from hand-written genes to a real neural policy without adding
torch as a hard dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .mlp import Adam, MLP, DTYPE


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - np.max(logits, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    batch_size: int = 256
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    weight_decay: float = 0.0     # AdamW-style, decoupled
    kl_coef: float = 0.0          # RLHF anchor to the reference policy


@dataclass
class Step:
    obs: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool
    ref_logp: float = 0.0         # log-prob under the frozen reference policy
    raw_obs: np.ndarray | None = None   # arena-space observation, for the world model


@dataclass
class Episode:
    steps: List[Step] = field(default_factory=list)
    last_obs: np.ndarray | None = None


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64,
                 cfg: PPOConfig | None = None, seed: int | None = None):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.cfg = cfg or PPOConfig()
        rng = np.random.default_rng(seed)
        self.policy = MLP([self.obs_dim, self.hidden, self.n_actions],
                          activation="tanh", out_activation="identity", rng=rng)
        self.value = MLP([self.obs_dim, self.hidden, 1],
                         activation="tanh", out_activation="identity", rng=rng)
        self.policy_adam = Adam(self.policy.params, lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)
        self.value_adam = Adam(self.value.params, lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)

    # ---------------------------------------------------------------- acting
    def act(self, obs, rng: np.random.Generator, deterministic: bool = False):
        x = np.asarray(obs, dtype=DTYPE).reshape(1, -1)
        logits = self.policy.forward(x)[0]
        probs = softmax(logits)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(rng.choice(self.n_actions, p=probs))
        logp = float(log_softmax(logits)[action])
        value = float(self.value.forward(x)[0, 0])
        return action, logp, value

    def value_of(self, obs) -> float:
        x = np.asarray(obs, dtype=DTYPE).reshape(1, -1)
        return float(self.value.forward(x)[0, 0])

    def logp_of(self, obs, action: int) -> float:
        """Log-probability this agent assigns to ``action`` in ``obs``."""
        x = np.asarray(obs, dtype=DTYPE).reshape(1, -1)
        return float(log_softmax(self.policy.forward(x)[0])[int(action)])

    def entropy_of(self, obs) -> float:
        x = np.asarray(obs, dtype=DTYPE).reshape(1, -1)
        logits = self.policy.forward(x)[0]
        p = softmax(logits)
        lp = log_softmax(logits)
        return float(-np.sum(p * lp))

    def param_norm(self) -> float:
        return float(np.sqrt(self.policy.param_norm() ** 2 + self.value.param_norm() ** 2))

    # --------------------------------------------------------------- training
    def train(self, episodes: Sequence[Episode], rng: np.random.Generator) -> dict:
        all_steps: List[Step] = []
        all_adv = []
        all_ret = []
        for ep in episodes:
            if not ep.steps:
                continue
            rewards = np.asarray([s.reward for s in ep.steps], dtype=np.float64)
            values = np.asarray([s.value for s in ep.steps], dtype=np.float64)
            dones = np.asarray([1.0 if s.done else 0.0 for s in ep.steps], dtype=np.float64)
            adv = np.zeros_like(rewards)
            gae = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
                nonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.cfg.gamma * next_value * nonterminal - values[t]
                gae = delta + self.cfg.gamma * self.cfg.lam * nonterminal * gae
                adv[t] = gae
            all_steps.extend(ep.steps)
            all_adv.extend(adv.tolist())
            all_ret.extend((adv + values).tolist())

        if not all_steps:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0,
                    "updates": 0}

        adv = np.asarray(all_adv, dtype=np.float64)
        ret = np.asarray(all_ret, dtype=np.float64)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = np.stack([s.obs for s in all_steps]).astype(DTYPE)
        actions = np.asarray([s.action for s in all_steps], dtype=np.int64)
        old_logp = np.asarray([s.logp for s in all_steps], dtype=np.float64)
        ref_logp = np.asarray([s.ref_logp for s in all_steps], dtype=np.float64)
        n = len(all_steps)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                   "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0,
                   "ref_kl": 0.0, "updates": 0}
        updates = 0

        for _epoch in range(self.cfg.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.cfg.batch_size):
                idx = order[start:start + self.cfg.batch_size]
                if len(idx) < 2:
                    continue
                b_obs = obs[idx]
                b_act = actions[idx]
                b_old = old_logp[idx]
                b_ref = ref_logp[idx]
                b_adv = adv[idx]
                b_ret = ret[idx]
                bsz = len(idx)

                logits = self.policy.forward(b_obs)
                lp_all = log_softmax(logits)
                p = np.exp(lp_all)
                logp = lp_all[np.arange(bsz), b_act]
                entropy = -np.sum(p * lp_all, axis=1)
                ratio = np.exp(logp - b_old)

                unclipped = ratio * b_adv
                clipped = np.clip(ratio, 1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * b_adv
                # RLHF anchor: sampled KL(pi || pi_ref).  Shifting the loss by
                # +beta * KL pulls the policy back towards the cloned prior.
                ref_kl = logp - b_ref if self.cfg.kl_coef else np.zeros(bsz)
                policy_loss = -float(np.mean(np.minimum(unclipped, clipped))) \
                    - self.cfg.entropy_coef * float(np.mean(entropy)) \
                    + self.cfg.kl_coef * float(np.mean(ref_kl))

                # Standard PPO gradient: only the unclipped branch (or the
                # clipped branch when it is the min) contributes.
                active = ((b_adv > 0) & (ratio < 1.0 + self.cfg.clip)) | \
                         ((b_adv < 0) & (ratio > 1.0 - self.cfg.clip))
                dlogp = -b_adv * ratio * active.astype(np.float64) / bsz
                if self.cfg.kl_coef:
                    dlogp = dlogp + self.cfg.kl_coef / bsz

                onehot = np.zeros((bsz, self.n_actions), dtype=np.float64)
                onehot[np.arange(bsz), b_act] = 1.0
                dlogits = dlogp[:, None] * (onehot - p)
                # Entropy bonus: loss contains -coef * H.
                dlogits += self.cfg.entropy_coef * p * (lp_all + entropy[:, None]) / bsz

                policy_grads = self.policy.backward(dlogits.astype(DTYPE))
                grad_norm = float(np.sqrt(sum(float((g * g).sum())
                                             for layer in policy_grads for g in layer)))
                self.policy_adam.step(policy_grads, max_norm=self.cfg.max_grad_norm)

                values = self.value.forward(b_obs)[:, 0]
                value_loss = 0.5 * float(np.mean((values - b_ret) ** 2))
                dvalues = (self.cfg.value_coef * (values - b_ret) / bsz).astype(DTYPE)
                value_grads = self.value.backward(dvalues[:, None])
                self.value_adam.step(value_grads, max_norm=self.cfg.max_grad_norm)

                with np.errstate(divide="ignore", invalid="ignore"):
                    approx_kl = float(np.mean(b_old - logp))
                clip_frac = float(np.mean(np.abs(ratio - 1.0) > self.cfg.clip))

                metrics["policy_loss"] += policy_loss
                metrics["value_loss"] += value_loss
                metrics["entropy"] += float(np.mean(entropy))
                metrics["approx_kl"] += approx_kl
                metrics["ref_kl"] += float(np.mean(ref_kl))
                metrics["clip_frac"] += clip_frac
                metrics["grad_norm"] += grad_norm
                metrics["updates"] += 1
                updates += 1

                if self.cfg.target_kl is not None and approx_kl > 1.5 * self.cfg.target_kl:
                    break
            if self.cfg.target_kl is not None and metrics["approx_kl"] / max(1, updates) > 1.5 * self.cfg.target_kl:
                break

        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
                  "grad_norm", "ref_kl"):
            metrics[k] = metrics[k] / max(1, updates)
        metrics["updates"] = updates
        return metrics

    # -------------------------------------------------------------- checkpoint
    def state_dict(self, prefix="agent"):
        out = {}
        out.update(self.policy.state_dict(prefix + "_policy"))
        out.update(self.value.state_dict(prefix + "_value"))
        out.update(self.policy_adam.state_dict(prefix + "_policy_adam"))
        out.update(self.value_adam.state_dict(prefix + "_value_adam"))
        meta = np.asarray([self.obs_dim, self.n_actions, self.hidden], dtype=np.int64)
        out[prefix + "__meta"] = meta
        return out

    def load_state_dict(self, state, prefix="agent"):
        meta = np.asarray(state[prefix + "__meta"]).tolist()
        if [int(x) for x in meta] != [self.obs_dim, self.n_actions, self.hidden]:
            raise ValueError("agent shape mismatch")
        self.policy.load_state_dict(state, prefix + "_policy")
        self.value.load_state_dict(state, prefix + "_value")
        self.policy_adam.load_state_dict(state, prefix + "_policy_adam")
        self.value_adam.load_state_dict(state, prefix + "_value_adam")

    def save(self, path, extra=None):
        arrays = self.state_dict("agent")
        if extra:
            arrays.update(extra)
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path, cfg: PPOConfig | None = None):
        with np.load(str(path), allow_pickle=False) as data:
            meta = np.asarray(data["agent__meta"]).tolist()
            agent = cls(int(meta[0]), int(meta[1]), int(meta[2]), cfg=cfg, seed=0)
            agent.load_state_dict(data, "agent")
            return agent

    def freeze(self) -> "FrozenPolicy":
        return FrozenPolicy(self)


class FrozenPolicy:
    """A deterministic view over an agent, used as a league opponent."""

    def __init__(self, agent: PPOAgent):
        self.agent = agent

    def act(self, obs, rng=None, deterministic=True):
        rng = rng if rng is not None else np.random.default_rng(0)
        return self.agent.act(obs, rng, deterministic=deterministic)
