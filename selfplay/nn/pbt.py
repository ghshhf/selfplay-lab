"""Weight-space evolution (PBT-lite).

The gene pool in ``domains/neuro/selfevolve.py`` mutates a *scripted* opponent:
nine hand-authored numbers.  That evolves difficulty, not policy.  This module
mutates the network itself:

* clone a member, add Gaussian noise proportional to each tensor's scale;
* score everyone on the same fixed suite;
* keep the top half, refill the rest by mutating survivors.

That is the smallest honest version of population-based training: selection on
measured performance, variation in parameter space, no gradient involved.
"""

from __future__ import annotations

import numpy as np

from .ppo import PPOAgent


def clone_agent(agent: PPOAgent, sigma: float = 0.0,
                rng: np.random.Generator | None = None) -> PPOAgent:
    """Deep copy of ``agent``; with ``sigma > 0`` the copy is perturbed."""
    new = PPOAgent(agent.obs_dim, agent.n_actions, agent.hidden,
                   cfg=agent.cfg, seed=0)
    new.load_state_dict(agent.state_dict(), "agent")
    if sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        perturb_params(new.policy.params, sigma, rng)
        perturb_params(new.value.params, sigma, rng)
    return new


def perturb_params(params, sigma: float, rng: np.random.Generator) -> None:
    """Add ``sigma * std(tensor)`` noise in place, tensor by tensor.

    Relative noise matters: one absolute sigma either does nothing to the
    policy head or destroys the first layer.  Scaling per tensor keeps every
    layer at the same "distance" from its parent.
    """
    for layer in params:
        for p in layer:
            scale = float(np.std(p))
            if scale <= 0.0:
                scale = 1.0
            p[...] = p + rng.normal(0.0, sigma * scale, size=p.shape)


class PBTPopulation:
    """A small population that is scored and recombined every ``evolve`` call."""

    def __init__(self, base_agent: PPOAgent, size: int = 6, sigma: float = 0.03,
                 seed: int = 0):
        self.size = max(2, int(size))
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.members: list[PPOAgent] = [clone_agent(base_agent)]
        while len(self.members) < self.size:
            self.members.append(clone_agent(base_agent, self.sigma, self.rng))
        self.last_scores: list[float] = []

    # ---------------------------------------------------------------- evolution
    def evolve(self, fitness) -> tuple[PPOAgent, dict]:
        """Score every member, keep the top half, mutate survivors to refill.

        ``fitness(agent) -> float`` must be deterministic: it is called once per
        member per generation and compared directly.
        """
        scores = [float(fitness(m)) for m in self.members]
        order = list(np.argsort(-np.asarray(scores)))
        keep_n = max(1, self.size // 2)
        survivors = [self.members[i] for i in order[:keep_n]]
        survivors = [clone_agent(a) for a in survivors]

        children = []
        while len(survivors) + len(children) < self.size:
            parent = survivors[len(children) % len(survivors)]
            children.append(clone_agent(parent, self.sigma, self.rng))

        self.members = survivors + children
        self.last_scores = scores
        best = survivors[0]
        return best, {
            "pbt_best": round(scores[order[0]], 4),
            "pbt_mean": round(float(np.mean(scores)), 4),
            "pbt_worst": round(scores[order[-1]], 4),
            "pbt_sigma": self.sigma,
        }

    def best(self) -> PPOAgent:
        if not self.last_scores:
            return self.members[0]
        return self.members[int(np.argmax(self.last_scores))]

    def state_dict(self, prefix="pbt"):
        """Only the champion is persisted; the population rebuilds from it."""
        return self.best().state_dict(prefix)
