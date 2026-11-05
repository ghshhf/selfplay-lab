"""Opponent pool for neural self-play.

Three levels are deliberately mixed:

* random      -- enough signal for a fresh policy to learn basic approach/hit
* scripted    -- human-readable ``Bot`` genes, so the agent cannot only overfit
                  to its own current policy
* league      -- frozen snapshots of earlier neural policies, the usual
                  defence against cycling / forgetting in self-play
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DUEL_DIR = ROOT / "domains" / "duel"
if str(DUEL_DIR) not in sys.path:
    sys.path.insert(0, str(DUEL_DIR))

from bot import Bot  # noqa: E402

N_ACTIONS = 6


# A small, fixed curriculum of scripted opponents.  The first one is intentionally
# weak; later ones add reaction speed and guard pressure.
SCRIPTED_GENES = [
    {"weapon": "none", "aggr": 0.20, "guard": 0.15, "kick": 0.05,
     "dist": 1.35, "react": 14, "idle": 0.40, "loot": 0.05, "avoid": 0.05},
    {"weapon": "none", "aggr": 0.45, "guard": 0.35, "kick": 0.20,
     "dist": 1.10, "react": 8, "idle": 0.20, "loot": 0.20, "avoid": 0.15},
    {"weapon": "polearm", "aggr": 0.55, "guard": 0.40, "kick": 0.10,
     "dist": 2.20, "react": 6, "idle": 0.12, "loot": 0.15, "avoid": 0.20},
]

# Harder rungs for the world curriculum.  Measured against the BC policy at
# 600 frames, both sides: gene1 ~1.0, gene2 ~0.73, hard[0] ~0.03.
# They are deliberately steep, so a level-up means the evaluator has found a
# real gradient again rather than another saturated rung.
HARD_GENES = [
    {"weapon": "none", "aggr": 0.60, "guard": 0.50, "kick": 0.50,
     "dist": 1.20, "react": 5, "idle": 0.08, "loot": 0.20, "avoid": 0.15},
    {"weapon": "shield", "aggr": 0.80, "guard": 0.70, "kick": 0.70,
     "dist": 1.15, "react": 3, "idle": 0.02, "loot": 0.20, "avoid": 0.20},
]

# The ladder consumed by WorldCurriculum: name + the opponent used both for
# training (all rungs up to the active one) and for the frozen holdout.
CURRICULUM_LEVELS = [
    {"name": "L0-baseline", "gene": SCRIPTED_GENES[1]},
    {"name": "L1-polearm", "gene": SCRIPTED_GENES[2]},
    {"name": "L2-pressure", "gene": HARD_GENES[0]},
    {"name": "L3-strong", "gene": HARD_GENES[1]},
]


class RandomOpponent:
    def __init__(self, seed=0, n_actions=N_ACTIONS):
        self.rng = np.random.default_rng(seed)
        self.n_actions = int(n_actions)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, me, opp, obs):
        return int(self.rng.integers(self.n_actions))


class ScriptedOpponent:
    def __init__(self, gene, seed=0, reach_scale=1.0):
        self.gene = dict(gene)
        self.reach_scale = float(reach_scale)
        self.rng = np.random.default_rng(seed)
        self.bot = None

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        g = self.gene
        self.bot = Bot(
            rng=self.rng,
            weapon=g.get("weapon", "none"),
            aggression=g.get("aggr", 0.4),
            guard_bias=g.get("guard", 0.3),
            kick_bias=g.get("kick", 0.2),
            prefer_dist=g.get("dist", 1.2),
            react=g.get("react", 8),
            idle_bias=g.get("idle", 0.2),
            loot_bias=g.get("loot", 0.2),
            avoid_bias=g.get("avoid", 0.2),
            reach_scale=self.reach_scale,
        )

    def act(self, me, opp, obs):
        return int(self.bot(me, opp))


ARENA_HALF = 4.0  # duel.py normalises x by this, so obs[12] = dx_meters / 4


def heuristic_action(obs, pocket: bool = False) -> int:
    """A cheap scripted expert used only to bootstrap the neural policy.

    Observation layout (self-centric, 15 dims)::

        0 self.x, 1 self.vx, 2 self.hp, 3 self.state, 4 self.timer,
        5 self.facing, 6 opp.x, 7 opp.vx, 8 opp.hp, 9 opp.state,
        10 opp.timer, 11 opp.facing, 12 dx, 13 |dx|, 14 cooldown

    Distances are converted back to metres before comparing with move reach.
    (The earlier version divided by 8 -- the arena half-width -- which made the
    expert walk into body contact and time out instead of kicking at range.)

    The policy is a deliberate one-threshold rule: close the gap, then kick from
    just inside kick reach.  Keeping it deterministic and low-dimensional is what
    makes cloning work -- a policy with strategy switches near the same distance
    teaches the network an ambiguous boundary, and the cloned policy then whiffs
    kicks from outside reach.

    ``pocket=True`` is for the articulated arena: walk into jab range (0.95 m)
    and punch.  The jab's 4-frame windup and chin-height contact make the head
    the primary target, which is exactly the behaviour we want the clone to
    inherit.  The default keeps the 1-D duel's out-fighting kick range.
    """
    dx = abs(float(obs[12])) * ARENA_HALF
    if pocket:
        return 2 if dx <= 0.95 else 0
    return 0 if dx > 1.30 else 3


class SnapshotOpponent:
    """Deterministic frozen neural policy, acting on the observation.

    ``encode`` maps the arena observation to the policy's input space.  With
    ``--latent-policy`` a snapshot expects 8 latent numbers, not the 33 hand-
    crafted ones, so the same encoder must be applied on both sides.
    """

    def __init__(self, agent, encode=None):
        self.agent = agent
        self.encode = encode
        self._rng = np.random.default_rng(0)

    def reset(self, seed):
        pass

    def act(self, me, opp, obs):
        if self.encode is not None:
            obs = self.encode.forward(np.asarray(obs, dtype=np.float32)[None, :])[0]
        action, _logp, _value = self.agent.act(obs, self._rng, deterministic=True)
        return int(action)
