"""A small, deterministic affect model for the freeform duel.

Why an emotion function in a fighting game?
------------------------------------------
A boxer does not fight like a spreadsheet.  Getting hit changes *how* they
fight: a hurt fighter becomes cautious, a fighter who lands two clean counters
starts walking forward, a fighter who keeps whiffing gets frustrated and
over-commits.  The policy in this repo only sees the current frame, so it
cannot represent any of that.  Four leaky integrators give it the missing
memory of the fight's emotional history.

The state is four scalars in [0, 1]:

    composure  calm and technical; drops on damage, whiffs and guard breaks
    fear       defensive urgency; rises with low health and opponent pressure
    anger      frustration; rises on damage taken and whiffs
    momentum   confidence; rises on consecutive clean hits, resets when hit

They are not a reward.  They are appended to the observation, so the policy
can learn "when fear > 0.7, block more" without anyone scripting it.  The
update rule is a leaky integrator with emotional contagion:

    x <- decay * x + event(x) + contagion(opponent) + danger(hp)

Events come straight from the engine: a landed hit, a blocked hit, a whiff, a
guard break.  Contagion means the opponent's anger raises your fear and their
momentum suppresses yours, which is the part that makes it feel like a person
on the other side instead of an isolated stat block.

Everything is NumPy and deterministic given the reset RNG, so runs remain
reproducible.
"""

from __future__ import annotations

import numpy as np

AFFECT_NAMES = ("composure", "fear", "anger", "momentum")
AFFECT_DIM = len(AFFECT_NAMES)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


class Affect:
    """Leaky-integrator emotional state for one fighter."""

    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            composure, fear, anger, momentum = 0.85, 0.10, 0.05, 0.15
        else:
            # A little personality variation at the start: two people do not
            # walk into the same fight equally calm.
            composure = float(0.72 + 0.18 * rng.random())
            fear = float(0.04 + 0.08 * rng.random())
            anger = float(0.02 + 0.06 * rng.random())
            momentum = float(0.08 + 0.14 * rng.random())
        self.composure = composure
        self.fear = fear
        self.anger = anger
        self.momentum = momentum

    # --------------------------------------------------------------- read-out
    def vector(self) -> np.ndarray:
        return np.array([self.composure, self.fear, self.anger, self.momentum],
                        dtype=np.float32)

    def as_dict(self) -> dict:
        return {k: round(float(getattr(self, k)), 3) for k in AFFECT_NAMES}

    def clamp(self) -> None:
        self.composure = _clip01(self.composure)
        self.fear = _clip01(self.fear)
        self.anger = _clip01(self.anger)
        self.momentum = _clip01(self.momentum)

    @property
    def arousal(self) -> float:
        """How emotionally activated the fighter is (fear + anger)."""
        return _clip01(0.5 * self.fear + 0.5 * self.anger)

    # ---------------------------------------------------------------- updates
    def tick(self, hp_ratio: float, opp_fear: float, opp_anger: float,
             opp_momentum: float) -> None:
        """One frame of decay, danger and emotional contagion."""
        # Emotional time scales are seconds, not frames: at 60 Hz a fear
        # impulse should still be felt 3-5 seconds later, otherwise the policy
        # sees only the current hit and the whole layer is useless memory.
        self.composure += (0.82 - self.composure) * 0.008
        self.fear *= 0.995
        self.anger *= 0.992
        self.momentum *= 0.985

        # Low health is scary; cornered fighters also stop thinking clearly.
        danger = max(0.0, 0.35 - float(hp_ratio)) / 0.35
        self.fear += 0.006 * danger
        self.composure -= 0.004 * danger

        # Contagion: an angry opponent is intimidating; a dominant opponent
        # drains your momentum even if you are not being hit this frame.
        self.fear += 0.004 * opp_anger
        self.composure -= 0.002 * opp_anger
        self.momentum -= 0.003 * opp_momentum
        self.fear -= 0.001 * opp_fear        # seeing them scared steadies you
        self.clamp()

    def hit_landed(self, dmg: float, head: bool = False,
                   blocked: bool = False) -> None:
        if blocked:
            # Pushing into a guard is neither success nor failure.
            self.composure += 0.02
            self.anger += 0.02
            self.momentum += 0.01
        else:
            self.momentum += 0.12 + dmg / 45.0 + (0.05 if head else 0.0)
            self.composure += 0.05
            self.anger -= 0.02
            self.fear -= 0.03
        self.clamp()

    def hit_taken(self, dmg: float, head: bool = False,
                  blocked: bool = False) -> None:
        if blocked:
            self.composure += 0.03
            self.fear += 0.01
            self.anger += 0.02
        else:
            pain = min(1.0, dmg / 18.0)
            self.composure -= 0.12 * pain + (0.04 if head else 0.0)
            self.anger += 0.10 * pain
            self.fear += 0.08 * pain
            self.momentum = 0.0
        self.clamp()

    def whiffed(self) -> None:
        self.composure -= 0.035
        self.anger += 0.030
        self.momentum *= 0.85
        self.clamp()

    def guard_break(self) -> None:
        self.composure -= 0.18
        self.fear += 0.15
        self.anger += 0.10
        self.momentum = 0.0
        self.clamp()
