"""World curriculum driven by evaluator health and world-model surprise.

This is the neuro domain's version of the "world gets harder when the champion
maxes out" rule used by the genetic domains.  Two signals gate a level-up:

* the evaluator is saturated -- the frozen holdout set can no longer separate
  policies, so the current world no longer produces learning signal;
* the world model is no longer surprised -- its one-step prediction error is
  below ``wm_threshold``, meaning the agent has seen this world enough.

Either signal alone is unreliable: a saturated evaluator can still hide a world
the model cannot predict (the gomoku "missing dimension" failure), and a low
prediction error can hide a policy that merely memorised a weak opponent.  Both
must agree before the curriculum spends a harder level.

The curriculum also steps *down* on ``floor_lock``, so an over-hard world does
not silently stall training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class CurriculumConfig:
    patience: int = 2
    wm_threshold: float = 0.02
    wm_window: int = 3


class WorldCurriculum:
    """A monotone ladder of scripted opponents, gated by health + surprise."""

    def __init__(self, levels: Sequence[dict], cfg: CurriculumConfig | None = None):
        if not levels:
            raise ValueError("curriculum needs at least one level")
        self.levels = [dict(level) for level in levels]
        self.cfg = cfg or CurriculumConfig()
        self.level = 0
        self.saturated_streak = 0
        self.floor_streak = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------- accessors
    def holdout_gene(self) -> dict:
        """The current level's evaluation opponent."""
        return dict(self.levels[self.level]["gene"])

    def scripted_pool(self) -> list[dict]:
        """Training opponents: the whole ladder up to the current level.

        Keeping earlier rungs in the mix stops the policy from forgetting the
        easy world the moment the curriculum moves on.
        """
        return [dict(level["gene"]) for level in self.levels[: self.level + 1]]

    @property
    def level_name(self) -> str:
        return str(self.levels[self.level].get("name", "L%d" % self.level))

    @property
    def max_level(self) -> int:
        return len(self.levels) - 1

    # ------------------------------------------------------- level transitions
    def observe(self, iteration: int, health_status: str,
                world_mse: Sequence[float] | None) -> dict:
        """Update the level from one iteration's health report."""
        if health_status == "saturated":
            self.saturated_streak += 1
            self.floor_streak = 0
        elif health_status == "floor_lock":
            self.floor_streak += 1
            self.saturated_streak = 0
        else:
            self.saturated_streak = 0
            self.floor_streak = 0

        action, reason = "hold", "not_mastered"
        if self.floor_streak >= self.cfg.patience and self.level > 0:
            self.level -= 1
            self.floor_streak = 0
            self.saturated_streak = 0
            action, reason = "down", "holdout_floor"
        elif self.saturated_streak >= self.cfg.patience:
            if self.level >= self.max_level:
                action, reason = "hold", "max_level"
            elif self._world_is_predictable(world_mse):
                self.level += 1
                self.saturated_streak = 0
                action, reason = "up", "evaluator_saturated_and_world_known"
            else:
                action, reason = "hold", "world_model_uncertain"

        record = {
            "iter": int(iteration),
            "level": self.level,
            "name": self.level_name,
            "action": action,
            "reason": reason,
        }
        self.history.append(record)
        self.history = self.history[-50:]
        return record

    def _world_is_predictable(self, world_mse: Sequence[float] | None) -> bool:
        # Without a world model there is no surprise signal to gate on, so the
        # evaluator alone decides (the pre-world-model behaviour).
        if not world_mse:
            return True
        window = [float(x) for x in world_mse if x is not None][-self.cfg.wm_window:]
        if not window:
            return True
        return sum(window) / len(window) < self.cfg.wm_threshold

    # ------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        return {
            "level": self.level,
            "saturated_streak": self.saturated_streak,
            "floor_streak": self.floor_streak,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict) -> None:
        self.level = min(int(state.get("level", 0)), self.max_level)
        self.saturated_streak = int(state.get("saturated_streak", 0))
        self.floor_streak = int(state.get("floor_streak", 0))
        self.history = list(state.get("history", []))[-50:]
