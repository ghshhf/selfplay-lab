"""Freeform duel: the articulated rules plus a vertical axis and three new actions.

Why this file exists
--------------------
The articulated domain fixed contact (capsules, head hurtbox) but left the body
on a rail: no height, no duck, no jump.  Everything the policy could express
was still "walk forward / walk back / swing".  This layer hands the fighters
the rest of the movement vocabulary and lets the geometry decide what is
strong:

* **JUMP** (action 6) -- gravity 9.8 m/s^2, ~0.33 m apex, ~31 frames airborne.
* **DUCK**  (action 7) -- drops the head to 1.19 m, so a jab (1.64 m) or a head
  kick (1.70 m) passes over it, while a front kick at hip height still lands.
* **KICK from a duck** becomes a low roundhouse (foot ~0.46 m): below every
  guard capsule, so blocking cannot stop it, but it only reaches the legs.
* **KICK in the air** becomes a flying kick: the foot rises to chin height,
  trading 9 frames of windup for 9 damage.

That is a deliberate rock-paper-scissors the engine never encodes as a rule
table -- it falls out of the capsules:

    duck beats jab / head kick
    front kick beats duck
    low kick beats block
    jump beats low kick

Six actions -> nine, 23 observations -> 29 (height and vertical speed for both
fighters plus two duck flags).  No scripting of *when* to use them; the policy
has to find that itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DUEL_DIR = Path(__file__).resolve().parent
if str(DUEL_DIR) not in sys.path:
    sys.path.insert(0, str(DUEL_DIR))

try:  # package import
    from .duel import (FPS, WALK_SPEED, S_IDLE, S_ATTACK, S_BLOCK, S_HURT,
                       S_KO, S_DUCK, S_JUMP, FWD, BACK, BLOCK, IDLE, PUNCH,
                       KICK)
    from .articulated import ArticulatedDuel, HIGH_KICK_SWITCH
    from . import limbs
except ImportError:  # script import
    from duel import (FPS, WALK_SPEED, S_IDLE, S_ATTACK, S_BLOCK, S_HURT,
                      S_KO, S_DUCK, S_JUMP, FWD, BACK, BLOCK, IDLE, PUNCH,
                      KICK)
    from articulated import ArticulatedDuel, HIGH_KICK_SWITCH
    import limbs

JUMP, DUCK = 6, 7
N_ACTIONS = 8
ACTION_NAMES = ("FWD", "BACK", "PUNCH", "KICK", "BLOCK", "IDLE",
                "JUMP", "DUCK")

GRAVITY = 9.8
JUMP_V = 2.55          # m/s -> apex 0.33 m, ~31 frames airborne
AIR_CONTROL = 0.55     # horizontal speed multiplier while airborne
AIR_DRAG = 0.82        # per-frame vx decay while airborne
AIR_HEIGHT = 0.08      # above this a KICK is a flying kick

TECH = {
    PUNCH: limbs.ART_MOVES[PUNCH],
    KICK: limbs.ART_MOVES[KICK],
    limbs.HIGH_KICK: limbs.ART_MOVES[limbs.HIGH_KICK],
    limbs.LOW_KICK: limbs.ART_MOVES[limbs.LOW_KICK],
    limbs.FLYING: limbs.ART_MOVES[limbs.FLYING],
}

OBS_DIM = 29   # 23 articulated dims + own/opp height, vertical speed, duck flags


class FreeformDuel(ArticulatedDuel):
    def reset(self, seed=None):
        self.jumps = 0
        self.low_hits = 0
        self.flying_hits = 0
        self.air_frames = 0
        return super().reset(seed)

    # ------------------------------------------------------------------- pose
    def pose_of(self, f):
        spec = TECH.get(f.move) if f.state == S_ATTACK else None
        return limbs.pose(f.facing, f.x, f.state, f.move, f.phase, f.timer,
                          spec, self.frames, root_y=f.y)

    def spec_of(self, f, move):
        # Low kicks / flying kicks are not in WEAPONS[...]["attacks"]; the
        # inherited tick has to be able to look up their frame windows.
        return TECH[move]

    # ----------------------------------------------------------------- action
    def _apply_action(self, me, opp, act, idx, rewards):
        if not me.alive:
            return
        me.facing = 1 if opp.x >= me.x else -1
        if me.state in (S_ATTACK, S_HURT):
            return

        if act == JUMP:
            if me.y <= 0.0:
                self.jumps += 1
                me.vy = JUMP_V
                me.y = 1e-4
                me.state, me.timer = S_JUMP, 2
                me.vx *= 0.7
            return
        if act == DUCK:
            if me.y <= 0.0:
                me.state, me.timer = S_DUCK, 2
                me.vx = 0.0
            return
        if act in (FWD, BACK) and me.y > 0.0:
            # Air control is deliberately weaker than walking, otherwise a
            # jump is just a faster dash with no risk.
            spd = WALK_SPEED * me.speed_mul() * me.speed_buff * AIR_CONTROL
            me.vx = (spd if act == FWD else -spd * 0.85) * me.facing
            me.state = S_JUMP
            return

        if act in (PUNCH, KICK) and me.cooldown <= 0:
            dist = abs(opp.x - me.x)
            move = PUNCH
            if act == KICK:
                if me.y > AIR_HEIGHT:
                    move = limbs.FLYING
                elif me.state == S_DUCK:
                    move = limbs.LOW_KICK
                elif dist <= HIGH_KICK_SWITCH:
                    move = limbs.HIGH_KICK
                else:
                    move = KICK
            windup, active, recover, reach, dmg = TECH[move]
            self.n_swings[idx] += 1
            me.move = move
            me.state = S_ATTACK
            me.phase = "windup"
            me.timer = windup
            me.landed = False
            if me.y <= 0.0:
                me.vx = 0.0
            return

        super()._apply_action(me, opp, act, idx, rewards)

    # -------------------------------------------------------------- observation
    def _fighter_obs(self, me, other):
        base = super()._fighter_obs(me, other)
        # The base obs normalises state by 5; with DUCK/JUMP (6/7) that leaves
        # the [0, 1] band, so rescale to 6 here.
        base[3] = me.state / 6.0
        base[9] = other.state / 6.0
        extra = np.array([
            me.y / 2.0, me.vy / 4.0,
            other.y / 2.0, other.vy / 4.0,
            1.0 if me.state == S_DUCK else 0.0,
            1.0 if other.state == S_DUCK else 0.0,
        ], dtype=np.float32)
        return np.concatenate([base, extra])

    # ---------------------------------------------------------------- physics
    def _integrate(self):
        super()._integrate()
        for f in (self.p0, self.p1):
            if f.y > 0.0 or f.vy > 0.0:
                self.air_frames += 1
                f.vx *= AIR_DRAG
                f.vy -= GRAVITY / FPS
                f.y += f.vy / FPS
                if f.y <= 0.0:
                    f.y, f.vy = 0.0, 0.0
                    if f.state == S_JUMP:
                        f.state, f.timer = S_IDLE, 1

    def _tick(self, f):
        if f.state == S_JUMP and f.y > 0.0:
            if f.cooldown > 0:
                f.cooldown -= 1
            return
        if f.state == S_JUMP:
            f.state, f.timer = S_IDLE, 1
        super()._tick(f)

    # ------------------------------------------------------------------ combat
    def _hit_info(self, me, opp, seg):
        if not opp.alive or opp.state == S_KO:
            return None
        blocked = self._guard_hit(opp, seg)
        part, mul, clearance = limbs.closest_hurt(self.pose_of(opp), seg)
        if clearance > 0.0:
            return None
        return part, mul, blocked, me.move, TECH[me.move][4]

    def _apply_limb_hit(self, me, opp, opp_idx, part, mul, blocked, move, dmg,
                        rewards):
        if not blocked:
            if move == limbs.LOW_KICK:
                self.low_hits += 1
            elif move == limbs.FLYING:
                self.flying_hits += 1
        super()._apply_limb_hit(me, opp, opp_idx, part, mul, blocked, move,
                                dmg, rewards)

    # -------------------------------------------------------------------- info
    def _advance(self, a0, a1):
        rewards, done, info = super()._advance(a0, a1)
        info["jumps"] = self.jumps
        info["low_hits"] = self.low_hits
        info["flying_hits"] = self.flying_hits
        info["air_frames"] = self.air_frames
        return rewards, done, info
