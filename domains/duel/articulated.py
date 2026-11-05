"""Articulated duel: same rules as ``Duel`` but contact is limb geometry.

Differences from the original environment
-----------------------------------------
* Punch / kick hit with a **capsule** (forearm-fist / shin-foot) instead of a
  horizontal ``dist <= reach`` check.
* Hits are part-aware: head 1.35x, torso 1.0x, leg 0.75x damage.
* Two active limbs that meet in neutral space **clash**: both moves cancel,
  nobody takes damage.
* Blocking is a real forearm guard; the attack capsule must reach it.
* The observation grows from 15 to 23 dims: hand/foot endpoints for both
  fighters, relative to me.

Locomotion is still 1-D (x axis) -- the point of this domain is to make limb
contact correct before opening the whole body up to 2-D platforming physics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DUEL_DIR = Path(__file__).resolve().parent
if str(DUEL_DIR) not in sys.path:
    sys.path.insert(0, str(DUEL_DIR))

try:  # package import
    from .duel import Duel, MAX_HP, S_ATTACK, S_BLOCK, S_HURT, S_KO
    from . import limbs
    from .limbs import ART_MOVES, HIGH_KICK
except ImportError:  # script import
    from duel import Duel, MAX_HP, S_ATTACK, S_BLOCK, S_HURT, S_KO
    import limbs
    from limbs import ART_MOVES, HIGH_KICK

OBS_DIM = 23
CLASH_STUN = 8
CLASH_KNOCKBACK = 0.9
HIGH_KICK_SWITCH = 0.84   # inside this distance a KICK becomes a head kick
HEAD_STUN = 6             # extra hit-stun on an unblocked head hit


class ArticulatedDuel(Duel):
    def __init__(self, seed=None, hp=MAX_HP, max_frames=1800, start_gap=2.2,
                 env_cfg=None, weapons=("none", "none")):
        super().__init__(seed=seed, hp=hp, max_frames=max_frames,
                         start_gap=start_gap, env_cfg=env_cfg, weapons=weapons)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None):
        self.clashes = 0
        self.head_hits = 0
        self.part_hits = {"head": 0, "torso": 0, "leg": 0, "guard": 0}
        return super().reset(seed)

    # ------------------------------------------------------------------- pose
    def pose_of(self, f):
        spec = ART_MOVES.get(f.move) if f.state == S_ATTACK else None
        return limbs.pose(f.facing, f.x, f.state, f.move, f.phase,
                          f.timer, spec, self.frames)

    # ------------------------------------------------------------- techniques
    def _apply_action(self, me, opp, act, idx, rewards):
        """Pick the kick technique from range, then defer everything else.

        ``KICK`` is a single button: at range it is a front kick (mae geri);
        inside 0.84 m it becomes a head kick (mawashi geri), which is slower,
        shorter and hits the head for 1.35x.  Keeping this inside the engine
        means the policy still has six actions and the technique choice is
        deterministic from the observation.
        """
        if not me.alive:
            return
        me.facing = 1 if opp.x >= me.x else -1
        if me.state in (S_ATTACK, S_HURT):
            return
        if act in (limbs.PUNCH, limbs.KICK) and me.cooldown <= 0:
            dist = abs(opp.x - me.x)
            move = (HIGH_KICK if act == limbs.KICK and dist <= HIGH_KICK_SWITCH
                    else act)
            windup, active, recover, reach, dmg = ART_MOVES[move]
            self.n_swings[idx] += 1
            me.move = move
            me.state = S_ATTACK
            me.phase = "windup"
            me.timer = windup
            me.landed = False
            me.vx = 0.0
            return
        super()._apply_action(me, opp, act, idx, rewards)

    # -------------------------------------------------------------- observation
    def _fighter_obs(self, me, other):
        base = me.obs(other)
        pm = self.pose_of(me)
        po = self.pose_of(other)
        extra = np.array([
            (pm["hand"][0] - me.x) / 2.0, pm["hand"][1] / 2.0,
            (pm["foot"][0] - me.x) / 2.0, pm["foot"][1] / 2.0,
            (po["hand"][0] - me.x) / 2.0, po["hand"][1] / 2.0,
            (po["foot"][0] - me.x) / 2.0, po["foot"][1] / 2.0,
        ], dtype=np.float32)
        return np.concatenate([base, extra])

    def _obs(self):
        return self._fighter_obs(self.p0, self.p1), \
            self._fighter_obs(self.p1, self.p0)

    # ---------------------------------------------------------------- combat
    def _attack_seg(self, me):
        return limbs.attack_segments(self.pose_of(me), me.move)

    def _guard_hit(self, opp, seg):
        if opp.state != S_BLOCK:
            return False
        return any(limbs.capsule_overlap(seg, g)
                   for g in limbs.guard_capsules(self.pose_of(opp)))

    def _combat(self, rewards):
        pending = []
        for me, opp, idx in ((self.p0, self.p1, 0), (self.p1, self.p0, 1)):
            if me.state == S_ATTACK and me.phase == "active" and not me.landed:
                pending.append((me, opp, idx, self._attack_seg(me)))

        if not pending:
            return

        # A limb that already reaches a hurtbox hits; it does not clash.  Only
        # limbs meeting in neutral space collide with each other.
        hits_body = [limbs.closest_hurt(self.pose_of(opp), seg)[2] <= 0.0
                     for (_me, opp, _idx, seg) in pending]

        if len(pending) == 2 and not any(hits_body) \
                and limbs.capsule_overlap(pending[0][3], pending[1][3]):
            for (me, _opp, _idx, _seg) in pending:
                me.landed = True
                me.state = S_HURT
                me.timer = CLASH_STUN
                me.vx = -me.facing * CLASH_KNOCKBACK
                me.phase = ""
                me.move = None
            self.clashes += 1
            self.events.append((self.frames, 0, 0.0, False, -1,
                                abs(self.p1.x - self.p0.x), "clash"))
            return

        # Resolve all connecting limbs against pre-hit state, then apply them.
        # This makes close-range simultaneous hits trade instead of letting
        # the first processed fighter cancel the second's active frame.
        resolved = []
        for me, opp, idx, seg in pending:
            if me.state != S_ATTACK:
                continue  # the other side's hit interrupted this attack
            hit = self._hit_info(me, opp, seg)
            if hit is not None:
                resolved.append((me, opp, idx, hit))
        for me, opp, idx, (part, mul, blocked, move, dmg) in resolved:
            self._apply_limb_hit(me, opp, idx, part, mul, blocked, move, dmg,
                                 rewards)

    def _hit_info(self, me, opp, seg):
        if not opp.alive or opp.state == S_KO:
            return None
        blocked = self._guard_hit(opp, seg)
        part, mul, clearance = limbs.closest_hurt(self.pose_of(opp), seg)
        if clearance > 0.0:
            return None
        dmg = ART_MOVES[me.move][4]
        return part, mul, blocked, me.move, dmg

    def _apply_limb_hit(self, me, opp, opp_idx, part, mul, blocked, move, dmg,
                        rewards):
        me.landed = True
        block_mul = me.block_mul() if blocked else 1.0
        want = dmg * mul * block_mul * me.power_buff * opp.armor_buff
        hp_before = opp.hp
        opp.hp = max(0.0, opp.hp - want)
        real = hp_before - opp.hp
        self.damage_dealt[1 - opp_idx] += real

        self.part_hits["guard" if blocked else part] += 1
        self.events.append((self.frames, move, real, blocked,
                            1 - opp_idx, abs(opp.x - me.x), part))

        rewards[1 - opp_idx] -= real * 0.5
        rewards[opp_idx] += real * 0.8

        if not blocked:
            if part == "head":
                self.head_hits += 1
            opp.state = S_HURT
            opp.timer = 10 + me.hurt_add() + (HEAD_STUN if part == "head" else 0)
            opp.vx = me.facing * self.knockback
            opp.phase = ""
            opp.move = None

    # ------------------------------------------------------------------- info
    def _advance(self, a0, a1):
        rewards, done, info = super()._advance(a0, a1)
        info["clashes"] = self.clashes
        info["head_hits"] = self.head_hits
        info["parts"] = dict(self.part_hits)
        return rewards, done, info
