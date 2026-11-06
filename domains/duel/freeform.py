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

**SUB** (action 8) is a substitution (kawarimi): 4 frames of invisibility,
then a teleport to 0.9 m behind the opponent, for 40 of 100 chakra and a
72-frame cooldown.  Chakra regenerates slowly and is also earned by landing
hits, so the escape competes with the rest of the kit instead of being free.

Eight actions -> nine, 23 observations -> 33 (height and vertical speed for
both fighters, two duck flags, chakra and substitution cooldown for both).
No scripting of *when* to use them; the policy has to find that itself.

With ``arena3d=True`` the sagittal plane gains a depth axis: ``Fighter.z/vz``,
a tenth action ``SIDESTEP`` (a committed 18-frame side-hop: 9 frames out,
9 back, 0.45 m), depth-aware hit detection (an attack whiffs when the bodies
are more than 0.32 m apart in z), and four extra observations.  The 2D path
is byte-for-byte unchanged when the flag is off -- z stays zero and the
dimensions stay 35/9.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

DUEL_DIR = Path(__file__).resolve().parent
if str(DUEL_DIR) not in sys.path:
    sys.path.insert(0, str(DUEL_DIR))

try:  # package import
    from .duel import (FPS, WALK_SPEED, S_IDLE, S_ATTACK, S_BLOCK, S_HURT,
                       S_KO, S_DUCK, S_JUMP, FWD, BACK, BLOCK, IDLE, PUNCH,
                       KICK, BODY_GAP)
    from .articulated import ArticulatedDuel, HIGH_KICK_SWITCH
    from . import limbs, physics, space3d
except ImportError:  # script import
    from duel import (FPS, WALK_SPEED, S_IDLE, S_ATTACK, S_BLOCK, S_HURT,
                      S_KO, S_DUCK, S_JUMP, FWD, BACK, BLOCK, IDLE, PUNCH,
                      KICK, BODY_GAP)
    from articulated import ArticulatedDuel, HIGH_KICK_SWITCH
    import limbs, physics, space3d

JUMP, DUCK, SUB = 6, 7, 8
N_ACTIONS = 9
ACTION_NAMES = ("FWD", "BACK", "PUNCH", "KICK", "BLOCK", "IDLE",
                "JUMP", "DUCK", "SUB")
S_SUB = 8

# ---- third axis (--arena3d) ------------------------------------------------
# A sidestep moves the body out of the attack line without changing the
# distance.  It is a committed 18-frame side-hop (out 9 frames, back 9) rather
# than a persistent offset: without the automatic return, a 1-D fight would
# turn into two parallel duels that can never touch -- and a 35-frame manual
# return could not punish the whiff it had just created.
SIDESTEP = 9
S_SIDESTEP = 9
N_ACTIONS_3D = 10
ACTION_NAMES_3D = ACTION_NAMES + ("SIDESTEP",)
Z_LINE = 0.15            # inside this band a sidestep goes outward
Z_HIT_WIDTH = 0.32       # depth offset that makes a straight attack whiff
SWEEP_Z_WIDTH = 0.55     # a lateral sweep still connects this far off the line
MASS_EXTRA = 2           # +[self mass, opponent mass] when mass_obs is on
SIDESTEP_V = 3.0         # m/s
SIDESTEP_FRAMES = 9      # out 9 frames, back 9: 0.45 m, 18-frame commitment
DEPTH_HALF = 1.2         # arena half-depth in metres
Z_DECEL = physics.KINETIC_MU * physics.GRAVITY
OBS_DIM_3D = 39          # 35 + z / vz for both fighters
EVADE_REWARD = 0.8       # opt-in payoff for a sidestep that made active air whiff

CHAKRA_MAX = 100.0
CHAKRA_REGEN = 0.6        # per frame -> full bar in ~2.8 s
CHAKRA_HIT_GAIN = 12.0    # attacker earns chakra by landing
CHAKRA_BLOCK_GAIN = 4.0   # defender earns a little by guarding
SUB_COST = 40.0
SUB_STARTUP = 4           # invulnerable frames before the swap
SUB_RECOVER = 6           # punishable frames after landing behind
SUB_COOLDOWN = 72
SUB_DIST = 0.9            # metres behind the opponent
SUB_TOTAL = SUB_STARTUP + SUB_RECOVER

# Guard is a second health bar that only blocks pay into.  Holding block is
# strong (chip damage is only 28%), so it needs a cost: a guard meter.  Empty
# it and the next hit goes through and stuns for half a second.  This is the
# standard "guard gauge" of 2D fighters (Street Fighter, Guilty Gear), and it
# gives the emotion layer something to act on: a frightened fighter's hands
# shake, so their guard drains faster.
GUARD_MAX = 100.0
GUARD_REGEN = 1.2         # per frame while not blocking
GUARD_BLOCK_REGEN = 0.2   # barely recovers while turtling
GUARD_COST = 2.4          # guard points per point of chip damage (x composure)
GUARD_BREAK_STUN = 28     # punish window after a guard break
GUARD_RESET = 35.0        # guard after breaking (do not instantly re-break)

# Counter-hit: hitting a fighter during windup/recovery is the classic
# footsies reward.  It makes timing matter beyond raw damage.
COUNTER_MUL = 1.35
COUNTER_STUN = 8

GRAVITY = 9.8
JUMP_V = 2.55          # m/s -> apex 0.33 m, ~31 frames airborne
AIR_CONTROL = 0.55     # horizontal speed multiplier while airborne
AIR_DRAG = 0.82        # per-frame vx decay while airborne
AIR_HEIGHT = 0.08      # above this a KICK is a flying kick

# Wall slam: being knocked into the arena wall is a second, smaller hit.
# Without it the wall is only a boundary; with it, cornering the opponent
# becomes a strategic goal instead of an accident.  Damage scales with the
# impact speed, so a clean heavy kick near the wall hurts more than a jab.
WALL_SLAM_PER_SPEED = 1.2   # damage per m/s of incoming knockback
WALL_SLAM_MIN = 1.5
WALL_SLAM_MAX = 7.0
WALL_SLAM_STUN = 8          # extra hurt frames after the bounce
WALL_SLAM_BONUS = 1.0       # attacker shaping on top of the damage reward

TECH = {
    PUNCH: limbs.ART_MOVES[PUNCH],
    KICK: limbs.ART_MOVES[KICK],
    limbs.HIGH_KICK: limbs.ART_MOVES[limbs.HIGH_KICK],
    limbs.LOW_KICK: limbs.ART_MOVES[limbs.LOW_KICK],
    limbs.FLYING: limbs.ART_MOVES[limbs.FLYING],
    limbs.SWEEP: limbs.ART_MOVES[limbs.SWEEP],
}

OBS_DIM = 35   # 29 + chakra/sub cooldown + guard for both fighters
GUARD_EXTRA = 2
EMOTION_EXTRA = 8   # four affect dims for each fighter


class FreeformDuel(ArticulatedDuel):
    def __init__(self, *args, arena3d=False, **kwargs):
        # Set before Duel.__init__, whose reset()/obs reads both.
        self.arena3d = bool(arena3d)
        ec = dict(kwargs.get("env_cfg") or {})
        self.depth_half = float(ec.get("depth", DEPTH_HALF))
        # Off by default: a 40-round A/B showed the shaping makes 3D worse
        # (.375 vs .500), see FINDINGS 22.5.  Opt in with --evade-reward.
        self.evade_reward = float(ec.get("evade_reward", 0.0))
        # Weight class is part of the body, so a shared policy cannot tell
        # whether it is driving the 50 kg or the 100 kg fighter unless mass
        # is in the observation.  Off by default to keep old obs shapes.
        self.mass_obs = bool(ec.get("mass_obs", False))
        # Weight-class curriculum: when set to (lo, hi), both bodies are
        # resampled every episode so the policy sees a continuum of builds.
        self.rand_mass = ec.get("rand_mass") or None
        super().__init__(*args, **kwargs)

    def reset(self, seed=None):
        self.jumps = 0
        self.low_hits = 0
        self.flying_hits = 0
        self.air_frames = 0
        self.subs = 0
        self.sub_avoided = 0
        self.guard_breaks = 0
        self.counters = 0
        self.wall_bounces = 0
        self.wall_slams = 0
        self.sidesteps = 0
        self.sweeps = 0
        self.evades = 0
        self.hitstop = 0
        self.knockback_peak = 0.0
        self._slam_rewards = [0.0, 0.0]
        self._evade_rewards = [0.0, 0.0]
        obs = super().reset(seed)
        if self.rand_mass:
            lo, hi = float(self.rand_mass[0]), float(self.rand_mass[1])
            rng = np.random.default_rng(0 if seed is None else int(seed) + 7919)
            m0, m1 = rng.uniform(lo, hi, size=2)
            self.p0.mass, self.p1.mass = float(m0), float(m1)
        for f in (self.p0, self.p1):
            f.chakra = CHAKRA_MAX
            f.sub_cd = 0
            f.invuln = 0
            f.guard = GUARD_MAX
            f.z = 0.0
            f.vz = 0.0
            f.side_dir = 1.0
            f.evade_paid = False
            if self.physics:
                f.mass_speed = physics.mass_speed_factor(f.mass)
        return self._obs()

    # ------------------------------------------------------------------- pose
    def _build_pose(self, f):
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
        if act == SUB:
            self._try_substitution(me, opp)
            return
        if me.state == S_SIDESTEP:
            # Side-hop cancel: a kick during the hop becomes a lateral sweep.
            # This is the attack the hop was missing -- it covers the depth the
            # hop created and slides the body back onto the line as it swings.
            if self.arena3d and act == KICK and me.cooldown <= 0:
                self._begin_sweep(me, idx)
            return
        if me.state in (S_ATTACK, S_HURT, S_SUB):
            return

        if act == SIDESTEP:
            self._try_sidestep(me)
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
            spd = (WALK_SPEED * me.speed_mul() * me.speed_buff
                   * me.mass_speed * AIR_CONTROL)
            me.vx = (spd if act == FWD else -spd * 0.85) * me.facing
            me.state = S_JUMP
            return

        if self.physics and act == IDLE:
            # No half-speed hack: Coulomb friction is the only thing that
            # should stop a sliding body.
            me.state, me.timer = S_IDLE, 1
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
            me.evade_paid = False
            if me.y <= 0.0:
                me.vx = 0.0
            return

        super()._apply_action(me, opp, act, idx, rewards)

    # ------------------------------------------------------------- substitution
    def _try_substitution(self, me, opp):
        """Spend chakra to vanish and reappear behind the opponent.

        Allowed from idle, walk, block, duck, jump and *hit-stun* -- never
        mid-attack, so an unsafe swing cannot be cancelled for free.
        """
        if me.state in (S_ATTACK, S_KO, S_SUB):
            return
        if me.sub_cd > 0 or me.chakra < SUB_COST:
            return
        me.chakra -= SUB_COST
        me.sub_cd = SUB_COOLDOWN
        me.state, me.phase, me.timer = S_SUB, "vanish", SUB_STARTUP
        me.invuln = SUB_STARTUP + 2
        me.move = None
        me.landed = False
        me.vx = 0.0

    def _swap_behind(self, f, opp):
        facing = 1 if opp.x >= f.x else -1
        new_x = float(np.clip(opp.x + facing * SUB_DIST, -self.half, self.half))
        f.x = new_x
        f.vx = 0.0
        f.z, f.vz = 0.0, 0.0        # the swap happens on the centre line
        f.facing = 1 if opp.x >= f.x else -1
        self.subs += 1

    # -------------------------------------------------------------- observation
    def _fighter_obs(self, me, other):
        base = super()._fighter_obs(me, other)
        # The base obs normalises state by 5; with DUCK/JUMP/SUB/SIDESTEP
        # (6/7/8/9) that leaves the [0, 1] band, so rescale to 9 here.
        base[3] = me.state / 9.0
        base[9] = other.state / 9.0
        extra = np.array([
            me.y / 2.0, me.vy / 4.0,
            other.y / 2.0, other.vy / 4.0,
            1.0 if me.state == S_DUCK else 0.0,
            1.0 if other.state == S_DUCK else 0.0,
            getattr(me, "chakra", CHAKRA_MAX) / CHAKRA_MAX,
            getattr(other, "chakra", CHAKRA_MAX) / CHAKRA_MAX,
            getattr(me, "sub_cd", 0) / SUB_COOLDOWN,
            getattr(other, "sub_cd", 0) / SUB_COOLDOWN,
            getattr(me, "guard", GUARD_MAX) / GUARD_MAX,
            getattr(other, "guard", GUARD_MAX) / GUARD_MAX,
        ], dtype=np.float32)
        if self.arena3d:
            extra = np.concatenate([extra, np.array([
                me.z / self.depth_half, me.vz / 4.0,
                other.z / self.depth_half, other.vz / 4.0,
            ], dtype=np.float32)])
        if self.emotion:
            extra = np.concatenate([
                extra, me.affect.vector(), other.affect.vector()])
        if self.mass_obs:
            extra = np.concatenate([extra, np.array([
                me.mass / 140.0, other.mass / 140.0,
            ], dtype=np.float32)])
        return np.concatenate([base, extra])

    # -------------------------------------------------------------------- 3D
    def _try_sidestep(self, me):
        """Committed side-hop: out for 9 frames, then back for 9.

        It is a complete maneuver rather than a persistent offset.  A toggle
        that left the body displaced for 35+ frames was safe but could not
        punish a whiff -- the opponent's recovery ended before you got back
        into range.  Eighteen frames is long enough to read as a dodge and
        short enough to counterattack.
        """
        if me.y > 0.0:
            return
        if abs(me.z) <= Z_LINE:
            direction = 1.0
        else:
            direction = -1.0 if me.z > 0.0 else 1.0
        me.side_dir = direction
        me.vz = direction * SIDESTEP_V
        me.state, me.phase, me.timer = S_SIDESTEP, "out", SIDESTEP_FRAMES
        me.vx = 0.0
        self.sidesteps += 1

    def _begin_sweep(self, me, idx):
        """Start the lateral sweep that cancels the rest of a side-hop."""
        windup, _active, _recover, _reach, _dmg = TECH[limbs.SWEEP]
        self.n_swings[idx] += 1
        me.move = limbs.SWEEP
        me.state, me.phase, me.timer = S_ATTACK, "windup", windup
        me.landed = False
        me.evade_paid = False
        me.vx = 0.0
        me.vz = 0.0

    def _step_z(self, f):
        if not self.arena3d:
            return
        if f.state == S_ATTACK and f.move == limbs.SWEEP:
            # The sweep itself carries the body back to the centre line at the
            # same speed as the hop; the leg covers the offset while it does.
            step = SIDESTEP_V / FPS
            if abs(f.z) <= step:
                f.z, f.vz = 0.0, 0.0
            else:
                f.z -= math.copysign(step, f.z)
            return
        if f.state == S_SIDESTEP:
            # Scripted hop profile; friction would shorten the outbound leg.
            f.z += f.vz / FPS
            return
        decel = Z_DECEL / FPS
        if abs(f.vz) <= max(decel, 0.05):
            f.vz = 0.0
        else:
            f.vz -= math.copysign(decel, f.vz)
        f.z += f.vz / FPS
        if f.z > self.depth_half:
            f.z, f.vz = self.depth_half, 0.0
        elif f.z < -self.depth_half:
            f.z, f.vz = -self.depth_half, 0.0

    def _off_line(self, a, b, width=Z_HIT_WIDTH):
        """True when the depth gap is wide enough for a straight attack to miss."""
        return self.arena3d and space3d.sidestep_clears(a.z, b.z, width)

    def bodies_separate(self):
        return not self._off_line(self.p0, self.p1)

    def _limbs_align(self):
        return not self._off_line(self.p0, self.p1)

    # ---------------------------------------------------------------- physics
    def _integrate(self):
        if self.physics:
            # Mass/friction/wall model.  Body separation is weighted by inverse
            # mass, so walking into the heavier fighter moves you, not them.
            for i, f in enumerate((self.p0, self.p1)):
                if f.y > 0.0 or f.vy > 0.0:
                    self.air_frames += 1
                physics.step(f, 1.0 / FPS)
                self._step_z(f)
                impact = abs(f.vx)
                if physics.collide_walls(f, self.half, self.walls,
                                         bounce=(f.state == S_HURT)):
                    self.wall_bounces += 1
                    self._wall_slam(i, f, 1 - i, impact)
            if (not self._wall_blocks(self.p0.x, self.p1.x)
                    and self.bodies_separate()):
                physics.separate_pair(self.p0, self.p1, BODY_GAP)
            return
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
            self._step_z(f)

    def _wall_slam(self, victim_idx, victim, attacker_idx, impact):
        """A knockback that reaches the arena wall is a second hit.

        The bounce itself only happens for hurt bodies (see physics.collide_
        walls), so simply reversing the velocity is not enough: without
        consequences the wall is scenery.  Damage is proportional to the
        incoming speed, and the attacker gets both the damage credit and a
        small shaping bonus, which is what makes cornering worth pursuing.
        """
        dmg = min(WALL_SLAM_MAX, max(WALL_SLAM_MIN,
                                    WALL_SLAM_PER_SPEED * float(impact)))
        hp_before = victim.hp
        victim.hp = max(0.0, victim.hp - dmg)
        real = hp_before - victim.hp
        if real <= 0.0:
            return
        self.damage_dealt[attacker_idx] += real
        victim.timer = max(victim.timer, WALL_SLAM_STUN)
        self.wall_slams += 1
        self._slam_rewards[attacker_idx] += real * 0.8 + WALL_SLAM_BONUS
        self._slam_rewards[victim_idx] -= real * 0.5
        self.events.append((self.frames, "wall", real, False, attacker_idx,
                            abs(victim.x), "wall"))

    def _tick(self, f):
        if f.state == S_SIDESTEP:
            f.timer -= 1
            if f.timer <= 0:
                if f.phase == "out":
                    f.phase, f.timer = "back", SIDESTEP_FRAMES
                    f.vz = -f.side_dir * SIDESTEP_V
                else:
                    f.state, f.phase, f.timer = S_IDLE, "", 1
                    f.z, f.vz = 0.0, 0.0
            return
        if f.state == S_SUB:
            f.timer -= 1
            if f.timer <= 0:
                if f.phase == "vanish":
                    opp = self.p1 if f is self.p0 else self.p0
                    self._swap_behind(f, opp)
                    f.phase, f.timer = "land", SUB_RECOVER
                else:
                    f.state, f.phase, f.timer = S_IDLE, "", 1
            return
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
        # A punch travels along x at the attacker's z plane.  If the defender
        # has stepped out of that plane the limb passes through empty space,
        # even though the 2D silhouette overlaps.  With env_cfg["evade_reward"]
        # enabled, a side-hop that made an *active* swing whiff also pays the
        # dodger -- avoiding damage alone was not enough to make depth a skill,
        # but the payoff turned out to hurt (FINDINGS 22.5), so it is opt-in.
        width = SWEEP_Z_WIDTH if me.move == limbs.SWEEP else Z_HIT_WIDTH
        if self._off_line(me, opp, width):
            if (self.evade_reward > 0.0
                    and opp.state == S_SIDESTEP and me.state == S_ATTACK
                    and me.phase == "active" and not me.evade_paid):
                me.evade_paid = True
                idx = 0 if me is self.p0 else 1
                self._evade_rewards[1 - idx] += self.evade_reward
                self.evades += 1
            return None
        if getattr(opp, "invuln", 0) > 0:
            # The substitution window: the limb passes through empty air.
            _part, _mul, clearance = limbs.closest_hurt(self.pose_of(opp), seg)
            if clearance <= 0.0:
                self.sub_avoided += 1
            me.landed = True
            return None
        blocked = self._guard_hit(opp, seg)
        part, mul, clearance = limbs.closest_hurt(self.pose_of(opp), seg)
        if clearance > 0.0:
            return None
        return part, mul, blocked, me.move, TECH[me.move][4]

    def counter_bonus(self, me, opp):
        # Hitting a windup or recovery is a counter.  Hitting an *active* limb
        # is a trade and is handled by the clash path, so it is excluded.
        if opp.state == S_ATTACK and opp.phase in ("windup", "recover"):
            self.counters += 1
            return COUNTER_MUL, COUNTER_STUN
        return 1.0, 0

    def _apply_limb_hit(self, me, opp, opp_idx, part, mul, blocked, move, dmg,
                        rewards):
        if not blocked:
            if move == limbs.LOW_KICK:
                self.low_hits += 1
            elif move == limbs.FLYING:
                self.flying_hits += 1
            elif move == limbs.SWEEP:
                self.sweeps += 1
        real = super()._apply_limb_hit(me, opp, opp_idx, part, mul, blocked,
                                       move, dmg, rewards)
        opp.vz = 0.0        # a hit cancels any lateral movement
        if self.physics:
            # The base hit model assigns a fixed knockback speed.  With mass
            # on, overwrite it with an impulse so heavy/clean hits carry more
            # and the attacker gets pushed back too.
            opp.vx = 0.0
            physics.strike(me, opp, real, blocked=blocked)
            self.knockback_peak = max(self.knockback_peak, abs(opp.vx))
            if opp.alive:
                self.hitstop = max(self.hitstop, physics.hitstop_frames(
                    real, head=(part == "head"), blocked=blocked))
        me.chakra = min(CHAKRA_MAX, me.chakra + CHAKRA_HIT_GAIN)
        if blocked:
            opp.chakra = min(CHAKRA_MAX, opp.chakra + CHAKRA_BLOCK_GAIN)
            # Fear makes the hands shake: low composure drains guard faster.
            composure = (opp.affect.composure if getattr(opp, "affect", None)
                         is not None else 1.0)
            fear_mul = 1.0 + (1.0 - composure) * 0.6
            opp.guard -= GUARD_COST * dmg * mul * fear_mul
            if opp.guard <= 0.0:
                self._guard_break(me, opp)

    def _guard_break(self, me, opp):
        opp.guard = GUARD_RESET
        opp.state = S_HURT
        opp.timer = GUARD_BREAK_STUN
        if self.physics:
            opp.vx = 0.0
            physics.strike(me, opp, 6.0)
        else:
            opp.vx = me.facing * self.knockback * 0.6
        opp.phase = ""
        opp.move = None
        if getattr(opp, "affect", None) is not None:
            opp.affect.guard_break()
        self.guard_breaks += 1

    # -------------------------------------------------------------------- info
    def _match_info(self):
        return {
            "jumps": self.jumps, "low_hits": self.low_hits,
            "flying_hits": self.flying_hits, "air_frames": self.air_frames,
            "subs": self.subs, "sub_avoided": self.sub_avoided,
            "guard_breaks": self.guard_breaks, "counters": self.counters,
            "wall_bounces": self.wall_bounces, "wall_slams": self.wall_slams,
            "sidesteps": self.sidesteps, "sweeps": self.sweeps,
            "evades": self.evades,
            "knockback_peak": round(self.knockback_peak, 3),
            "guard": (self.p0.guard, self.p1.guard),
            "hp": (self.p0.hp, self.p1.hp),
        }

    def _advance(self, a0, a1):
        if self.physics and self.hitstop > 0:
            # Impact freeze: neither body, timer nor emotion advances.  The
            # frame counter still ticks, so episode length stays bounded.
            self.hitstop -= 1
            self.frames += 1
            return [0.0, 0.0], False, self._match_info()
        self._slam_rewards = [0.0, 0.0]
        self._evade_rewards = [0.0, 0.0]
        for f in (self.p0, self.p1):
            if f.alive:
                f.chakra = min(CHAKRA_MAX, f.chakra + CHAKRA_REGEN)
                regen = (GUARD_BLOCK_REGEN if f.state == S_BLOCK
                         else GUARD_REGEN)
                f.guard = min(GUARD_MAX, f.guard + regen)
                if f.sub_cd > 0:
                    f.sub_cd -= 1
                if f.invuln > 0:
                    f.invuln -= 1
        rewards, done, info = super()._advance(a0, a1)
        # Wall-slam damage happens inside _integrate, after _combat has already
        # distributed this frame's hit rewards; fold it in afterwards.
        rewards[0] += self._slam_rewards[0] + self._evade_rewards[0]
        rewards[1] += self._slam_rewards[1] + self._evade_rewards[1]
        info.update(self._match_info())
        return rewards, done, info
