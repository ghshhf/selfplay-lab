"""Tests for the freeform (vertical-axis) duel.

Run from the repository root::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.duel import limbs
from domains.duel.duel import (S_IDLE, S_ATTACK, S_BLOCK, S_DUCK, S_JUMP,
                              S_HURT, FWD, BACK, PUNCH, KICK, BLOCK, IDLE)
from domains.duel.freeform import (FreeformDuel, JUMP, DUCK, SUB, S_SUB,
                                   SIDESTEP, S_SIDESTEP, N_ACTIONS, OBS_DIM,
                                   OBS_DIM_3D, N_ACTIONS_3D, ACTION_NAMES_3D, TECH,
                                   AIR_HEIGHT, Z_HIT_WIDTH, SIDESTEP_V,
                                   JUMP_V, GRAVITY, CHAKRA_MAX, CHAKRA_REGEN,
                                   SUB_COST, SUB_DIST, SUB_COOLDOWN,
                                   GUARD_REGEN, ACTION_NAMES)


def freeze(dist=0.9, emotion=False, physics=False):
    env = FreeformDuel(seed=0, max_frames=900, start_gap=dist,
                       emotion=emotion, physics=physics)
    env.reset(0)
    env.p0.x, env.p1.x = -dist * 0.5, dist * 0.5
    env.p0.vx = env.p1.vx = 0.0
    env.p0.facing, env.p1.facing = 1, -1
    env.p0.state, env.p0.timer = S_IDLE, 1
    env.p1.state, env.p1.timer = S_IDLE, 1
    return env


def idle(env, frames, a0=IDLE, a1=IDLE):
    info = {"winner": -1}
    for _ in range(frames):
        _o0, _o1, _r0, _r1, done, info = env.step(a0, a1)
        if done:
            break
    return info


def pose_clearance(move, target_pose, root_y=0.0, crouch=None):
    spec = TECH[move]
    attacker = limbs.pose(1, 0.0, S_ATTACK, move, "active", 1, spec, 0,
                          root_y=root_y)
    seg = limbs.attack_segments(attacker, move)
    return limbs.closest_hurt(target_pose, seg)


class TestDimensions(unittest.TestCase):
    def test_observation_and_action_space(self):
        env = FreeformDuel(seed=1, max_frames=300)
        o0, o1 = env.reset(1)
        self.assertEqual(len(o0), OBS_DIM)
        self.assertEqual(len(o1), OBS_DIM)
        self.assertEqual(OBS_DIM, 35)
        self.assertEqual(N_ACTIONS, 9)
        self.assertEqual(len(ACTION_NAMES), 9)
        self.assertIn(limbs.LOW_KICK, TECH)
        self.assertIn(limbs.FLYING, TECH)


class TestVerticalPhysics(unittest.TestCase):
    def test_jump_arc_and_landing(self):
        env = freeze(0.9)
        heights = []
        for i in range(45):
            idle(env, 1, a0=JUMP if i == 0 else IDLE)
            heights.append(env.p0.y)
        apex = max(heights)
        expected = JUMP_V ** 2 / (2 * GRAVITY)
        self.assertAlmostEqual(apex, expected, delta=0.03)
        self.assertEqual(env.p0.y, 0.0)
        self.assertEqual(env.p0.state, S_IDLE)

    def test_no_double_jump(self):
        env = freeze(0.9)
        idle(env, 1, a0=JUMP)
        apex_before = env.p0.vy
        idle(env, 6, a0=JUMP)
        self.assertLess(env.p0.vy, apex_before + 1e-6)
        self.assertEqual(env.jumps, 1)

    def test_air_control_is_slower_than_walking(self):
        env = freeze(0.9)
        idle(env, 1, a0=JUMP)
        idle(env, 1, a0=FWD)
        air_vx = abs(env.p0.vx)
        env.reset(0)
        env.p0.x, env.p1.x = -0.5, 0.5
        idle(env, 1, a0=FWD)
        walk_vx = abs(env.p0.vx)
        self.assertLess(air_vx, walk_vx)


class TestDuckGeometry(unittest.TestCase):
    def test_duck_lowers_head_below_jab(self):
        stand = limbs.pose(1, 0.7, S_IDLE, None, "", 1, None, 0)
        duck = limbs.pose(1, 0.7, S_DUCK, None, "", 1, None, 0)
        self.assertAlmostEqual(duck["head"][1], 1.19, delta=0.02)
        self.assertLess(duck["head"][1], stand["head"][1] - 0.35)

        jab = pose_clearance(limbs.PUNCH, duck)
        self.assertGreater(jab[2], 0.0)     # whiffs over the head
        head_kick = pose_clearance(limbs.HIGH_KICK, duck)
        self.assertGreater(head_kick[2], 0.0)

    def test_front_kick_still_hits_a_duck(self):
        duck = limbs.pose(1, 0.7, S_DUCK, None, "", 1, None, 0)
        hit = pose_clearance(limbs.KICK, duck)
        self.assertLess(hit[2], 0.0)
        self.assertEqual(hit[0], "torso")

    def test_duck_state_expires(self):
        env = freeze(0.9)
        idle(env, 1, a0=DUCK)
        self.assertEqual(env.p0.state, S_DUCK)
        idle(env, 3, a0=IDLE)
        self.assertEqual(env.p0.state, S_IDLE)


class TestTechniqueSelection(unittest.TestCase):
    def test_kick_from_duck_becomes_low_kick(self):
        env = freeze(0.8)
        idle(env, 1, a0=DUCK)
        idle(env, 1, a0=KICK)
        self.assertEqual(env.p0.move, limbs.LOW_KICK)

    def test_kick_in_air_becomes_flying(self):
        env = freeze(0.9)
        idle(env, 1, a0=JUMP)
        idle(env, 3, a0=IDLE)
        self.assertGreater(env.p0.y, AIR_HEIGHT)
        idle(env, 1, a0=KICK)
        self.assertEqual(env.p0.move, limbs.FLYING)

    def test_grounded_kick_still_switches_by_range(self):
        far = freeze(1.10)
        idle(far, 1, a0=KICK)
        self.assertEqual(far.p0.move, limbs.KICK)
        near = freeze(0.70)
        idle(near, 1, a0=KICK)
        self.assertEqual(near.p0.move, limbs.HIGH_KICK)


class TestRockPaperScissors(unittest.TestCase):
    def test_low_kick_ignores_guard(self):
        env = freeze(0.78)
        env.p1.state, env.p1.timer = S_BLOCK, 2
        idle(env, 1, a0=DUCK)
        idle(env, 1, a0=KICK)
        info = idle(env, 20)
        self.assertGreaterEqual(env.low_hits, 1)
        last = [e for e in env.events if e[1] == limbs.LOW_KICK][-1]
        self.assertFalse(last[3])                 # blocked == False
        self.assertEqual(last[6], "leg")
        self.assertGreater(env.p1.hp, 0.0)
        self.assertEqual(info["low_hits"], env.low_hits)

    def test_jump_clears_a_low_kick(self):
        stand = limbs.pose(1, 0.7, S_IDLE, None, "", 1, None, 0)
        airborne = limbs.pose(1, 0.7, S_JUMP, None, "", 1, None, 0, root_y=0.31)
        hit_ground = pose_clearance(limbs.LOW_KICK, stand)
        hit_air = pose_clearance(limbs.LOW_KICK, airborne)
        self.assertLess(hit_ground[2], 0.0)
        self.assertGreater(hit_air[2], 0.0)

    def test_flying_kick_reaches_the_head(self):
        stand = limbs.pose(1, 0.75, S_IDLE, None, "", 1, None, 0)
        hit = pose_clearance(limbs.FLYING, stand, root_y=0.31)
        self.assertLess(hit[2], 0.0)
        self.assertEqual(hit[0], "head")


class TestEnvironmentIntegrity(unittest.TestCase):
    def test_damage_reconciles(self):
        env = freeze(0.8)
        for _ in range(200):
            _o0, _o1, _r0, _r1, done, info = env.step(KICK, BLOCK)
            if done:
                break
        dealt = env.damage_dealt
        lost = (env.hp0 - env.p0.hp) + (env.hp0 - env.p1.hp)
        self.assertAlmostEqual(sum(dealt), lost, delta=0.05)

    def test_deterministic_replay(self):
        def run():
            env = FreeformDuel(seed=11, max_frames=400, start_gap=1.0)
            env.reset(11)
            rng = np.random.default_rng(3)
            for _ in range(200):
                a0, a1 = int(rng.integers(N_ACTIONS)), int(rng.integers(N_ACTIONS))
                done = env.step(a0, a1)[4]
                if done:
                    break
            return env.p0.hp, env.p1.hp, env.frames, env.jumps
        self.assertEqual(run(), run())

    def test_info_exposes_free_actions(self):
        env = freeze(0.9)
        idle(env, 1, a0=JUMP)
        idle(env, 25)
        info = idle(env, 1)
        for key in ("jumps", "low_hits", "flying_hits", "air_frames"):
            self.assertIn(key, info)
        self.assertGreater(info["air_frames"], 10)


class PlayModeTest(unittest.TestCase):
    """The human seat has a reaction budget; the headless loop must obey it."""

    def test_human_does_not_act_before_reaction_window(self):
        from domains.duel.play import HumanPlayer
        h = HumanPlayer(style="counter", reaction=12)
        h.reset(0)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[12] = 0.3          # opponents 1.2 m away
        obs[3] = 0.0
        for _ in range(11):
            self.assertEqual(h.act(obs), IDLE)
        acts = {h.act(obs) for _ in range(30)}
        self.assertTrue(acts.issubset({FWD, BACK, PUNCH, KICK, BLOCK, IDLE}))

    def test_auto_rounds_terminate_and_attribute_damage(self):
        import argparse
        from domains.duel import play as P
        args = argparse.Namespace(rounds=2, seed=1, reaction=12, style="counter",
                                  quiet=True, json="", max_frames=900)
        arena = FreeformDuel(seed=1, max_frames=900)
        opp = P.RandomOpponent(seed=1)
        summary = P.auto(args, arena, opp)
        self.assertEqual(len(summary), 2)
        for s in summary:
            self.assertGreater(s["frames"], 0)
            # no healing in the default arena, so HP lost == damage taken
            self.assertAlmostEqual(s["dealt"][0], 100.0 - s["hp"][1], places=4)
            self.assertAlmostEqual(s["dealt"][1], 100.0 - s["hp"][0], places=4)

    def test_renderer_draws_bars_and_both_fighters(self):
        from domains.duel.play import render
        env = freeze(1.0)
        text = render(env, "test", "neural")
        self.assertIn("HP", text)
        self.assertIn("O", text)
        self.assertIn("#", text)
        self.assertGreaterEqual(len(text.splitlines()), 8)


class TestSubstitution(unittest.TestCase):
    def test_swap_lands_behind_and_spends_chakra(self):
        env = freeze(1.6)
        env.p0.chakra = CHAKRA_MAX
        idle(env, 1, a0=SUB)
        self.assertEqual(env.p0.state, S_SUB)
        self.assertAlmostEqual(env.p0.chakra, CHAKRA_MAX - SUB_COST, delta=1.0)
        idle(env, 6)
        self.assertGreater(env.p0.x, env.p1.x)      # reappeared behind
        self.assertAlmostEqual(abs(env.p0.x - env.p1.x), SUB_DIST, delta=0.05)
        self.assertEqual(env.subs, 1)
        self.assertGreater(env.p0.sub_cd, 0)

    def test_refuses_without_chakra(self):
        env = freeze(1.6)
        env.p0.chakra = SUB_COST - 5.0
        idle(env, 1, a0=SUB)
        self.assertNotEqual(env.p0.state, S_SUB)
        self.assertLess(env.p0.chakra, SUB_COST)

    def test_cooldown_blocks_a_second_swap(self):
        env = freeze(1.6)
        idle(env, 1, a0=SUB)
        idle(env, 12)                                # finish the animation
        env.p0.chakra = CHAKRA_MAX
        x_before = env.p0.x
        idle(env, 1, a0=SUB)
        self.assertNotEqual(env.p0.state, S_SUB)
        self.assertAlmostEqual(env.p0.x, x_before, delta=1e-6)
        self.assertEqual(env.subs, 1)
        self.assertLess(env.p0.sub_cd, SUB_COOLDOWN)

    def test_startup_frames_make_the_attack_whiff(self):
        env = freeze(0.9)
        idle(env, 1, a0=PUNCH)                       # jab windup is 4 frames
        hp = env.p1.hp
        idle(env, 2, a1=SUB)                         # vanish just before impact
        idle(env, 14)                                # through the landing
        self.assertEqual(env.p1.hp, hp)              # no damage went through
        self.assertGreaterEqual(env.sub_avoided, 1)
        self.assertEqual(env.p1.state, S_IDLE)

    def test_attack_cannot_be_cancelled_into_substitution(self):
        env = freeze(1.6)
        env.p0.chakra = CHAKRA_MAX
        idle(env, 1, a0=PUNCH)
        idle(env, 1, a0=SUB)                         # mid-attack, must be ignored
        self.assertEqual(env.p0.state, S_ATTACK)
        self.assertAlmostEqual(env.p0.chakra, CHAKRA_MAX, delta=1.0)

    def test_chakra_regenerates(self):
        env = freeze(1.6)
        env.p0.chakra = 0.0
        idle(env, 10)
        self.assertAlmostEqual(env.p0.chakra, 10 * CHAKRA_REGEN, delta=0.2)

    def test_landing_a_hit_pays_chakra(self):
        env = freeze(0.9)
        env.p0.chakra = 10.0
        before = env.p0.chakra
        idle(env, 1, a0=PUNCH)
        idle(env, 8)
        self.assertGreater(env.p0.chakra, before + CHAKRA_REGEN * 8)


class TestArena3D(unittest.TestCase):
    def _env(self, dist=0.9, evade_reward=0.0):
        env = FreeformDuel(seed=0, max_frames=900, start_gap=dist, arena3d=True,
                           env_cfg={"evade_reward": evade_reward})
        env.reset(0)
        env.p0.x, env.p1.x = -dist * 0.5, dist * 0.5
        env.p0.vx = env.p1.vx = 0.0
        env.p0.facing, env.p1.facing = 1, -1
        env.p0.state, env.p0.timer = S_IDLE, 1
        env.p1.state, env.p1.timer = S_IDLE, 1
        return env

    def test_space_grows_by_depth(self):
        env = self._env()
        o0, o1 = env.reset(0)
        self.assertEqual(len(o0), OBS_DIM_3D)
        self.assertEqual(len(o1), OBS_DIM_3D)
        self.assertEqual(N_ACTIONS_3D, 10)
        self.assertEqual(ACTION_NAMES_3D[-1], "SIDESTEP")

    def test_sidestep_leaves_and_returns_to_the_line(self):
        env = self._env()
        idle(env, 1, a0=SIDESTEP)
        idle(env, 7)                                  # 8 frames into the hop
        self.assertGreater(env.p0.z, Z_HIT_WIDTH)     # out of the attack line
        self.assertTrue(env._off_line(env.p0, env.p1))
        idle(env, 12)                                 # hop finished
        self.assertAlmostEqual(env.p0.z, 0.0)
        self.assertEqual(env.p0.state, S_IDLE)
        self.assertFalse(env._off_line(env.p0, env.p1))

    def test_straight_punch_whiffs_across_depth(self):
        env = self._env()
        env.p1.z = 0.6
        hp = env.p1.hp
        idle(env, 1, a0=PUNCH)
        idle(env, 10)
        self.assertEqual(env.p1.hp, hp)
        self.assertEqual(env.n_swings[0], 1)          # the swing did happen
        env.p1.z = 0.0
        idle(env, 25, a0=PUNCH)
        self.assertLess(env.p1.hp, hp)

    def test_separation_is_skipped_across_depth(self):
        env = self._env(0.4)
        env.p0.x, env.p1.x = -0.2, 0.2
        env.p1.z = 0.6
        idle(env, 1)
        self.assertAlmostEqual(env.p0.x, -0.2, delta=1e-6)
        self.assertAlmostEqual(env.p1.x, 0.2, delta=1e-6)

    def test_depth_edges_stop_the_body(self):
        env = self._env()
        env.p0.vz = 5.0
        idle(env, 30)
        self.assertEqual(env.p0.z, env.depth_half)
        self.assertEqual(env.p0.vz, 0.0)

    def test_active_whiff_across_depth_pays_the_dodger(self):
        env = self._env(evade_reward=0.8)
        env.p0.state, env.p0.phase, env.p0.move = S_ATTACK, "active", PUNCH
        env.p0.timer, env.p0.landed, env.p0.evade_paid = 3, False, False
        env.p1.state, env.p1.phase, env.p1.timer = S_SIDESTEP, "out", 5
        env.p1.z, env.p1.vz = 0.4, 0.0
        hp = env.p1.hp
        _o0, _o1, r0, r1, _done, info = env.step(IDLE, IDLE)
        self.assertEqual(env.p1.hp, hp)
        self.assertEqual(env.evades, 1)
        self.assertGreater(r1, 0.0)
        self.assertEqual(r0, 0.0)

    def test_evade_reward_is_off_by_default(self):
        env = self._env()
        env.p0.state, env.p0.phase, env.p0.move = S_ATTACK, "active", PUNCH
        env.p0.timer, env.p0.landed, env.p0.evade_paid = 3, False, False
        env.p1.state, env.p1.phase, env.p1.timer = S_SIDESTEP, "out", 5
        env.p1.z, env.p1.vz = 0.4, 0.0
        _o0, _o1, r0, r1, _done, info = env.step(IDLE, IDLE)
        self.assertEqual(env.evades, 0)
        self.assertEqual(r1, 0.0)

    def test_depth_is_visible_in_the_observation(self):
        env = self._env()
        obs0, _obs1 = env.reset(0)
        idle(env, 1, a0=SIDESTEP)
        idle(env, 7)
        obs0, _obs1 = env._obs()
        self.assertGreater(float(obs0[OBS_DIM]), 0.3)

    def test_kick_during_side_hop_becomes_a_sweep(self):
        env = self._env()
        idle(env, 1, a0=SIDESTEP)
        idle(env, 3)
        idle(env, 1, a0=KICK)
        self.assertEqual(env.p0.move, limbs.SWEEP)
        self.assertEqual(env.p0.state, S_ATTACK)
        self.assertEqual(env.sidesteps, 1)

    def test_sweep_returns_to_the_attack_line(self):
        env = self._env()
        idle(env, 1, a0=SIDESTEP)
        idle(env, 3)
        idle(env, 1, a0=KICK)
        self.assertGreater(env.p0.z, 0.0)
        idle(env, 25)
        self.assertEqual(env.p0.z, 0.0)
        self.assertEqual(env.p0.state, S_IDLE)

    def test_sweep_connects_across_the_depth_it_created(self):
        env = self._env()
        idle(env, 1, a0=SIDESTEP)
        idle(env, 3)
        idle(env, 1, a0=KICK)
        idle(env, 12)
        self.assertGreaterEqual(env.sweeps, 1)
        self.assertLess(env.p1.hp, 100.0)

    def test_sweep_z_tolerance_exceeds_straight_attacks(self):
        env = self._env()
        env.p0.state, env.p0.phase = S_ATTACK, "active"
        env.p0.move, env.p0.timer, env.p0.landed = limbs.SWEEP, 3, False
        env.p0.z, env.p1.z = 0.45, 0.0
        hp = env.p1.hp
        idle(env, 1)
        self.assertLess(env.p1.hp, hp)             # 0.40 > 0.32: a punch would whiff
        self.assertEqual(env.sweeps, 1)

    def test_straight_punch_at_the_same_offset_whiffs(self):
        env = self._env()
        env.p0.state, env.p0.phase = S_ATTACK, "active"
        env.p0.move, env.p0.timer, env.p0.landed = PUNCH, 2, False
        env.p0.z, env.p1.z = 0.45, 0.0
        hp = env.p1.hp
        idle(env, 1)
        self.assertEqual(env.p1.hp, hp)


class TestEmotionAndGuard(unittest.TestCase):
    def test_emotion_extends_observation(self):
        env = FreeformDuel(seed=2, max_frames=300, emotion=True)
        o0, o1 = env.reset(2)
        self.assertEqual(len(o0), OBS_DIM + 8)
        info = idle(env, 1)
        self.assertIn("emotion", info)
        for fighter in (env.p0, env.p1):
            vec = fighter.affect.vector()
            self.assertEqual(vec.shape, (4,))
            self.assertTrue(np.all(vec >= 0.0) and np.all(vec <= 1.0))

    def test_hurt_raises_fear_and_lowers_composure(self):
        from domains.duel.affect import Affect
        a = Affect()
        composure = a.composure
        a.hit_taken(18.0, head=True)
        self.assertLess(a.composure, composure)
        self.assertGreater(a.fear, 0.0)
        self.assertGreater(a.anger, 0.0)
        a.hit_landed(12.0)
        self.assertGreater(a.momentum, 0.0)

    def test_low_health_and_contagion_raise_fear(self):
        from domains.duel.affect import Affect
        a = Affect()
        before = a.fear
        for _ in range(30):
            a.tick(0.10, opp_fear=0.0, opp_anger=1.0, opp_momentum=0.0)
        self.assertGreater(a.fear, before + 0.05)

    def test_guard_drains_and_breaks(self):
        env = freeze(0.9)
        env.p1.guard = 2.0
        idle(env, 40, a0=KICK, a1=BLOCK)
        self.assertGreaterEqual(env.guard_breaks, 1)
        self.assertAlmostEqual(env.p1.guard, 35.0, delta=GUARD_REGEN * 30)

    def test_guard_regenerates_when_not_blocking(self):
        env = freeze(2.0)
        env.p0.guard = 20.0
        idle(env, 20)
        self.assertGreater(env.p0.guard, 20.0 + 10.0)

    def test_counter_window_is_rewarded(self):
        env = freeze(0.9)
        env.p0.state, env.p0.phase, env.p0.timer = S_ATTACK, "recover", 10
        mul, stun = env.counter_bonus(env.p1, env.p0)
        self.assertGreater(mul, 1.0)
        self.assertGreater(stun, 0)
        self.assertEqual(env.counters, 1)
        env.p0.phase = "active"
        self.assertEqual(env.counter_bonus(env.p1, env.p0), (1.0, 0))

    def test_fear_weakens_guard(self):
        # Two identical blocked hits, one calm and one frightened.
        calm = freeze(0.9, emotion=True)
        calm.p1.affect.composure = 1.0
        calm.p1.guard = 50.0
        idle(calm, 40, a0=KICK, a1=BLOCK)
        scared = freeze(0.9, emotion=True)
        scared.p1.affect.composure = 0.0
        scared.p1.guard = 50.0
        idle(scared, 40, a0=KICK, a1=BLOCK)
        self.assertLess(scared.p1.guard, calm.p1.guard)


class TestRigidBodyPhysics(unittest.TestCase):
    def test_jump_still_works_with_physics(self):
        env = freeze(0.9, physics=True)
        heights = []
        for i in range(45):
            idle(env, 1, a0=JUMP if i == 0 else IDLE)
            heights.append(env.p0.y)
        self.assertGreater(max(heights), 0.25)
        self.assertEqual(env.p0.y, 0.0)

    def test_wall_bounce_is_physical(self):
        env = freeze(0.9, physics=True)
        env.p0.x, env.p0.vx = env.half - 0.01, 1.5
        env.p0.state, env.p0.timer = S_HURT, 20
        info = idle(env, 2)
        self.assertGreaterEqual(env.wall_bounces, 1)
        self.assertLessEqual(env.p0.x, env.half)
        self.assertLess(env.p0.vx, 0.0)
        self.assertEqual(info["wall_bounces"], env.wall_bounces)

    def test_walking_into_wall_stops_instead_of_bouncing(self):
        env = freeze(0.9, physics=True)
        env.p0.x, env.p1.x = 3.37, 3.99
        env.p0.vx = 2.5
        env.p0.state, env.p0.timer = S_IDLE, 1
        env.p1.state, env.p1.timer = S_IDLE, 1
        env.p0.facing, env.p1.facing = 1, -1
        idle(env, 20, a0=FWD)
        self.assertEqual(env.wall_bounces, 0)
        self.assertLessEqual(env.p0.x, env.half)

    def test_separation_is_mass_weighted(self):
        env = freeze(0.9, physics=True)
        env.p0.x, env.p1.x = 0.0, 0.5
        env.p0.mass, env.p1.mass = 40.0, 80.0
        env.p0.vx = env.p1.vx = 0.0
        x0, x1 = env.p0.x, env.p1.x
        idle(env, 1)
        self.assertGreater(abs(env.p0.x - x0), abs(env.p1.x - x1))
        self.assertAlmostEqual(env.p1.x - env.p0.x, 0.62, places=3)

    def test_hitstop_freezes_the_world(self):
        env = freeze(0.9, physics=True)
        env.hitstop = 3
        x0, timer0, frames0 = env.p0.x, env.p0.timer, env.frames
        idle(env, 1, a0=JUMP, a1=JUMP)
        self.assertEqual(env.p0.x, x0)
        self.assertEqual(env.p0.timer, timer0)
        self.assertEqual(env.p0.y, 0.0)
        self.assertEqual(env.frames, frames0 + 1)
        self.assertEqual(env.hitstop, 2)

    def test_knockback_travels_further_than_legacy(self):
        # A clean unblocked hit should slide the victim roughly 0.5-1.0 m
        # with Coulomb friction, instead of the legacy exponential 0.37 m.
        env = freeze(1.2, physics=True)
        idle(env, 60, a0=KICK, a1=IDLE)
        self.assertGreaterEqual(env.knockback_peak, 0.5)
        self.assertGreater(env.p1.x - 0.6, 0.1)

    def test_wall_slam_hurts_and_pays_the_attacker(self):
        env = freeze(0.9, physics=True)
        env.p1.x, env.p1.vx = env.half - 0.01, 2.5
        env.p1.state, env.p1.timer = S_HURT, 20
        hp0 = env.p1.hp
        _o0, _o1, r0, r1, _done, info = env.step(IDLE, IDLE)
        self.assertGreaterEqual(env.wall_slams, 1)
        self.assertLess(env.p1.hp, hp0)
        self.assertGreater(r0, 0.0)
        self.assertLess(r1, 0.0)
        self.assertGreaterEqual(info["wall_slams"], 1)
        self.assertEqual(env.events[-1][1], "wall")

    def test_walking_into_wall_does_not_slam(self):
        env = freeze(0.9, physics=True)
        env.p0.x, env.p0.vx = env.half - 0.01, 2.5
        env.p0.state, env.p0.timer = S_IDLE, 1
        idle(env, 3, a0=FWD)
        self.assertEqual(env.wall_slams, 0)

    def test_heavier_walks_slower(self):
        def walk(mass):
            env = FreeformDuel(seed=0, max_frames=900, start_gap=6.0,
                               physics=True, env_cfg={"mass0": mass})
            env.reset(0)
            env.p0.x, env.p1.x = -3.0, 3.0
            env.p0.facing, env.p1.facing = 1, -1
            env.p0.state = env.p1.state = S_IDLE
            start = env.p0.x
            idle(env, 20, a0=FWD)
            return env.p0.x - start

        self.assertGreater(walk(45.0), walk(110.0) * 1.2)

    def test_heavier_fighter_is_pushed_less(self):
        def hit_with_mass(m):
            env = FreeformDuel(seed=0, max_frames=900, start_gap=1.2,
                               physics=True, env_cfg={"mass1": m})
            env.reset(0)
            env.p0.x, env.p1.x = -0.6, 0.6
            env.p0.facing, env.p1.facing = 1, -1
            env.p0.state, env.p1.state = S_IDLE, S_IDLE
            start = env.p1.x
            idle(env, 60, a0=KICK, a1=IDLE)
            return env.knockback_peak, env.p1.x - start

        light_peak, light_slide = hit_with_mass(50.0)
        heavy_peak, heavy_slide = hit_with_mass(120.0)
        self.assertEqual(light_peak > heavy_peak * 2, True)
        self.assertGreater(light_slide, heavy_slide)


if __name__ == "__main__":
    unittest.main()
