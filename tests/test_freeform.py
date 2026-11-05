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
                              FWD, BACK, PUNCH, KICK, BLOCK, IDLE)
from domains.duel.freeform import (FreeformDuel, JUMP, DUCK, N_ACTIONS,
                                   OBS_DIM, TECH, AIR_HEIGHT, JUMP_V, GRAVITY)


def freeze(dist=0.9):
    env = FreeformDuel(seed=0, max_frames=900, start_gap=dist)
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
        self.assertEqual(OBS_DIM, 29)
        self.assertEqual(N_ACTIONS, 8)
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


if __name__ == "__main__":
    unittest.main()
