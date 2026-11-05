"""Tests for the articulated limb model and contact rules.

Run from the repository root::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.duel import limbs
from domains.duel.articulated import ArticulatedDuel, HEAD_STUN, OBS_DIM
from domains.duel.limbs import ART_MOVES, HIGH_KICK
from domains.duel.duel import S_ATTACK, S_BLOCK, S_HURT, S_IDLE, PUNCH, KICK


def freeze(dist):
    env = ArticulatedDuel(seed=0, max_frames=600, start_gap=1.0)
    env.reset(0)
    env.p0.x, env.p1.x = -dist * 0.5, dist * 0.5
    env.p0.vx = env.p1.vx = 0.0
    env.p0.facing, env.p1.facing = 1, -1
    env.p0.state, env.p0.timer = S_IDLE, 1
    env.p1.state, env.p1.timer = S_IDLE, 1
    return env


def attack(env, who=0, move=PUNCH):
    f = env.p0 if who == 0 else env.p1
    f.state = S_ATTACK
    f.move = move
    f.phase = "active"
    f.timer = 3
    f.landed = False
    f.facing = 1 if who == 0 else -1


class TestKinematics(unittest.TestCase):
    def test_punch_extends_forward(self):
        p = limbs.pose(1, 0.0, S_ATTACK, PUNCH, "active", 3, ART_MOVES[PUNCH], 0)
        self.assertAlmostEqual(p["hand"][0], 0.750, delta=0.02)
        self.assertAlmostEqual(p["hand"][1], 1.643, delta=0.02)
        # facing -1 mirrors x
        q = limbs.pose(-1, 0.0, S_ATTACK, PUNCH, "active", 3, ART_MOVES[PUNCH], 0)
        self.assertAlmostEqual(q["hand"][0], -0.750, delta=0.02)

    def test_kick_reaches_farther_than_punch(self):
        pk = limbs.pose(1, 0.0, S_ATTACK, KICK, "active", 3, ART_MOVES[KICK], 0)
        pp = limbs.pose(1, 0.0, S_ATTACK, PUNCH, "active", 3, ART_MOVES[PUNCH], 0)
        self.assertGreater(pk["foot"][0], pp["hand"][0] + 0.15)

    def test_head_kick_is_high_and_short(self):
        p = limbs.pose(1, 0.0, S_ATTACK, HIGH_KICK, "active", 3,
                       ART_MOVES[HIGH_KICK], 0)
        pk = limbs.pose(1, 0.0, S_ATTACK, KICK, "active", 3, ART_MOVES[KICK], 0)
        self.assertGreater(p["foot"][1], 1.60)
        self.assertLess(p["foot"][0], pk["foot"][0] - 0.30)

    def test_attack_timing_matches_real_techniques(self):
        # jab < front kick < head kick: speed is paid for with damage
        jab, front, head = ART_MOVES[PUNCH], ART_MOVES[KICK], ART_MOVES[HIGH_KICK]
        self.assertLess(jab[0], front[0])
        self.assertLess(front[0], head[0])
        self.assertLess(jab[4], front[4])
        self.assertLess(front[4], head[4])

    def test_duel_and_limb_specs_agree(self):
        from domains.duel.duel import WEAPONS
        self.assertEqual(WEAPONS["none"]["attacks"][HIGH_KICK],
                         ART_MOVES[HIGH_KICK])

    def test_extension_curve(self):
        spec = ART_MOVES[KICK]
        windup, active, recover = spec[0], spec[1], spec[2]
        self.assertAlmostEqual(
            limbs.extension(S_ATTACK, "windup", windup, spec), 0.0)
        self.assertAlmostEqual(
            limbs.extension(S_ATTACK, "active", active, spec), 1.0)
        self.assertAlmostEqual(
            limbs.extension(S_ATTACK, "recover", 1, spec), 0.0)
        self.assertEqual(limbs.extension(S_IDLE, "", 1, spec), 0.0)


class TestCapsuleMath(unittest.TestCase):
    def test_crossing_segments_touch(self):
        self.assertEqual(limbs.segment_distance((0, 0), (1, 1),
                                               (0, 1), (1, 0)), 0.0)

    def test_parallel_segments(self):
        d = limbs.segment_distance((0, 0), (1, 0), (0, 2), (1, 2))
        self.assertAlmostEqual(d, 2.0)

    def test_point_segment(self):
        self.assertAlmostEqual(limbs.point_segment_dist((0.5, 1), (0, 0), (1, 0),), 1.0)

    def test_hurt_part_multipliers(self):
        p = limbs.pose(1, 0.0, S_IDLE, None, "", 1, None, 0)
        parts = {part: mul for (_a, _b, _r, part, mul) in limbs.hurt_capsules(p)}
        self.assertEqual(parts["head"], 1.35)
        self.assertEqual(parts["torso"], 1.0)
        self.assertEqual(parts["leg"], 0.75)


class TestContactRules(unittest.TestCase):
    def test_punch_hits_head_at_close_range(self):
        env = freeze(0.9)
        attack(env, 0, PUNCH)
        env._combat([0.0, 0.0])
        self.assertLess(env.p1.hp, 100.0)
        self.assertEqual(env.part_hits["head"], 1)
        self.assertEqual(env.head_hits, 1)
        self.assertEqual(env.p1.timer, 10 + env.p0.hurt_add() + HEAD_STUN)

    def test_punch_misses_at_range(self):
        env = freeze(1.35)
        attack(env, 0, PUNCH)
        env._combat([0.0, 0.0])
        self.assertEqual(env.p1.hp, 100.0)

    def test_kick_reaches_where_punch_cannot(self):
        env = freeze(1.25)
        attack(env, 0, PUNCH)
        env._combat([0.0, 0.0])
        self.assertEqual(env.p1.hp, 100.0)

        env = freeze(1.25)
        attack(env, 0, KICK)
        env._combat([0.0, 0.0])
        self.assertLess(env.p1.hp, 100.0)

    def test_block_absorbs_and_keeps_guard(self):
        env = freeze(0.9)
        env.p1.state, env.p1.timer = S_BLOCK, 2
        attack(env, 0, PUNCH)
        env._combat([0.0, 0.0])
        self.assertLess(100.0 - env.p1.hp, ART_MOVES[PUNCH][4])  # reduced damage
        self.assertEqual(env.p1.state, S_BLOCK)
        self.assertEqual(env.part_hits["guard"], 1)

    def test_kick_button_picks_head_kick_in_the_pocket(self):
        env = freeze(0.80)
        env._apply_action(env.p0, env.p1, KICK, 0, [0.0, 0.0])
        self.assertEqual(env.p0.move, HIGH_KICK)

        env = freeze(1.20)
        env._apply_action(env.p0, env.p1, KICK, 0, [0.0, 0.0])
        self.assertEqual(env.p0.move, KICK)

    def test_head_kick_lands_on_head(self):
        env = freeze(0.80)
        attack(env, 0, HIGH_KICK)
        env._combat([0.0, 0.0])
        self.assertLess(env.p1.hp, 100.0)
        self.assertEqual(env.part_hits["head"], 1)
        # 1.35x on the head, and the head hit adds extra hit-stun
        self.assertAlmostEqual(100.0 - env.p1.hp, ART_MOVES[HIGH_KICK][4] * 1.35,
                               places=4)
        self.assertEqual(env.p1.timer, 10 + env.p0.hurt_add() + HEAD_STUN)

    def test_clash_does_not_hurt(self):
        env = freeze(1.5)
        attack(env, 0, PUNCH)
        attack(env, 1, PUNCH)
        env._combat([0.0, 0.0])
        self.assertEqual(env.p1.hp, 100.0)
        self.assertEqual(env.p0.hp, 100.0)
        self.assertEqual(env.clashes, 1)
        self.assertEqual(env.p0.state, S_HURT)
        self.assertEqual(env.p1.state, S_HURT)

    def test_body_contact_trades_instead_of_clashing(self):
        env = freeze(0.7)
        attack(env, 0, PUNCH)
        attack(env, 1, PUNCH)
        env._combat([0.0, 0.0])
        self.assertLess(env.p0.hp, 100.0)
        self.assertLess(env.p1.hp, 100.0)
        self.assertEqual(env.clashes, 0)


class TestEnvironment(unittest.TestCase):
    def test_obs_is_23_and_finite(self):
        env = ArticulatedDuel(seed=1, max_frames=300)
        o0, o1 = env.reset(1)
        self.assertEqual(OBS_DIM, 23)
        self.assertEqual(len(o0), OBS_DIM)
        self.assertEqual(o0.shape, o1.shape)
        self.assertTrue(bool(np.isfinite(o0).all()))

    def test_determinism_and_damage_accounting(self):
        def rollout():
            env = ArticulatedDuel(seed=7, max_frames=400, start_gap=2.0)
            o0, o1 = env.reset(7)
            obs = [o0.copy(), o1.copy()]
            for t in range(400):
                a0, a1 = t % 6, (t + 2) % 6
                o0, o1, _r0, _r1, done, info = env.step(a0, a1)
                obs.append(o0.copy())
                if done:
                    break
            return env, obs, info

        env_a, obs_a, info_a = rollout()
        env_b, obs_b, info_b = rollout()
        self.assertEqual(len(obs_a), len(obs_b))
        for a, b in zip(obs_a, obs_b):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(info_a["winner"], info_b["winner"])
        # total damage dealt must equal the HP that disappeared
        lost = (100.0 - env_a.p0.hp) + (100.0 - env_a.p1.hp)
        self.assertAlmostEqual(sum(env_a.damage_dealt), lost, places=6)
        # ...and each counter must belong to the side that threw the attack
        self.assertAlmostEqual(env_a.damage_dealt[0], 100.0 - env_a.p1.hp,
                               places=6)
        self.assertAlmostEqual(env_a.damage_dealt[1], 100.0 - env_a.p0.hp,
                               places=6)
        for side in (0, 1):
            self.assertAlmostEqual(
                sum(e[2] for e in env_a.events if e[4] == side),
                env_a.damage_dealt[side], places=6)
        self.assertIn("clashes", info_a)
        self.assertIn("parts", info_a)


if __name__ == "__main__":
    unittest.main()
