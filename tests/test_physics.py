"""Tests for the mass/friction/wall/impulse physics layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.duel import physics


def body(x=0.0, y=0.0, vx=0.0, vy=0.0, mass=70.0):
    return SimpleNamespace(x=x, y=y, vx=vx, vy=vy, mass=mass)


class TestFriction(unittest.TestCase):
    def test_sliding_body_stops(self):
        b = body(vx=2.2)
        start = b.x
        for _ in range(180):  # 3 s
            physics.step(b)
        self.assertEqual(b.vx, 0.0)
        self.assertLess(abs(b.x - start), 1.2)
        self.assertGreater(abs(b.x - start), 0.4)

    def test_static_body_does_not_drift(self):
        b = body()
        physics.step(b)
        self.assertEqual(b.x, 0.0)
        self.assertEqual(b.vx, 0.0)


class TestVertical(unittest.TestCase):
    def test_jump_apex_and_landing(self):
        b = body(y=1e-4, vy=2.55)
        peak = 0.0
        landed_at = None
        for i in range(120):
            physics.step(b)
            peak = max(peak, b.y)
            if b.y == 0.0 and landed_at is None:
                landed_at = i
        # Semi-implicit Euler lands a little short of the analytic apex; the
        # point is that it is the right order of magnitude.
        self.assertGreater(peak, 0.28)
        self.assertLess(peak, 0.34)
        self.assertGreaterEqual(landed_at, 25)
        self.assertLessEqual(landed_at, 40)

    def test_landing_is_inelastic(self):
        b = body(y=0.01, vy=-2.0)
        physics.step(b)
        self.assertEqual(b.y, 0.0)
        self.assertEqual(b.vy, 0.0)


class TestWalls(unittest.TestCase):
    def test_arena_wall_bounces(self):
        b = body(x=3.99, vx=1.0)
        b.x += b.vx / physics.FPS
        bounced = physics.collide_walls(b, 4.0)
        self.assertTrue(bounced)
        self.assertAlmostEqual(b.x, 4.0)
        self.assertLess(b.vx, 0.0)
        self.assertAlmostEqual(abs(b.vx), physics.WALL_RESTITUTION, places=6)

    def test_slow_contact_stops_instead_of_bouncing(self):
        b = body(x=4.001, vx=0.1)
        bounced = physics.collide_walls(b, 4.0)
        self.assertFalse(bounced)
        self.assertEqual(b.vx, 0.0)

    def test_interior_wall_stops(self):
        b = body(x=1.0, vx=1.0)
        bounced = physics.collide_walls(b, 4.0, walls=[(1.2, 0.3)])
        self.assertFalse(bounced)
        self.assertEqual(b.vx, 0.0)
        self.assertLessEqual(b.x, 0.9)


class TestSeparation(unittest.TestCase):
    def test_overlap_resolved(self):
        a, b = body(x=0.0), body(x=0.5)
        overlap = physics.separate_pair(a, b, 0.62)
        self.assertAlmostEqual(overlap, 0.12)
        self.assertAlmostEqual(b.x - a.x, 0.62)

    def test_light_body_moves_more(self):
        a, b = body(x=0.0, mass=40.0), body(x=0.5, mass=80.0)
        before = (a.x, b.x)
        physics.separate_pair(a, b, 0.62)
        moved_a = abs(a.x - before[0])
        moved_b = abs(b.x - before[1])
        self.assertGreater(moved_a, moved_b)
        self.assertAlmostEqual(moved_a / moved_b, 2.0, places=6)

    def test_closing_velocity_is_resolved(self):
        a, b = body(x=0.0), body(x=0.55, vx=-1.0)
        p_before = physics.momentum(a) + physics.momentum(b)
        physics.separate_pair(a, b, 0.62)
        p_after = physics.momentum(a) + physics.momentum(b)
        self.assertAlmostEqual(p_before, p_after, places=9)
        self.assertGreater(b.vx, -1.0)
        self.assertLess(a.vx, 0.0)


class TestImpulse(unittest.TestCase):
    def test_strike_pushes_both_bodies(self):
        a, b = body(x=0.0), body(x=0.8)
        j = physics.strike(a, b, 11.0)
        self.assertGreater(j, 0.0)
        self.assertGreater(b.vx, 0.0)
        self.assertLess(a.vx, 0.0)
        self.assertAlmostEqual(b.vx, j / 70.0, places=9)

    def test_blocked_strike_is_weaker(self):
        a1, b1 = body(x=0.0), body(x=0.8)
        a2, b2 = body(x=0.0), body(x=0.8)
        physics.strike(a1, b1, 11.0)
        physics.strike(a2, b2, 11.0, blocked=True)
        self.assertLess(abs(b2.vx), abs(b1.vx))
        self.assertAlmostEqual(abs(b2.vx) / abs(b1.vx),
                               physics.BLOCK_IMPULSE, places=6)

    def test_heavier_target_moves_less(self):
        light, heavy = body(x=0.8, mass=50.0), body(x=0.8, mass=100.0)
        physics.strike(body(x=0.0), light, 11.0)
        physics.strike(body(x=0.0), heavy, 11.0)
        self.assertGreater(abs(light.vx), abs(heavy.vx))
        self.assertAlmostEqual(abs(light.vx) / abs(heavy.vx), 2.0, places=6)


class TestWeightClass(unittest.TestCase):
    def test_default_mass_is_neutral(self):
        self.assertEqual(physics.mass_speed_factor(70.0), 1.0)

    def test_heavy_is_slower_and_light_is_faster(self):
        self.assertLess(physics.mass_speed_factor(100.0), 1.0)
        self.assertGreater(physics.mass_speed_factor(50.0), 1.0)
        self.assertAlmostEqual(physics.mass_speed_factor(280.0), 0.5, places=9)

    def test_monotone_in_mass(self):
        # Same rule for both seats; only the kg value differs.
        speeds = [physics.mass_speed_factor(m)
                  for m in (40.0, 55.0, 70.0, 90.0, 120.0)]
        self.assertEqual(speeds, sorted(speeds, reverse=True))


class TestHitstop(unittest.TestCase):
    def test_scales_with_damage_and_blocks(self):
        self.assertEqual(physics.hitstop_frames(3.0), 2)
        self.assertGreater(physics.hitstop_frames(9.0),
                           physics.hitstop_frames(3.0))
        self.assertEqual(physics.hitstop_frames(20.0, head=True), 6)
        self.assertEqual(physics.hitstop_frames(20.0, blocked=True), 1)

    def test_bounded(self):
        self.assertLessEqual(physics.hitstop_frames(99.0, head=True), 6)


if __name__ == "__main__":
    unittest.main()
