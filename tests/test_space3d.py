"""Tests for 3D geometry primitives (preparation for the 3D arena)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.duel.space3d import (Vec3, capsule_distance, capsules_overlap,
                                  isometric, orthographic,
                                  point_segment_distance,
                                  segment_segment_distance, sidestep_clears,
                                  sphere_overlap)


class TestVec3(unittest.TestCase):
    def test_arithmetic(self):
        a, b = Vec3(1, 2, 3), Vec3(3, 2, 1)
        self.assertEqual(a + b, Vec3(4, 4, 4))
        self.assertEqual(a - b, Vec3(-2, 0, 2))
        self.assertEqual(a * 2, Vec3(2, 4, 6))
        self.assertAlmostEqual(a.dot(b), 10.0)
        self.assertAlmostEqual(a.length(), 14 ** 0.5)


class TestSegmentDistance(unittest.TestCase):
    def test_point_on_segment(self):
        self.assertAlmostEqual(
            point_segment_distance(Vec3(0.5, 0, 0), Vec3(0, 0, 0),
                                   Vec3(1, 0, 0)), 0.0, places=9)

    def test_point_beyond_segment_clamps(self):
        self.assertAlmostEqual(
            point_segment_distance(Vec3(2, 0, 0), Vec3(0, 0, 0),
                                   Vec3(1, 0, 0)), 1.0, places=9)

    def test_parallel_segments(self):
        self.assertAlmostEqual(
            segment_segment_distance(Vec3(0, 0, 0), Vec3(1, 0, 0),
                                     Vec3(0, 1, 0), Vec3(1, 1, 0)),
            1.0, places=9)

    def test_crossing_segments_touch(self):
        self.assertAlmostEqual(
            segment_segment_distance(Vec3(-1, 0, 0), Vec3(1, 0, 0),
                                     Vec3(0, -1, 0), Vec3(0, 1, 0)),
            0.0, places=9)

    def test_skew_segments(self):
        # One segment on z=0, the other on z=2: never closer than 2.
        self.assertAlmostEqual(
            segment_segment_distance(Vec3(-1, 0, 0), Vec3(1, 0, 0),
                                     Vec3(0, -1, 2), Vec3(0, 1, 2)),
            2.0, places=9)


class TestCapsules(unittest.TestCase):
    def test_capsule_surface_distance(self):
        d = capsule_distance(Vec3(-1, 0, 0), Vec3(1, 0, 0), 0.3,
                             Vec3(0, -1, 2), Vec3(0, 1, 2), 0.4)
        self.assertAlmostEqual(d, 2.0 - 0.7, places=9)

    def test_overlap_and_clear(self):
        self.assertTrue(capsules_overlap(Vec3(-1, 0, 0), Vec3(1, 0, 0), 0.3,
                                         Vec3(0, 0, 0.5), Vec3(0, 0, 1.0), 0.3))
        self.assertFalse(capsules_overlap(Vec3(-1, 0, 0), Vec3(1, 0, 0), 0.2,
                                          Vec3(0, -1, 2), Vec3(0, 1, 2), 0.2))

    def test_sphere_overlap(self):
        self.assertTrue(sphere_overlap(Vec3(0, 0, 0), 0.5, Vec3(0.8, 0, 0), 0.5))
        self.assertFalse(sphere_overlap(Vec3(0, 0, 0), 0.5, Vec3(1.2, 0, 0), 0.5))


class TestProjection(unittest.TestCase):
    def test_isometric_separates_depth(self):
        front, back = isometric(Vec3(0, 1, 0)), isometric(Vec3(0, 1, 1))
        self.assertNotEqual(front, back)

    def test_orthographic_drops_depth(self):
        self.assertEqual(orthographic(Vec3(2, 1, 5)), (2, 1))

    def test_sidestep_needs_a_third_axis(self):
        self.assertFalse(sidestep_clears(0.0, 0.0))
        self.assertTrue(sidestep_clears(0.0, 0.4))


if __name__ == "__main__":
    unittest.main()
