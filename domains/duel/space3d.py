"""3D geometry primitives for the next stage of the duel.

The current arena is a sagittal plane: x (forward) and y (height).  The
defining mechanic of a real 3D fighter (Tekken, Soulcalibur, Virtua Fighter)
is the third axis: a sidestep moves the fighter *out of the attack line*, so
the same punch whiffs even though the distance never changed.  You cannot bolt
that on with a boolean "dodged" flag; hit detection has to become volumetric.

This module is the foundation for that, kept free of environment policy:

* ``Vec3`` -- positions and velocities;
* ``point_segment_distance`` / ``segment_segment_distance`` -- the distance
  core of limb-vs-limb and limb-vs-body collision;
* ``capsule_distance`` -- two limb capsules (segment + radius), which is what
  ``limbs.capsule_overlap`` already does in 2D;
* ``orthographic`` / ``isometric`` -- project 3D state back onto the existing
  2D renderer so the debug viewer keeps working while the simulation gains
  depth.

Everything is NumPy with explicit shapes, so it can be unit-tested without an
engine.  The 2D code keeps using its own fast path; nothing here is on the hot
path yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return float(np.sqrt(self.dot(self)))

    def normalized(self) -> "Vec3":
        n = self.length()
        return Vec3(0.0, 0.0, 0.0) if n < 1e-12 else self * (1.0 / n)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


def point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> float:
    """Shortest distance between point ``p`` and segment ``a-b``."""
    ab = b - a
    denom = ab.dot(ab)
    if denom < 1e-12:
        return (p - a).length()
    t = max(0.0, min(1.0, (p - a).dot(ab) / denom))
    closest = a + ab * t
    return (p - closest).length()


def segment_segment_distance(a0: Vec3, a1: Vec3,
                             b0: Vec3, b1: Vec3) -> float:
    """Shortest distance between two 3D segments (Ericson, RTCD 5.1.9)."""
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    aa = d1.dot(d1)
    e = d2.dot(d2)
    f = d2.dot(r)

    if aa <= 1e-12 and e <= 1e-12:
        return r.length()
    if aa <= 1e-12:
        s, t = 0.0, max(0.0, min(1.0, f / e))
    else:
        c = d1.dot(r)
        if e <= 1e-12:
            t = 0.0
            s = max(0.0, min(1.0, -c / aa))
        else:
            b = d1.dot(d2)
            denom = aa * e - b * b
            if abs(denom) > 1e-12:
                s = max(0.0, min(1.0, (b * f - c * e) / denom))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / aa))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / aa))
    return ((a0 + d1 * s) - (b0 + d2 * t)).length()


def capsule_distance(a0: Vec3, a1: Vec3, ra: float,
                     b0: Vec3, b1: Vec3, rb: float) -> float:
    """Surface-to-surface distance between two capsules (negative = overlap)."""
    return segment_segment_distance(a0, a1, b0, b1) - (ra + rb)


def capsules_overlap(a0: Vec3, a1: Vec3, ra: float,
                     b0: Vec3, b1: Vec3, rb: float) -> bool:
    return capsule_distance(a0, a1, ra, b0, b1, rb) <= 0.0


def sphere_overlap(c0: Vec3, r0: float, c1: Vec3, r1: float) -> bool:
    return (c0 - c1).length() <= r0 + r1


def orthographic(v: Vec3) -> tuple[float, float]:
    """Front view: drop depth, keep x and height."""
    return v.x, v.y


def isometric(v: Vec3, depth_scale: float = 0.5) -> tuple[float, float]:
    """Cheap pseudo-3D projection for the debug viewer.

    Depth moves the point up-right, so two fighters at the same x but different
    z stop drawing on top of each other.
    """
    return v.x + v.z * depth_scale, v.y + v.z * depth_scale


def sidestep_clears(attacker_z: float, defender_z: float,
                    body_width: float = 0.32) -> bool:
    """Whether a lateral offset is enough to leave a straight attack line.

    In 2D this is always ``False``; it is the single predicate that has to
    change when the arena becomes 3D.
    """
    return abs(float(defender_z) - float(attacker_z)) > body_width
