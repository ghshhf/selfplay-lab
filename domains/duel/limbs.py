"""2D sagittal-plane limb geometry: forward kinematics + capsule collision.

Coordinate system: ``x`` is horizontal (towards the opponent is positive),
``y`` is vertical with the floor at ``y = 0``.  Only the *geometry* lives in 2D;
locomotion still integrates on the x axis exactly like ``duel.py``.  This is a
deliberate scope cut: make limb contact correct first, then decide whether the
whole body needs 2D platforming physics.

Poses are deterministic functions of the fighter state/move/timer, so no extra
state has to be checkpointed -- a replay can be rebuilt from ``duel`` state alone.
"""

from __future__ import annotations

import math

# ---- body proportions (metres, fighting stance) ----
HIP_Y = 0.96
TORSO_TOP_Y = 1.40
HEAD_Y = 1.66
HEAD_R = 0.12
TORSO_R = 0.19
SHOULDER_Y = 1.44
UPPER_ARM = 0.38
FOREARM = 0.40
HAND_R = 0.08
THIGH = 0.50
SHIN = 0.50
LEG_R = 0.085

FIST_R = 0.09
FOOT_R = 0.10
GUARD_R = 0.10

# state ids mirror duel.py
S_IDLE, S_WALK, S_ATTACK, S_BLOCK, S_HURT, S_KO = range(6)
S_DUCK, S_JUMP = 6, 7
PUNCH, KICK = 2, 3
HIGH_KICK = 6  # internal technique, selected automatically inside kicking range
LOW_KICK = 7   # internal: low roundhouse from a crouch, hits the legs
FLYING = 8     # internal: airborne kick at head height
# (windup, active, recover, nominal reach, damage) -- frames at 60 Hz.
# Jab is the fastest option; front kick out-ranges it; the head kick is slower
# but lands 1.35x on the head and only exists in the pocket (< 0.84 m).
ART_MOVES = {
    PUNCH: (4, 3, 7, 1.04, 5.5),       # jab / oi-zuki
    KICK: (7, 4, 11, 1.29, 8.5),       # front kick / mae geri
    HIGH_KICK: (12, 4, 18, 0.83, 12.5),  # head kick / mawashi geri
    LOW_KICK: (6, 3, 12, 0.95, 5.0),   # low roundhouse / gedan mawashi
    FLYING: (9, 4, 16, 1.10, 9.0),     # flying kick / hien geri
}


def extension(state, phase, timer, spec) -> float:
    """0..1 extension of the attacking limb over windup / active / recover."""
    windup, active, recover = spec[0], spec[1], spec[2]
    if state != S_ATTACK:
        return 0.0
    if phase == "windup":
        return max(0.0, min(1.0, 1.0 - timer / max(1.0, float(windup))))
    if phase == "active":
        return 1.0
    return max(0.0, min(1.0, (timer - 1.0) / max(1.0, float(recover))))


def joint_points(facing, root_x, shoulder_deg, elbow_deg, hip_deg, knee_deg,
                 root_y=0.0, crouch=0.0):
    """Forward kinematics in the sagittal plane.

    Angles are degrees from straight-down, positive towards the facing
    direction.  ``elbow_deg`` / ``knee_deg`` are relative joint bends.
    ``root_y`` lifts the whole body (jump); ``crouch`` (0..1) lowers hips,
    torso and head, which is what makes high attacks whiff over a duck.
    """
    f = 1.0 if facing >= 0 else -1.0
    sx, sy = root_x, SHOULDER_Y + root_y - 0.38 * crouch
    a1 = math.radians(shoulder_deg)
    a2 = math.radians(shoulder_deg + elbow_deg)
    ex = sx + UPPER_ARM * math.sin(a1) * f
    ey = sy - UPPER_ARM * math.cos(a1)
    hx = ex + FOREARM * math.sin(a2) * f
    hy = ey - FOREARM * math.cos(a2)

    px, py = root_x, HIP_Y + root_y - 0.30 * crouch
    b1 = math.radians(hip_deg)
    b2 = math.radians(hip_deg + knee_deg)
    kx = px + THIGH * math.sin(b1) * f
    ky = py - THIGH * math.cos(b1)
    fx = kx + SHIN * math.sin(b2) * f
    fy = ky - SHIN * math.cos(b2)

    return {
        "shoulder": (sx, sy), "elbow": (ex, ey), "hand": (hx, hy),
        "hip": (px, py), "knee": (kx, ky), "foot": (fx, fy),
        "neck": (root_x, TORSO_TOP_Y + root_y - 0.38 * crouch),
        "head": (root_x, HEAD_Y + root_y - 0.47 * crouch),
    }


def pose(facing, root_x, state, move, phase, timer, spec, frame,
         root_y=0.0, crouch=None):
    """Map a fighter's discrete state to joint angles, then to points."""
    ext = extension(state, phase, timer, spec) if spec else 0.0
    if crouch is None:
        crouch = 1.0 if state == S_DUCK else 0.0

    if state == S_ATTACK and move == PUNCH:
        # High guard -> straight jab at chin height (4-frame windup).
        sh = 60.0 + 40.0 * ext
        el = 80.0 - 70.0 * ext
        hip, knee = 4.0, 6.0
    elif state == S_ATTACK and move == KICK:
        # Front kick: leg horizontal at hip height, out-ranges the jab.
        sh, el = 60.0, 80.0
        hip = 20.0 + 75.0 * ext
        knee = 60.0 - 60.0 * ext
    elif state == S_ATTACK and move == HIGH_KICK:
        # Head kick: leg climbs to ~1.70 m, shortening horizontal reach.
        sh, el = 55.0, 70.0
        hip = 20.0 + 110.0 * ext
        knee = 60.0 - 40.0 * ext
    elif state == S_ATTACK and move == LOW_KICK:
        # Low roundhouse from a half-crouch: foot sweeps at knee height (0.46 m),
        # below every guard capsule, so blocking cannot stop it.
        sh, el = 55.0, 95.0
        hip = 10.0 + 50.0 * ext
        knee = 6.0 * (1.0 - ext)
        crouch = 0.35
    elif state == S_ATTACK and move == FLYING:
        # Flying kick: body is airborne; the leg extends up-forward, so the
        # foot lands near a standing opponent's chin.
        sh, el = 45.0, 60.0
        hip = 95.0 + 20.0 * ext
        knee = 10.0 * (1.0 - ext)
    elif state == S_DUCK:
        sh, el = 55.0, 105.0
        hip, knee = 2.0, 76.0
    elif state == S_JUMP:
        sh, el = 48.0, 70.0
        hip, knee = 30.0, 70.0
    elif state == S_BLOCK:
        sh, el = 55.0, 110.0
        hip, knee = 6.0, 8.0
    elif state == S_WALK:
        swing = 12.0 if (frame // 12) % 2 == 0 else -12.0
        sh = 14.0 - 0.6 * swing
        el = 18.0
        hip = swing
        knee = 8.0 + max(0.0, swing)
    elif state == S_HURT:
        sh, el = 32.0, 40.0
        hip, knee = -8.0, 25.0
    elif state == S_KO:
        sh, el = 40.0, 20.0
        hip, knee = 85.0, 5.0
    else:
        sh, el = 12.0, 14.0
        hip, knee = 4.0, 6.0

    return joint_points(facing, root_x, sh, el, hip, knee,
                        root_y=root_y, crouch=crouch)


def attack_capsule(ch):
    """Capsule of the attacking limb: forearm/fist for PUNCH, shin/foot for KICK."""
    if ch == PUNCH:
        return ("elbow", "hand", FIST_R)
    return ("knee", "foot", FOOT_R)


def attack_segments(pose_dict, move):
    a, b, r = attack_capsule(move)
    return pose_dict[a], pose_dict[b], r


def hurt_capsules(pose_dict):
    """Torso, head, and both legs.  Arms are not hurtboxes (too fiddly)."""
    return (
        (pose_dict["hip"], pose_dict["neck"], TORSO_R, "torso", 1.0),
        (pose_dict["head"], pose_dict["head"], HEAD_R, "head", 1.35),
        (pose_dict["hip"], pose_dict["knee"], LEG_R, "leg", 0.75),
        (pose_dict["knee"], pose_dict["foot"], LEG_R, "leg", 0.75),
    )


def guard_capsules(pose_dict):
    """Forearm guard when blocking (both arms are posed identically)."""
    return (
        (pose_dict["elbow"], pose_dict["hand"], GUARD_R),
    )


# --------------------------------------------------------------------- geometry
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_dist(p, a, b):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    wx, wy = p[0] - ax, p[1] - ay
    den = vx * vx + vy * vy
    if den < 1e-12:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / den))
    return math.hypot(p[0] - (ax + t * vx), p[1] - (ay + t * vy))


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segment_distance(a, b, c, d):
    """Minimum distance between two 2D segments (0 if they intersect)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)):
        return 0.0
    return min(point_segment_dist(a, c, d), point_segment_dist(b, c, d),
               point_segment_dist(c, a, b), point_segment_dist(d, a, b))


def capsule_overlap(seg0, seg1):
    a, b, r0 = seg0
    c, d, r1 = seg1
    return segment_distance(a, b, c, d) <= r0 + r1


def closest_hurt(pose_dict, seg):
    """Return (part, multiplier, clearance) for the nearest hurt capsule.

    ``clearance`` is ``distance - radius``: negative means the attack overlaps
    that hurtbox.  The deepest overlap wins, so a hit on the head beats a
    shallow graze on a leg.
    """
    a, b, r0 = seg
    best = None
    for (c, d, r1, part, mul) in hurt_capsules(pose_dict):
        clearance = segment_distance(a, b, c, d) - (r0 + r1)
        if best is None or clearance < best[2]:
            best = (part, mul, clearance)
    return best
