"""Minimal rigid-body physics for the 1-D duel: mass, friction, walls, hitstop.

Why a separate module
---------------------
The duel's original "physics" was three lines: ``x += vx / FPS`` plus an
exponential decay and a position clamp.  That is a kinematic illusion, not a
model: a 70 kg fighter and a 40 kg fighter are pushed identically, a wall
deletes all momentum, and nothing carries mass.  This module replaces it with
the smallest set of rules that behaves like matter:

* **mass** -- ``Fighter.mass`` (kg); impulses change velocity by ``j / m``.
* **Coulomb friction** -- constant deceleration ``mu * g`` on the ground, so a
  knockback slides a measurable, mass-independent distance and then stops.
* **gravity** -- the same 9.8 m/s^2 the jump already used, now shared with any
  future airborne body.
* **restitution** -- arena walls bounce a fraction of the incoming speed;
  interior walls still stop the body (they are obstacles, not cushions).
* **impulse exchange** -- a hit pushes the defender by its damage and recoils
  the attacker, so trading heavy blows moves *both* bodies.
* **hitstop** -- a few frozen frames on impact; the standard trick that makes
  contact read as weight rather than as a number changing.

Everything is a pure function over an object with ``x, y, vx, vy, mass``, so
it is testable without an environment and usable by the 3D layer later.
Units: metres, seconds, kilograms.  One frame is ``1 / FPS``.
"""

from __future__ import annotations

import math

FPS = 60
DT = 1.0 / FPS

GRAVITY = 9.8
KINETIC_MU = 0.30          # ground friction coefficient -> 2.94 m/s^2
AIR_DRAG = 0.40            # linear horizontal drag per second while airborne
STOP_SPEED = 0.05          # below this a sliding body is considered still
WALL_RESTITUTION = 0.35
WALL_BOUNCE_MIN = 1.0      # slower contacts (walking into a wall) just stop

IMPULSE_PER_DAMAGE = 11.0  # kg*m/s of push per point of damage
BLOCK_IMPULSE = 0.45       # blocking converts most of the hit into chip push
ATTACKER_RECOIL = 0.22     # fraction of the impulse that pushes the attacker back

DEFAULT_MASS = 70.0


def mass_of(body) -> float:
    return float(getattr(body, "mass", DEFAULT_MASS))


def mass_speed_factor(mass: float, reference: float = DEFAULT_MASS) -> float:
    """Walking-speed multiplier for a weight class.

    Without a downside, extra mass would be a free upgrade: it already makes
    you harder to push, harder to launch into a wall and harder to separate.
    The cost is mobility -- speed scales with ``sqrt(reference / mass)``.
    At the default 70 kg this is exactly 1.0, so equal-weight matches are
    unchanged; 100 kg walks at 0.84x and 50 kg at 1.18x.
    """
    return math.sqrt(reference / max(1.0, float(mass)))


def velocity(body) -> float:
    return float(getattr(body, "vx", 0.0))


def momentum(body) -> float:
    return mass_of(body) * velocity(body)


def step(body, dt: float = DT) -> bool:
    """Advance one frame.  Returns True if the body just touched the ground.

    Grounded bodies get Coulomb friction; airborne bodies get gravity and
    linear drag.  Landing is inelastic (``vy = 0``): a bounce on every jump
    would add a second, unintended movement system.
    """
    airborne = body.y > 0.0 or body.vy > 0.0
    if airborne:
        body.vy -= GRAVITY * dt
        body.vx -= body.vx * AIR_DRAG * dt
        body.x += body.vx * dt
        body.y += body.vy * dt
        if body.y <= 0.0:
            body.y = 0.0
            body.vy = 0.0
            return True
        return False

    decel = KINETIC_MU * GRAVITY * dt
    if abs(body.vx) <= max(decel, STOP_SPEED):
        body.vx = 0.0
    else:
        body.vx -= math.copysign(decel, body.vx)
    body.x += body.vx * dt
    return False


def collide_walls(body, half: float, walls=(), bounce: bool = True) -> bool:
    """Clamp to arena/interior walls.  Returns True on an arena bounce.

    ``bounce`` is decided by the caller (Freeform only lets knockback bounce):
    a fighter holding forward against a wall should stop, not vibrate.
    Interior walls are obstacles: the body stops dead, because sliding along
    one would let a fighter tunnel through it next frame.
    """
    bounced = False
    if body.x <= -half:
        body.x = -half
        if body.vx < 0.0:
            bounced = _bounce(body, bounce)
    elif body.x >= half:
        body.x = half
        if body.vx > 0.0:
            bounced = _bounce(body, bounce)
    for (wx, ww) in walls:
        lo, hi = wx - ww, wx + ww
        if lo < body.x < hi:
            body.x = lo if (body.x - lo) < (hi - body.x) else hi
            body.vx = 0.0
    return bounced


def _bounce(body, allow: bool) -> bool:
    if allow and abs(body.vx) >= WALL_BOUNCE_MIN:
        body.vx = -body.vx * WALL_RESTITUTION
        return True
    body.vx = 0.0
    return False


def separate_pair(a, b, gap: float, restitution: float = 0.0) -> float:
    """Push two overlapping bodies apart, weighted by inverse mass.

    Also resolves the closing velocity, so a body walking into a heavier one
    loses its speed instead of teleporting through the seam.  Returns the
    overlap that was corrected (0.0 when already separated).
    """
    d = b.x - a.x
    if abs(d) >= gap:
        return 0.0
    n = 1.0 if d >= 0.0 else -1.0
    overlap = gap - abs(d)
    inv_a = 1.0 / mass_of(a)
    inv_b = 1.0 / mass_of(b)
    total = inv_a + inv_b
    a.x -= n * overlap * (inv_a / total)
    b.x += n * overlap * (inv_b / total)

    v_rel = (b.vx - a.vx) * n
    if v_rel < 0.0:  # closing
        j = -(1.0 + restitution) * v_rel / total
        a.vx -= j * inv_a * n
        b.vx += j * inv_b * n
    return overlap


def strike(attacker, defender, damage: float, blocked: bool = False,
           mul: float = 1.0) -> float:
    """Exchange an impulse along the attacker's facing direction.

    Returns the impulse magnitude.  Momentum is *not* conserved: muscles add
    energy on purpose, otherwise a hit could not move a heavier opponent.
    """
    j = IMPULSE_PER_DAMAGE * damage * mul
    if blocked:
        j *= BLOCK_IMPULSE
    direction = 1.0 if defender.x >= attacker.x else -1.0
    defender.vx += direction * j / mass_of(defender)
    attacker.vx -= direction * ATTACKER_RECOIL * j / mass_of(attacker)
    return j


def hitstop_frames(damage: float, head: bool = False, blocked: bool = False) -> int:
    """Frozen frames after impact.  Heavier / cleaner hits freeze longer."""
    if blocked:
        return 1
    n = 2
    if head:
        n += 1
    if damage >= 8.0:
        n += 1
    if damage >= 14.0:
        n += 2
    return min(n, 6)
