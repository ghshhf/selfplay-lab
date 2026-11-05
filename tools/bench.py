"""Where do the microseconds go?

The self-play loop is three hot paths: stepping the arena, turning state into
a 29-dim observation, and the policy forward pass.  This bench times each of
them on the current machine, so a change can be judged as "+X frames/s" rather
than "feels faster".

    python tools/bench.py --steps 30000
    python tools/bench.py --steps 30000 --json runs/bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUEL_DIR = ROOT / "domains" / "duel"
for p in (str(ROOT), str(DUEL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import articulated  # noqa: E402
import duel  # noqa: E402
import freeform  # noqa: E402

from selfplay.nn.ppo import PPOAgent, PPOConfig  # noqa: E402


def bench_env(cls, n, with_obs: bool = True):
    env = cls(seed=0, max_frames=10 ** 6)
    env.reset(0)
    acts = [int(i % 8) for i in range(n)]
    step = env.step if with_obs else env.step_raw
    t0 = time.perf_counter()
    for i in range(n):
        step(acts[i], acts[(i + 3) % n])
        if env.frames > 1500:
            env.reset(i)
    return (time.perf_counter() - t0) / n * 1e6


def bench_pair(cls, n):
    """step() vs step_raw(): the difference is the observation's true cost."""
    full = bench_env(cls, n, with_obs=True)
    raw = bench_env(cls, n, with_obs=False)
    return {"step_us": round(full, 2), "step_raw_us": round(raw, 2),
            "obs_us": round(full - raw, 2),
            "obs_share": round((full - raw) / full, 3)}


def bench_policy(n, obs_dim=29, actions=8, hidden=64):
    agent = PPOAgent(obs_dim, actions, hidden=hidden, cfg=PPOConfig(), seed=0)
    rng = np.random.default_rng(0)
    x = np.random.randn(obs_dim).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(n):
        agent.act(x, rng, deterministic=True)
    us = (time.perf_counter() - t0) / n * 1e6
    return round(us, 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args(argv)

    n = args.steps
    out = {
        "machine_frames_per_core": {},
        "env": {
            "duel_1d": bench_pair(duel.Duel, n),
            "articulated": bench_pair(articulated.ArticulatedDuel, n),
            "freeform": bench_pair(freeform.FreeformDuel, n),
        },
        "policy_forward_us": bench_policy(n * 2),
        "policy_forward_us_1d": bench_policy(n, obs_dim=15, actions=6),
    }
    for name, e in out["env"].items():
        frames = 1e6 / (e["step_us"] + out["policy_forward_us"])
        out["machine_frames_per_core"][name] = round(frames)

    print("环境步进（含观测）")
    print("  %-12s %8s %8s %7s" % ("env", "step us", "obs us", "obs 占比"))
    for name, e in out["env"].items():
        print("  %-12s %8.1f %8.1f %6.0f%%"
              % (name, e["step_us"], e["obs_us"], 100 * e["obs_share"]))
    print("策略前向  %.1f us (29->64->8)   %.1f us (15->64->6)"
          % (out["policy_forward_us"], out["policy_forward_us_1d"]))
    print("端到端估算（单核）")
    for name, fps in out["machine_frames_per_core"].items():
        print("  %-12s %7d 帧/s  ≈ %d iteration/h (6 局 × 500 帧)"
              % (name, fps, fps * 3600 // 3000))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
