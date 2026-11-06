"""Play the freeform duel yourself, or let a simulated human take the chair.

Two modes, same rules, same physics:

    # interactive: you are the left fighter
    python domains/duel/play.py --opponent neural --ckpt runs/neuro_free

    # headless: a reaction-limited "human" policy plays N rounds against the
    # trained agent and prints a punch-by-punch log
    python domains/duel/play.py --auto --rounds 5 \
        --opponent neural --ckpt runs/neuro_free

Keys (interactive)
------------------
    d / fwd      walk forward          j    punch
    a / back     walk backward         k    kick (auto: high / low / flying)
    s            duck                  l    block (12 frames)
    space        jump                  e    substitution (40 chakra, 72f cd)
    q            quit

The point of ``--auto`` is not to fake a scoreboard: the human policy only
sees the same 29-dim observation the network gets, re-plans every
``--reaction`` frames (default 12 = ~200 ms), and never reads the engine's
internal move type.  Whatever it exploits, a person could see on screen too.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

DUEL_DIR = Path(__file__).resolve().parent
ROOT = DUEL_DIR.parents[1]
if str(DUEL_DIR) not in sys.path:
    sys.path.insert(0, str(DUEL_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import freeform as F  # noqa: E402
import limbs as L  # noqa: E402
from duel import FWD, BACK, PUNCH, KICK, BLOCK, IDLE, S_ATTACK, S_HURT, \
    S_KO, MAX_HP  # noqa: E402

from selfplay.nn.ppo import PPOAgent, PPOConfig  # noqa: E402

JUMP, DUCK = F.JUMP, F.DUCK
ARENA_HALF = 4.0

MOVE_LABEL = {
    PUNCH: "直拳",
    KICK: "前踢",
    L.HIGH_KICK: "头踢",
    L.LOW_KICK: "低扫",
    L.FLYING: "飞踢",
}
PART_LABEL = {"head": "头", "torso": "躯干", "leg": "腿", "guard": "格挡"}


# ---------------------------------------------------------------- opponents
class NeuralOpponent:
    def __init__(self, ckpt: Path, seed: int = 0, deterministic: bool = True,
                 obs_dim: int = F.OBS_DIM, n_actions: int = F.N_ACTIONS):
        state = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        arrays = np.load(ckpt / "state.npz")
        meta = np.asarray(arrays["agent__meta"]).tolist()
        obs_dim_ckpt, act_ckpt, hidden = int(meta[0]), int(meta[1]), int(meta[2])
        if (obs_dim_ckpt, act_ckpt) != (obs_dim, n_actions):
            raise SystemExit(
                "checkpoint expects %d obs / %d actions, this build is %d / %d; "
                "retrain it with: python domains/neuro/train.py --env freeform "
                "%s--out-dir %s" % (obs_dim_ckpt, act_ckpt, obs_dim,
                                    n_actions,
                                    "--emotion " if obs_dim > F.OBS_DIM else "",
                                    ckpt))
        self.agent = PPOAgent(obs_dim_ckpt, act_ckpt, hidden=hidden,
                              cfg=PPOConfig(), seed=seed)
        self.agent.load_state_dict({k: arrays[k] for k in arrays.files}, "agent")
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic
        self.label = "neural(%s, %d iter)" % (state.get("run_id", "?"),
                                              state.get("next_iter", 0))

    def reset(self, seed):
        pass

    def act(self, me, opp, obs):
        a, _lp, _v = self.agent.act(obs, self.rng,
                                    deterministic=self.deterministic)
        return a


class RandomOpponent:
    label = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, me, opp, obs):
        return int(self.rng.integers(F.N_ACTIONS))


class ScriptedOpponent:
    """The rule-based Bot, upgraded to reach into the freeform action set."""

    def __init__(self, gene: dict, seed=0):
        self.gene = dict(gene)
        self.label = "scripted"
        self.rng = np.random.default_rng(seed)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, me, opp, obs):
        g = self.gene
        dist = abs(opp.x - me.x)
        if me.state in (S_ATTACK, S_HURT, S_KO):
            return IDLE
        u = self.rng.random()
        # Duck / jump occasionally, mostly as a probe for the new verbs.
        if u < g.get("duck", 0.0) and me.y <= 0.0 and me.state != DUCK:
            return DUCK
        if u < g.get("duck", 0.0) + g.get("jump", 0.0) and me.y <= 0.0:
            return JUMP
        if self.rng.random() < g.get("guard", 0.2) and dist < 1.2:
            return BLOCK
        if dist > g.get("dist", 1.15):
            return FWD
        if me.cooldown <= 0:
            return KICK if self.rng.random() < g.get("kick", 0.5) else PUNCH
        return IDLE


# ------------------------------------------------------------------- player
class HumanPlayer:
    """What the *player* side gets: delayed observations and slow replanning.

    ``reaction`` frames of observation lag plus a decision refresh at the same
    cadence.  The policy never looks at ``me.move`` / ``opp.move``; it infers
    an incoming kick from foot height, which is exactly what a person sees.
    """

    def __init__(self, style: str = "balanced", reaction: int = 12, seed: int = 0):
        self.style = style
        self.reaction = max(1, int(reaction))
        self.rng = np.random.default_rng(seed)
        # Style only moves the two distance thresholds; the reading of kicks,
        # ducks and cooldowns is identical.
        if style == "pressure":
            self.punch_range, self.stand_off = 0.95, 1.08
        elif style == "counter":
            self.punch_range, self.stand_off = 0.85, 1.22
        else:
            self.punch_range, self.stand_off = 0.92, 1.18
        self.reset(0)

    def reset(self, seed):
        self.queue = deque(maxlen=self.reaction + 1)
        self.plan = IDLE
        self.plan_ttl = 0

    # --------------------------------------------------------------- decision
    def _decide(self, obs) -> int:
        dx = float(obs[12]) * ARENA_HALF
        # People do not aim at a stale snapshot: they track motion.  Project the
        # delayed separation forward by the reaction window using relative
        # velocity (obs 1 / 7 are vx normalised by 4 m/s).
        rel_v = (float(obs[7]) - float(obs[1])) * 4.0
        dx += rel_v * (self.reaction / 60.0)
        dist = abs(dx)
        me_state = int(round(float(obs[3]) * 6))
        opp_state = int(round(float(obs[9]) * 6))
        opp_foot_y = float(obs[22]) * 2.0     # opponent's front foot height
        opp_duck = float(obs[27]) > 0.5
        my_y = float(obs[23]) * 2.0
        cooldown = float(obs[14]) * 20.0

        if me_state in (S_ATTACK, S_HURT, S_KO):
            return IDLE
        if my_y > 0.05:                        # airborne: only kick makes sense
            return KICK if dist < 1.2 else FWD

        # Blocking is *not* a reaction here: a 7-frame kick windup is 117 ms,
        # far below the human's 200 ms lag.  By the time the delayed frame
        # shows an attack, it has either landed or is in recovery, so holding
        # block only gives up tempo.  Instead the policy spends its reaction
        # budget on anticipation (duck / backstep) and punishes with the kick.
        if opp_state == S_ATTACK and dist < 1.45:
            if opp_foot_y > 1.25 and me_state != DUCK:
                return DUCK                     # head / flying kick incoming
            if opp_foot_y < 0.75 and dist < 1.05 and me_y_ok(obs):
                return JUMP                     # low sweep incoming
            # Anything else is a jab or a front kick: too fast to answer, so
            # keep the spacing plan instead of freezing into a guard.

        # A ducked opponent cannot be jabbed or head-kicked: use the front kick
        # at hip height instead of the short-range options.
        if opp_duck and dist < 1.3 and cooldown <= 0:
            return KICK
        if dist <= self.punch_range and cooldown <= 0:
            return PUNCH
        if dist <= 1.24 and cooldown <= 0:
            return KICK
        # Hysteresis: without it a 12-frame decision walks half a metre past
        # the target distance and lands in the pocket, where head kicks hurt.
        if dist > self.stand_off + 0.08:
            return FWD
        if dist < self.stand_off - 0.08:
            return BACK
        return IDLE

    # ------------------------------------------------------------------- act
    def act(self, obs) -> int:
        self.queue.append(np.asarray(obs, dtype=np.float32))
        if len(self.queue) <= self.reaction:
            return IDLE                     # still reacting to the reset
        if self.plan_ttl <= 0:
            self.plan = self._decide(self.queue[0])
            self.plan_ttl = self.reaction
        self.plan_ttl -= 1
        return self.plan


def me_y_ok(obs) -> bool:
    return float(obs[23]) < 0.05


# ---------------------------------------------------------------- rendering
def render(env, last: str, agent_label: str) -> str:
    p0, p1 = env.p0, env.p1
    half = env.half
    width, height = 62, 12

    def bar(hp, n=20):
        fill = int(round(max(0.0, hp) / MAX_HP * n))
        return "█" * fill + "·" * (n - fill)

    def chakra(f, n=10):
        fill = int(round(max(0.0, getattr(f, "chakra", 0.0)) / 100.0 * n))
        return "◆" * fill + "◇" * (n - fill)

    def guard_bar(f, n=10):
        fill = int(round(max(0.0, getattr(f, "guard", 100.0)) / 100.0 * n))
        return "▰" * fill + "▱" * (n - fill)

    def mood(f):
        if getattr(f, "affect", None) is None:
            return ""
        a = f.affect
        return "冷静%.2f 恐惧%.2f 怒%.2f 势%.2f" % (
            a.composure, a.fear, a.anger, a.momentum)

    grid = [[" "] * width for _ in range(height)]
    # ground line
    for x in range(width):
        grid[height - 1][x] = "─"
    spr = {"PLAYER": "O", "AGENT": "#"}
    for f, tag in ((p0, "PLAYER"), (p1, "AGENT")):
        # Depth is a screen-space shift, the cheapest readable projection:
        # a sidestepped fighter visibly leaves the line without a 3D engine.
        gx = int(round((f.x + f.z * 0.5 + half) / (2 * half) * (width - 1)))
        gy = int(round((1.78 - f.y) / 1.78 * (height - 2)))
        char = "o" if f.state == DUCK else spr[tag]
        if f.state == S_ATTACK:
            char = "*" if tag == "PLAYER" else "+"
        if f.state == S_KO:
            char = "x"
        grid[max(0, min(height - 2, gy))][max(0, min(width - 1, gx))] = char

    out = []
    out.append("  你 PLAYER  HP %5.1f [%s] 查克拉 %s 防 %s"
               % (p0.hp, bar(p0.hp), chakra(p0), guard_bar(p0)))
    out.append("  VS %-14s HP %5.1f [%s] 查克拉 %s 防 %s"
               % (agent_label[:14], p1.hp, bar(p1.hp), chakra(p1),
                  guard_bar(p1)))
    if env.emotion:
        out.append("  心态  %s" % mood(p0))
        out.append("  对手  %s" % mood(p1))
    out.append("  t=%5.2fs  x=%+.2f/%+.2f  y=%.2f/%.2f  %s"
               % (env.frames / 60.0, p0.x, p1.x, p0.y, p1.y, last))
    if getattr(env, "arena3d", False):
        out.append("  z=%+.2f/%+.2f  %s"
                   % (p0.z, p1.z,
                      "错开攻击线" if env._off_line(p0, p1) else "同线"))
    out.append("  ┌" + " " * width + "┐")
    for row in grid:
        out.append("  │" + "".join(row) + "│")
    out.append("  └" + " " * width + "┘")
    if getattr(env, "arena3d", False):
        out.append("  d/fwd  a/back  j 直拳  k 踢  s 下蹲  space 跳  l 格挡  "
                   "e 替身  w 侧移  q 退出")
    else:
        out.append("  d/fwd  a/back  j 直拳  k 踢  s 下蹲  space 跳  l 格挡  "
                   "e 替身  q 退出")
    return "\n".join(out)


KEYMAP = {
    "d": FWD, "j": PUNCH, "k": KICK, "s": DUCK, "l": BLOCK,
    " ": JUMP, "a": BACK, "f": FWD, "e": F.SUB, "w": F.SIDESTEP,
}


def interactive(args, arena, opp):
    import curses

    def loop(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        env = arena
        obs = arena.reset(int(args.seed))
        o0, o1 = obs
        hold, last = (IDLE, 0), "开打！"
        try:
            while True:
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    return
                if ch != -1:
                    key = chr(ch).lower()
                    if key in KEYMAP:
                        act = KEYMAP[key]
                        hold = (act, 14 if act in (FWD, BACK) else 2)
                        last = "输入 %s" % F.ACTION_NAMES[act]
                act, ttl = hold
                if ttl <= 0:
                    act = IDLE
                else:
                    hold = (act, ttl - 1)
                a1 = opp.act(env.p1, env.p0, o1)
                o0, o1, r0, r1, done, info = arena.step(act, a1)
                stdscr.clear()
                for i, line in enumerate(render(env, last, opp.label)):
                    stdscr.addstr(i, 0, line[:120])
                stdscr.refresh()
                if done:
                    stdscr.addstr(13, 0, "本局结束，按任意键重开，q 退出")
                    stdscr.nodelay(False)
                    if stdscr.getch() in (ord("q"), ord("Q")):
                        return
                    stdscr.nodelay(True)
                    obs = arena.reset(int(time.time()))
                    o0, o1 = obs
                    hold, last = (IDLE, 0), "重开"
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            pass

    curses.wrapper(loop)


# -------------------------------------------------------------------- report
def round_report(idx, env, events, winner, human_side=0):
    p0, p1 = env.p0, env.p1
    if winner == human_side:
        res = "你 胜"
    elif winner == -1:
        res = "同归于尽/超时判负" if p0.hp == p1.hp else "按血量判"
    else:
        res = "算法 胜"
    lines = ["第 %d 局: %s | 你 %5.1f  vs  算法 %5.1f | %.1f 秒"
             % (idx, res, max(0.0, p0.hp), max(0.0, p1.hp), env.frames / 60.0)]
    took = [e for e in events if e[4] != human_side]
    by_move = {}
    for (frame, move, dmg, blocked, side, dist, part) in events:
        key = ("你" if side == human_side else "算法",
               MOVE_LABEL.get(move, "?"), PART_LABEL.get(part, part), blocked)
        n, d = by_move.get(key, (0, 0.0))
        by_move[key] = (n + 1, d + dmg)
    for (who, move, part, blocked), (n, d) in sorted(by_move.items()):
        lines.append("    %s %-3s ×%-2d  %s球 共 %5.1f 伤害%s"
                     % (who, move, n, part, d, "（被格挡）" if blocked else ""))
    if any(e[4] != human_side and e[2] >= 8 for e in took):
        lines.append("    ← 算法打穿了几次重击，注意它的距离")
    return "\n".join(lines)


def auto(args, arena, opp):
    env = arena
    human = HumanPlayer(style=args.style, reaction=args.reaction,
                        seed=int(args.seed))
    summary = []
    for r in range(args.rounds):
        human.reset(args.seed + r)
        obs = arena.reset(int(args.seed) + r * 7919)
        o0, o1 = obs
        while True:
            a0 = human.act(o0)
            a1 = opp.act(env.p1, env.p0, o1)
            o0, o1, r0, r1, done, info = arena.step(a0, a1)
            if done:
                break
        winner = -1
        if env.p0.alive and not env.p1.alive:
            winner = 0
        elif env.p1.alive and not env.p0.alive:
            winner = 1
        elif env.p0.hp > env.p1.hp:
            winner = 0
        elif env.p1.hp > env.p0.hp:
            winner = 1
        summary.append({"round": r + 1, "winner": winner,
                        "hp": [round(env.p0.hp, 1), round(env.p1.hp, 1)],
                        "frames": env.frames,
                        "dealt": [round(env.damage_dealt[0], 1),
                                  round(env.damage_dealt[1], 1)],
                        "parts": dict(env.part_hits),
                        "jumps": getattr(env, "jumps", 0),
                        "low_hits": getattr(env, "low_hits", 0),
                        "flying_hits": getattr(env, "flying_hits", 0)})
        if not args.quiet:
            print(round_report(r + 1, env, env.events, winner))
            print()
    wins = sum(1 for s in summary if s["winner"] == 0)
    print("=" * 58)
    print("总战绩: %d 胜 / %d 局   (对手: %s, 反应 %d 帧 = %.2f s)"
          % (wins, len(summary), opp.label, args.reaction, args.reaction / 60.0))
    for s in summary:
        print("  第%d局  %s  剩余 %4.1f:%4.1f  造成 %4.1f  承受 %4.1f"
              % (s["round"], "胜" if s["winner"] == 0 else ("负" if s["winner"] == 1 else "平"),
                 s["hp"][0], s["hp"][1], s["dealt"][0], s["dealt"][1]))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False,
                                              indent=1), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Play the freeform duel")
    ap.add_argument("--opponent", choices=("neural", "scripted", "random"),
                    default="neural")
    ap.add_argument("--ckpt", type=Path, default=Path("runs/neuro_free"),
                    help="checkpoint dir for --opponent neural")
    ap.add_argument("--auto", action="store_true", help="headless: simulated human")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reaction", type=int, default=12, help="frames of lag")
    ap.add_argument("--style", choices=("balanced", "pressure", "counter"),
                    default="balanced")
    ap.add_argument("--max-frames", type=int, default=1800)
    ap.add_argument("--emotion", action="store_true",
                    help="enable the affect layer (checkpoint must be trained "
                         "with --emotion)")
    ap.add_argument("--physics", action="store_true",
                    help="enable mass/friction/wall-bounce/hitstop physics")
    ap.add_argument("--arena3d", action="store_true",
                    help="add the depth axis, SIDESTEP and z-aware hits")
    ap.add_argument("--mass0", type=float, default=70.0,
                    help="seat 0 weight class in kg (physics only)")
    ap.add_argument("--mass1", type=float, default=70.0,
                    help="seat 1 weight class in kg (physics only)")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    env_cfg = ({"mass0": args.mass0, "mass1": args.mass1}
               if args.physics and (args.mass0 != 70.0 or args.mass1 != 70.0)
               else None)
    arena = F.FreeformDuel(max_frames=args.max_frames, emotion=args.emotion,
                           physics=args.physics, env_cfg=env_cfg,
                           arena3d=args.arena3d)
    if args.opponent == "neural":
        ckpt = args.ckpt if args.ckpt.exists() else None
        if ckpt is None:
            raise SystemExit("no checkpoint at %s; train one with "
                             "domains/neuro/train.py --env freeform "
                             "--out-dir %s" % (args.ckpt, args.ckpt))
        opp = NeuralOpponent(ckpt, seed=args.seed, deterministic=True,
                             obs_dim=(F.OBS_DIM_3D if args.arena3d
                                      else F.OBS_DIM)
                             + (F.EMOTION_EXTRA if args.emotion else 0),
                             n_actions=(F.N_ACTIONS_3D if args.arena3d
                                        else F.N_ACTIONS))
    elif args.opponent == "scripted":
        opp = ScriptedOpponent({"aggr": 0.6, "guard": 0.3, "kick": 0.5,
                                "dist": 1.15, "duck": 0.12, "jump": 0.08},
                               seed=args.seed)
    else:
        opp = RandomOpponent(seed=args.seed)

    if args.auto:
        auto(args, arena, opp)
    else:
        interactive(args, arena, opp)


if __name__ == "__main__":
    main()
