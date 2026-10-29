"""
collect_duel.py —— 采集两个小人对打的数据

单进程、单环境、串行跑（这台机器一次跑一个就够了）。
每帧记一行，回合结束后单独记结局。

用法：
    python collect_duel.py --episodes 800 --out duel.csv --seed 20261029
"""

import argparse
import csv
import json
import time

import numpy as np

import duel as D
from bot import Bot

# 每帧一行。字段全部自解释，不靠记编码表：
#   x/vx   位置(m)、速度(m/s)          hp   血量 0~100
#   st     状态 0闲置 1走 2出招 3格挡 4硬直 5倒地
#   tm     当前状态剩余帧     cd   出招冷却帧
#   a      本帧下发的动作 0前进 1后退 2拳 3踢 4格挡 5待机
#   dist   两人间距(m)
#   atk    本帧出手的一方(-1 无)   mv 招式 1拳 2踢 0无
#   vic    本帧被命中(或格挡)的一方(-1 无)   blk 1=被格挡
#   hdist  命中判定当时的间距(m)，-1=本帧无命中（区别于 dist：dist 是积分后的观测值）
COLS = ["ep", "frame", "x0", "x1", "vx0", "vx1", "hp0", "hp1",
        "st0", "st1", "tm0", "tm1", "cd0", "cd1",
        "a0", "a1", "dist", "atk", "mv", "vic", "blk", "hdist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--max-frames", type=int, default=1800)
    ap.add_argument("--out", default="duel.csv")
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    episodes = []
    t0 = time.time()

    for ep in range(args.episodes):
        env = D.Duel(seed=int(rng.integers(1 << 30)), max_frames=args.max_frames)
        env.reset()
        b0, b1 = Bot.random(rng), Bot.random(rng)

        n_hit = 0
        n_block = 0
        while True:
            a0, a1 = b0(env.p0, env.p1), b1(env.p1, env.p0)
            _, _, _, _, done, info = env.step(a0, a1)
            f = env.p0, env.p1
            ev = env.events[-1] if len(env.events) > n_hit else None
            atk, mv, vic, blk, hdist = -1, 0, -1, 0, -1.0
            if ev is not None:
                # (frame, move, damage, blocked, attacker)
                n_hit += 1
                atk = ev[4]
                mv = 1 if ev[1] == D.PUNCH else 2
                vic = 1 - atk
                blk = 1 if ev[3] else 0
                hdist = round(ev[5], 4)
                n_block += blk
            rows.append((
                ep, env.frames,
                round(f[0].x, 4), round(f[1].x, 4),
                round(f[0].vx, 3), round(f[1].vx, 3),
                round(f[0].hp, 2), round(f[1].hp, 2),
                f[0].state, f[1].state,
                f[0].timer, f[1].timer,
                f[0].cooldown, f[1].cooldown,
                a0, a1,
                round(abs(f[1].x - f[0].x), 4),
                atk, mv, vic, blk, hdist,
            ))
            if done:
                break

        episodes.append({
            "ep": ep,
            "frames": info["frames"],
            "winner": info["winner"],
            "hp": [round(info["hp"][0], 1), round(info["hp"][1], 1)],
            "hits": info["hits"],
            "blocked": n_block,
            "damage": [round(d, 3) for d in info["damage"]],
            "bot0": {k: round(v, 3) if isinstance(v, float) else v
                     for k, v in b0.__dict__.items() if k != "rng"},
            "bot1": {k: round(v, 3) if isinstance(v, float) else v
                     for k, v in b1.__dict__.items() if k != "rng"},
        })

        if (ep + 1) % 100 == 0:
            el = time.time() - t0
            print("  %4d/%d 局  行数 %7d  %.1fs  平均 %.0f 帧/秒" %
                  (ep + 1, args.episodes, len(rows), el, len(rows) / el), flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)

    meta = {
        "episodes": args.episodes,
        "rows": len(rows),
        "seed": args.seed,
        "fps": D.FPS,
        "max_frames": args.max_frames,
        "moves": {D.ACTION_NAMES[k]: v for k, v in D.MOVES.items()},
        "cols": COLS,
        "episode_list": episodes,
        "seconds": round(time.time() - t0, 2),
    }
    with open(args.out.replace(".csv", "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    el = time.time() - t0
    print("完成：%d 局 / %d 行 / %.1f 秒（%.0f 帧每秒）" %
          (args.episodes, len(rows), el, len(rows) / el))
    print("每局平均 %.1f 秒对局时长" % (np.mean([e["frames"] for e in episodes]) / D.FPS))


if __name__ == "__main__":
    main()
