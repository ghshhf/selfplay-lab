# -*- coding: utf-8 -*-
"""冠军对冠军的推断代价测量（巅峰口径）。

为什么不用 analyze.py 里那个「完全信息 − 当轮最强」的差额：
那个差额的分子是一个**进化后的**策略，分母是**进化后的**对手，但完全信息
那一侧只换了信息条件、没换对手强度带来的天花板 —— 当对手强到「看得见也只有
六成胜率」时，两个天花板一起塌，差额被压平成 0.06 那种假小值。

这个脚本只做一件事：拿 best_now.json 的冠军双方，在**同一批 seed**（同一张图、
同一个起始局面）下各跑一遍不完全信息与完全信息，比较冠军在两种信息条件下的
真实胜率。这才是「推断值多少钱」的正确量法。

用法：
    python probe_peak.py --best evo_v4/best_now.json --n 160
"""
import argparse
import json
import math
import statistics

import env as E
from env import EvaderBot, HuntEnv, HunterBot

# 和 evolve.World.params() 的 Lv0 保持一致
BASE = dict(nodes=20, extra=8, obs_every=10, max_turns=75, obs_noise=0.6,
            n_machines=5, need=3, machine_ticks=2, gate_ticks=3,
            sabotage=0.22, hunter_slow=2)


def two_prop_z(k1, n1, k2, n2):
    """两个比例差的 z 统计量。用来确认差额不是噪声。"""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))
    return 0.0 if se <= 1e-12 else (p1 - p2) / se


def batch(gh, ge, n, seed0, visible=None):
    rows = []
    for i in range(n):
        r = HuntEnv(seed=seed0 + i, visible=visible, **BASE).run(
            HunterBot(gh), EvaderBot(ge))
        rows.append(r)
    return rows


def summarize(rows, tag):
    n = len(rows)
    esc = sum(1 for r in rows if r["escaped"])
    ca = sum(1 for r in rows if r["caught"])
    to = n - esc - ca
    md = statistics.mean(r["machines_done"] for r in rows)
    st = statistics.mean(r["still_turns"] for r in rows)
    tn = statistics.mean(r["turns"] for r in rows)
    print("  %-12s 逃出 %5.1f%%  被抓 %5.1f%%  超时 %5.1f%%  "
          "修机 %.2f/台  静止 %.1f回合(%.0f%%)"
          % (tag, 100.0 * esc / n, 100.0 * ca / n, 100.0 * to / n,
             md, st, 100.0 * st / max(1e-9, tn)))
    return esc, ca, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="evo_v4/best_now.json")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed0", type=int, default=9000)
    args = ap.parse_args()

    d = json.load(open(args.best, encoding="utf-8"))
    gh, ge = d["hunter"], d["evader"]
    print("=== 冠军对冠军（%s，gen %d，n=%d）===" % (args.best, d["gen"], args.n))
    print("  进化时的 fit_h=%.3f fit_e=%.3f（那是和**种群内随机对手**打的，不是这里）"
          % (d["fit_h"], d["fit_e"]))
    print()

    a = batch(gh, ge, args.n, args.seed0)
    b = batch(gh, ge, args.n, args.seed0, visible=1)
    esc_a, ca_a, n = summarize(a, "不完全信息")
    esc_b, ca_b, _ = summarize(b, "完全信息")
    print()

    # 逃犯视角：推断缺失 → 逃出率变化。猎手视角：抓捕率变化。
    d_esc = 100.0 * (esc_a - esc_b) / n
    d_ca = 100.0 * (ca_b - ca_a) / n
    z = two_prop_z(ca_a, n, ca_b, n)
    print("  推断代价（猎手抓捕率）: %+.1f 个百分点  （%.1f%% → %.1f%%）"
          % (d_ca, 100.0 * ca_a / n, 100.0 * ca_b / n))
    print("  推断代价（逃犯逃出率）: %+.1f 个百分点  （%.1f%% → %.1f%%）"
          % (d_esc, 100.0 * esc_a / n, 100.0 * esc_b / n))
    print("  比例差 z = %.2f  %s" % (z, "显著" if abs(z) >= 1.96 else "**不显著**"))
    print()
    if abs(z) >= 1.96 and d_ca > 5:
        print("  → 冠军之间，信息仍然是胜负手：看得见能让抓捕率抬高 %.1f 个点。"
              % d_ca)
    elif abs(z) < 1.96:
        print("  → 冠军之间，信息已经不再是胜负手（差额在噪声里）。")
    else:
        print("  → 信息有影响，但幅度有限。")


if __name__ == "__main__":
    main()
