# -*- coding: utf-8 -*-
"""probe_adversarial.py —— 对照实验：随机出题 vs 对抗出题，谁更难住冠军。

背景
----
code 域跑了 2500+ 代，冠军 fit 长期 =1.000（99.6% 的记录满分），选择压力归零。
一个待验证的假设是：出题是纯随机的，对「暴力撞」型冠军几乎无压力，改成
「盯着冠军基因定向出题」（adversarial.py）应当能恢复梯度。

假设必须被验证，不能只是"听起来对"。本脚本做 A/B：

    A 组（对照）：champ=None          → 均匀随机挑 bug 模式（改造前的行为）
    B 组（实验）：champ=<真实冠军基因> → 按 counters 加权挑克制它的 bug

两组跑**同一个 seed 序列**，唯一差别是挑选策略，这样差异才可归因。

诚实边界
--------
counters 标签是人工标注的假设（见 adversarial.py 文档）。本脚本的结果就是
裁定它的依据：若 B 组成功率显著低于 A 组，说明标签方向对；若持平或更高，
说明标签只是自我安慰，得换思路——结论照实写，不硬撑。
"""
import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np

import tasks
import adversarial as adv
from env import GENE_NAMES, BOUNDS, CodeEnv, Bot

# 与当前实际运行难度保持一致（Lv5 全开）
LEVELS = {"size": 3, "depth": 3, "hidden": 3, "scaffold": 3, "resource": 3}


def bounds_map():
    """{基因名: (lo, hi)}，来自 env 的 BOUNDS。"""
    return {GENE_NAMES[i]: tuple(BOUNDS[i]) for i in range(len(GENE_NAMES))}


def load_champ(path):
    """从 best_now.json 读真实冠军基因，返回 ({名:值}, [原始顺序])。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    gene = list(d["gene"])
    return {GENE_NAMES[i]: float(gene[i]) for i in range(len(GENE_NAMES))}, gene


def coverage_report():
    """标注覆盖率：多少 bug 模式带 counters 标签（含 COUNTERS 表兜底）。"""
    total = 0
    miss = []
    for fam in tasks.GOLDEN:
        for bp in fam["bug_patterns"]:
            total += 1
            if not (bp.get("counters")
                    or adv.COUNTERS.get((fam["id"], bp["name"]))):
                miss.append("%s::%s" % (fam["id"], bp["name"]))
    return {"total": total, "labeled": total - len(miss), "missing": miss}


def run_one(gene_list, env, reps=1):
    """用给定基因跑一局，返回 (passed, error, ms)。固定 bot rng 以降噪。"""
    bot = Bot(gene_list)
    bot.rng = random.Random(20261029)
    last = (False, "", 0)
    for _ in range(max(1, reps)):
        passed, error, ms, _out, _files = bot.submit(env)
        last = (bool(passed), error, ms)
        if passed:
            break
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每组出题局数")
    ap.add_argument("--seed", type=int, default=101, help="基础随机种子")
    ap.add_argument("--reps", type=int, default=1,
                    help="每局允许的重试次数（1=只看题本身难度）")
    ap.add_argument("--best", default="evo/best_now.json")
    ap.add_argument("--out", default="reports/adversarial_probe.json")
    args = ap.parse_args()

    cov = coverage_report()
    print("标注覆盖率: %d/%d" % (cov["labeled"], cov["total"]))
    if cov["missing"]:
        print("  未标注（走均匀权重）:", ", ".join(cov["missing"]))

    champ_d, champ_l = load_champ(args.best)
    print("真实冠军基因:")
    print("  " + "  ".join("%s=%.2f" % (k, v) for k, v in champ_d.items()))
    print()

    rows = []
    t_all = time.time()
    for i in range(args.n):
        seed = args.seed + i * 17
        # A 组：随机出题（改造前行为）
        envA = CodeEnv(seed=seed, champ=None, **LEVELS)
        pA, eA, msA = run_one(champ_l, envA, reps=args.reps)
        # B 组：对抗出题（针对冠军挑选）
        envB = CodeEnv(seed=seed, champ=champ_d, **LEVELS)
        pB, eB, msB = run_one(champ_l, envB, reps=args.reps)
        rows.append({
            "seed": seed,
            "A_task": envA.project["id"], "A_pass": pA, "A_ms": round(msA),
            "B_task": envB.project["id"], "B_pass": pB, "B_ms": round(msB),
            "B_counter": getattr(envB, "counter", ""),
        })
        print("[%02d/%02d] seed=%-5d A:%-26s %-4s | B:%-26s %-4s"
              % (i + 1, args.n, seed, envA.project["id"],
                 "PASS" if pA else "FAIL",
                 envB.project["id"], "PASS" if pB else "FAIL"))

    nA = sum(1 for r in rows if r["A_pass"])
    nB = sum(1 for r in rows if r["B_pass"])
    N = max(1, args.n)
    # 分布统计：对抗出题到底有没有改变抽到的题？
    # 这一条很关键——如果分布没变，说明加权逻辑压根没生效（实现问题）；
    # 如果分布变了但成功率没变，说明是题池本身太弱（设计问题）。两者结论不同。
    # 注意粒度：对抗加权作用在 **bug 模式**（fam::bp）层面，不是族层面。
    # 族仍是 rng.choice 均匀挑的，只统计族会看不出任何差异。
    distA = Counter(r["A_task"] for r in rows)
    distB = Counter(r["B_task"] for r in rows)
    summary = {
        "n": args.n, "reps": args.reps, "seed": args.seed, "levels": LEVELS,
        "A_pass": nA, "A_rate": round(nA / N, 3),
        "B_pass": nB, "B_rate": round(nB / N, 3),
        "delta_rate": round((nB - nA) / N, 3),
        "A_ms_avg": round(sum(r["A_ms"] for r in rows) / N),
        "B_ms_avg": round(sum(r["B_ms"] for r in rows) / N),
        "dist_A": dict(distA.most_common()),
        "dist_B": dict(distB.most_common()),
        "sec": round(time.time() - t_all, 1),
        "coverage": cov,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f,
                  ensure_ascii=False, indent=2)

    print("\n=== 结论 ===")
    print("随机出题  成功率 %.3f (%d/%d)  平均耗时 %dms"
          % (summary["A_rate"], nA, args.n, summary["A_ms_avg"]))
    print("对抗出题  成功率 %.3f (%d/%d)  平均耗时 %dms"
          % (summary["B_rate"], nB, args.n, summary["B_ms_avg"]))
    print("差值      %+.3f" % summary["delta_rate"])
    print("\n题目分布（族 -> 次数）")
    keys = sorted(set(distA) | set(distB))
    for k in keys:
        print("  %-14s  随机 %2d  对抗 %2d" % (k, distA.get(k, 0), distB.get(k, 0)))
    if summary["delta_rate"] < 0:
        print("→ 对抗出题确实更难住冠军，梯度有恢复")
    elif summary["delta_rate"] > 0:
        print("→ 对抗出题反而更容易，counters 标注方向错了，需重标")
    else:
        print("→ 无差异，需加大样本或换思路")
    print("\n明细: %s" % args.out)


if __name__ == "__main__":
    main()
