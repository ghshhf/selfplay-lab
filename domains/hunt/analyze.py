# -*- coding: utf-8 -*-
"""分析 hunt 共进化的一轮结果。

回答三个问题，全部用数字，不用印象：

1. **有没有形成军备竞赛**？两侧 best 是不是都在涨，
   还是一边倒地把另一边吃掉了（后者的 fit 会撞在 0 或 1 上不动）。
2. **信念有没有真的学会东西**？信念熵该降、峰值命中该升、NLL 该降。
   关键要看它们跟代数的相关性 —— 三条一起动才算数，动一条是巧合。
   注意：随着对手变强，这三个数也可能一起变差，所以必须同时看胜率。
3. **不完全信息的成本是多少**？完全信息对照 − 不完全信息 = 推断的负担。
   如果两者差不多，说明这个环境里的"推断"其实没什么必要，域就白设计了。
"""
import argparse
import csv
import math
import os
import statistics


def load(tsv):
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                rows.append({
                    "gen": int(r["gen"]),
                    "best_h": float(r["best_h"]), "mean_h": float(r["mean_h"]),
                    "best_e": float(r["best_e"]), "mean_e": float(r["mean_e"]),
                    "div_h": float(r["div_h"]), "div_e": float(r["div_e"]),
                    "ent": float(r["belief_entropy"]),
                    "nll": float(r["belief_nll"]),
                    "top1": float(r["belief_top1"]),
                    "ctrl": (float(r["visible_ctrl"])
                             if r.get("visible_ctrl") else None),
                    "level": int(r["level"]),
                })
            except (TypeError, ValueError):
                continue
    return rows


def corr(xs, ys):
    """皮尔逊相关。样本 < 3 时返回 None（不猜）。"""
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def seg(rows, key, lo, hi):
    vals = [r[key] for r in rows if lo <= r["gen"] <= hi]
    return statistics.mean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="evo/progress.tsv")
    ap.add_argument("--win", type=int, default=60, help="头尾对比窗口代数")
    args = ap.parse_args()

    if not os.path.exists(args.tsv):
        print("找不到 %s" % args.tsv)
        return
    rows = load(args.tsv)
    if len(rows) < 10:
        print("样本太少（%d 代），不足以出结论" % len(rows))
        return

    gens = [r["gen"] for r in rows]
    head, tail = gens[0], gens[-1]
    w = args.win
    lo_h, hi_h = head, head + w
    lo_t, hi_t = max(head, tail - w), tail

    print("=== hunt 共进化分析：%s（%d 代，gen %d → %d）==="
          % (args.tsv, len(rows), head, tail))
    print()

    # 1. 军备竞赛还是一边倒
    print("【1】有没有形成军备竞赛（头 %d 代 vs 尾 %d 代）" % (w, w))
    print("  %-14s %10s %10s %10s" % ("指标", "开局", "收尾", "变化"))
    for k, label in (("best_h", "猎手最佳"), ("best_e", "逃犯最佳"),
                     ("mean_h", "猎手均值"), ("mean_e", "逃犯均值"),
                     ("div_h", "猎手多样性"), ("div_e", "逃犯多样性")):
        a, b = seg(rows, k, lo_h, hi_h), seg(rows, k, lo_t, hi_t)
        print("  %-14s %10.3f %10.3f %+10.3f" % (label, a, b, b - a))

    tail_h = [r["best_h"] for r in rows if r["gen"] >= lo_t]
    tail_e = [r["best_e"] for r in rows if r["gen"] >= lo_t]
    h_locked = statistics.mean(tail_h) >= 0.97 or statistics.mean(tail_h) <= 0.03
    e_locked = statistics.mean(tail_e) >= 0.97 or statistics.mean(tail_e) <= 0.03
    if h_locked or e_locked:
        who = "逃犯" if statistics.mean(tail_h) <= 0.03 else "猎手"
        print("  → 一边倒：%s 被压到 %.3f，另一方锁死在 %.3f。没有竞赛。"
              % (who, statistics.mean(tail_h) if h_locked else statistics.mean(tail_e),
                 statistics.mean(tail_e) if e_locked else statistics.mean(tail_h)))
    elif (statistics.mean(tail_h) - seg(rows, "best_h", lo_h, hi_h)) > 0.02 and \
            (statistics.mean(tail_e) - seg(rows, "best_e", lo_h, hi_h)) > 0.02:
        print("  → 双方都在进步：这是一场真的军备竞赛")
    else:
        print("  → 未出现明确的双边进步")

    # 2. 信念有没有学会东西
    print()
    print("【2】信念质量随代数的变化（相关系数，越大越相关）")
    print("  %-10s %10s %10s %10s" % ("指标", "开局", "收尾", "与代数相关系数"))
    for k, label in (("ent", "信念熵"), ("nll", "真值NLL"), ("top1", "峰值命中")):
        vals = [r[k] for r in rows]
        a, b = seg(rows, k, lo_h, hi_h), seg(rows, k, lo_t, hi_t)
        c = corr(gens, vals)
        cs = "n/a" if c is None else "%+.3f" % c
        print("  %-10s %10.3f %10.3f %14s" % (label, a, b, cs))

    c_ent = corr(gens, [r["ent"] for r in rows])
    c_top = corr(gens, [r["top1"] for r in rows])
    c_nll = corr(gens, [r["nll"] for r in rows])
    learned = (c_ent is not None and c_ent < -0.2 and c_top is not None and c_top > 0.2
               and c_nll is not None and c_nll < -0.2)
    if learned:
        print("  → 三条一起朝好的方向走：信念确实学到了东西")
    else:
        print("  → 三条没有一致朝好走（对手也在变强时这是合理结果，见下）")

    # 信念变好 vs 对手变强：把胜率一并看
    c_h = corr(gens, [r["best_h"] for r in rows])
    print()
    print("  猎手最佳胜率与代数的相关系数: %s"
          % ("n/a" if c_h is None else "%+.3f" % c_h))
    print("  （若信念指标变差而猎手胜率不变差，说明是**对手变强**而不是**推断退化**；")
    print("    反过来若胜率也掉，那才是自己没学会。）")

    # 3. 不完全信息的成本
    ctrl = [r for r in rows if r["ctrl"] is not None]
    if ctrl:
        print()
        print("【3】不完全信息的成本（完全信息对照组）")
        print("  %-8s %12s %12s %10s" % ("gen", "完全信息", "不完全信息", "差额"))
        for r in ctrl:
            print("  %-8d %12.3f %12.3f %+10.3f"
                  % (r["gen"], r["ctrl"], r["best_h"], r["ctrl"] - r["best_h"]))
        gaps = [r["ctrl"] - r["best_h"] for r in ctrl]
        print("  平均差额 %+.3f" % statistics.mean(gaps))
        if statistics.mean(gaps) > 0.10:
            print("  → 推断确实有负担，域设计成立")
        else:
            print("  → 差额很小：这个环境里推断几乎不必要，域需要更难")
    else:
        print()
        print("【3】没有完全信息对照数据（visible_ctrl 列为空）")


if __name__ == "__main__":
    main()
