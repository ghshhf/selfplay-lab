# -*- coding: utf-8 -*-
"""probe_chain.py —— 对照实验：**换题型**到底能不能难住笔。

背景
----
上一轮（probe_adversarial.py）已经用 360 组证明了一件事：在**同一个题池里**
换挑法（随机 vs 盯着冠军基因加权）**零效果** —— 随机 338 / 对抗 338，一模一样。

那不是 bug（修过 counters 方向后重跑仍然零），是结论：

    加权只是在「同一个（过低的）难度平面」里重新洗牌，抬不高天花板。

于是问题自然换成了：**换题型本身**行不行？本脚本验证的就是这一步。

    A 组（对照）：chain_ratio=0 → 旧题型，单文件 + 单点 bug 注入
    B 组（实验）：chain_ratio=1 → 跨层隐含契约题（chains.py）
        bug 在 core 层，但 core 层的单元测试**仍然全绿**；
        只有理解了 service 隐含依赖了什么，才知道它错在哪。

两组跑**同一个 seed 序列**、同一个冠军基因、同一套难度旋钮（Lv 全开），
唯一差别是题型，这样差异才可归因。

额外观测
--------
跨层题的 editable 同时开放 core 与 service，因此可以看出笔是
**修根因**还是**在上层打补丁** —— 这在旧的单文件题里看不到。
脚本会统计最终落地的补丁改了哪个文件。
"""
import argparse
import json
import os
import random
import time
from collections import Counter

from env import GENE_NAMES, Bot, CodeEnv

# 与当前实际运行难度保持一致（Lv 全开）
LEVELS = {"size": 3, "depth": 3, "hidden": 3, "scaffold": 3, "resource": 3}


def load_champ(path):
    """从 best_now.json 读真实冠军基因，返回 ({名:值}, [原始顺序])。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    gene = list(d["gene"])
    return {GENE_NAMES[i]: float(gene[i]) for i in range(len(GENE_NAMES))}, gene


def run_one(gene_list, env, reps=1):
    """用给定基因跑一局，返回 (passed, error, ms, files)。固定 bot rng 以降噪。"""
    bot = Bot(gene_list)
    bot.rng = random.Random(20261029)
    last = (False, "", 0, None)
    for _ in range(max(1, reps)):
        passed, error, ms, _out, files = bot.submit(env)
        last = (bool(passed), error, ms, files)
        if passed:
            break
    return last


def _patched_where(env, files):
    """对比交付前/后的文件，判断补丁落在哪个层。"""
    if not files:
        return []
    return sorted(n for n in env.editable
                  if files.get(n) != env.files.get(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每组出题局数")
    ap.add_argument("--seed", type=int, default=101, help="基础随机种子")
    ap.add_argument("--reps", type=int, default=1, help="每局允许的重试次数")
    ap.add_argument("--best", default="evo/best_now.json")
    ap.add_argument("--out", default="reports/chain_probe.json")
    args = ap.parse_args()

    champ_d, champ_l = load_champ(args.best)
    print("真实冠军基因（两组共用）:")
    print("  " + "  ".join("%s=%.2f" % (k, v) for k, v in champ_d.items()))
    print("难度旋钮: %s" % LEVELS)
    print()

    rows = []
    t_all = time.time()
    for i in range(args.n):
        seed = args.seed + i * 17
        # A 组：旧题型（单文件）
        envA = CodeEnv(seed=seed, champ=None, chain_ratio=0.0, **LEVELS)
        pA, eA, msA, fA = run_one(champ_l, envA, reps=args.reps)
        # B 组：跨层题
        envB = CodeEnv(seed=seed, champ=None, chain_ratio=1.0, **LEVELS)
        pB, eB, msB, fB = run_one(champ_l, envB, reps=args.reps)

        rows.append({
            "seed": seed,
            "A_task": envA.project["id"], "A_pass": pA, "A_ms": round(msA),
            "A_patched": _patched_where(envA, fA),
            "B_task": envB.project["id"], "B_pass": pB, "B_ms": round(msB),
            "B_patched": _patched_where(envB, fB),
            "B_counter": getattr(envB, "counter", ""),
        })
        print("[%02d/%02d] seed=%-5d A:%-30s %-4s | B:%-30s %-4s %s"
              % (i + 1, args.n, seed, envA.project["id"],
                 "PASS" if pA else "FAIL",
                 envB.project["id"], "PASS" if pB else "FAIL",
                 ("改了:" + ",".join(rows[-1]["B_patched"]))
                 if rows[-1]["B_patched"] else ""))

    nA = sum(1 for r in rows if r["A_pass"])
    nB = sum(1 for r in rows if r["B_pass"])
    N = max(1, args.n)

    # 层归因统计：在跨层题上，笔是修 core（根因）还是改 service（打补丁）
    patch_layer = Counter()
    for r in rows:
        # 跨层题的 id 形如 chain::<fam>::<bp>，bug 文件名从 files 里拿不到，
        # 改用 counter 里记录的 "core X→service Y"
        c = r.get("B_counter", "")
        if "→" in c:
            core_file, svc_file = c.split("→")
            core_file = core_file.replace("跨层契约 ", "").strip()
            svc_file = svc_file.strip()
            touched = r["B_patched"]
            if not touched:
                patch_layer["未改动"] += 1
            elif core_file in touched and svc_file in touched:
                patch_layer["两层都改"] += 1
            elif core_file in touched:
                patch_layer["只改 core（根因）"] += 1
            elif svc_file in touched:
                patch_layer["只改 service（打补丁）"] += 1

    summary = {
        "n": args.n, "reps": args.reps, "seed": args.seed, "levels": LEVELS,
        "A_pass": nA, "A_rate": round(nA / N, 3),
        "B_pass": nB, "B_rate": round(nB / N, 3),
        "delta_rate": round((nB - nA) / N, 3),
        "A_ms_avg": round(sum(r["A_ms"] for r in rows) / N),
        "B_ms_avg": round(sum(r["B_ms"] for r in rows) / N),
        "dist_A": dict(Counter(r["A_task"] for r in rows).most_common()),
        "dist_B": dict(Counter(r["B_task"] for r in rows).most_common()),
        "patch_layer": dict(patch_layer),
        "sec": round(time.time() - t_all, 1),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f,
                  ensure_ascii=False, indent=2)

    print("\n=== 结论 ===")
    print("旧题型（单文件）  成功率 %.3f (%d/%d)  平均 %dms"
          % (summary["A_rate"], nA, args.n, summary["A_ms_avg"]))
    print("跨层契约题        成功率 %.3f (%d/%d)  平均 %dms"
          % (summary["B_rate"], nB, args.n, summary["B_ms_avg"]))
    print("差值              %+.3f" % summary["delta_rate"])
    print("\n跨层题上层归因（笔改了哪一层）:")
    for k, v in patch_layer.most_common():
        print("  %-24s %d" % (k, v))
    print("\n逐题:")
    for r in rows:
        if not r["B_pass"]:
            print("  FAIL %-32s %s" % (r["B_task"], ",".join(r["B_patched"]) or "(未改)"))
    print("\n明细: %s" % args.out)


if __name__ == "__main__":
    main()
