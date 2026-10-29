"""
inspect_stream.py —— 常驻管线的数据质检（读 shards/*.csv.gz，不看内存状态）

inspect_duel.py 管一次性采集的 duel.csv；这个管 stream_duel.py 吐出来的分片，
问的是另外几个问题：
  1. 分片有没有缺口、有没有重复编号（重启续跑最容易出这种错）
  2. 规则还是不是那套规则（血量单调、场地边界、射程、伤害对账）
  3. 打法覆盖到底撑开了没有 —— 这是常驻跑的全部意义
  4. 现在这个速度，一天会长多少，什么时候会撞上体积上限

用法：
    python inspect_stream.py                 # 看 shards/
    python inspect_stream.py --dir shards --last 3   # 只看最近 3 片
"""

import argparse
import csv
import glob
import gzip
import json
import os

import numpy as np

import duel as D
from stream_duel import COLS, REACT_EDGES, AGGR_EDGES

REACH = {1: D.MOVES[D.PUNCH][3], 2: D.MOVES[D.KICK][3]}
DMG = {1: D.MOVES[D.PUNCH][4], 2: D.MOVES[D.KICK][4]}
BLK_MUL = D.BLOCK_DAMAGE_MUL


def read_shard(csv_path):
    with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="shards")
    ap.add_argument("--last", type=int, default=0, help="只检查最近 N 片，0=全部")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "part-*.csv.gz")))
    if not paths:
        print("没有找到分片：%s/part-*.csv.gz" % args.dir)
        return
    total_paths = len(paths)
    if args.last:
        paths = paths[-args.last:]

    bad = []
    idx = []
    all_eps = {}
    max_hp_jump = 0.0
    max_bal_err = 0.0
    reach_viol = 0
    oob = 0
    n_hit = n_blk = 0
    act = np.zeros(6, dtype=np.int64)
    kinds = {}
    react_h = np.zeros(len(REACT_EDGES) - 1, dtype=np.int64)
    aggr_h = np.zeros(len(AGGR_EDGES) - 1, dtype=np.int64)
    grid = np.zeros((len(REACT_EDGES) - 1, len(AGGR_EDGES) - 1), dtype=np.int64)
    raw_bytes = comp_bytes = 0

    for p in paths:
        i = int(os.path.basename(p).split("-")[1].split(".")[0])
        idx.append(i)
        comp_bytes += os.path.getsize(p)
        meta_p = p.replace(".csv.gz", ".meta.json")
        try:
            with open(meta_p, encoding="utf-8") as fh:
                meta = json.load(fh)
        except FileNotFoundError:
            meta = {}
            bad.append("片 %05d 缺 meta" % i)

        # 列必须一字不差 —— 列漂了训练时会静默错位，最难查
        if meta.get("cols") and meta["cols"] != COLS:
            bad.append("片 %05d 列不一致" % i)

        eps_meta = {e["ep"]: e for e in meta.get("episode_list", [])}
        for e in meta.get("episode_list", []):
            for k in e.get("kind", []):
                kinds[k] = kinds.get(k, 0) + 1

        rows = read_shard(p)
        # 未压缩体积 ≈ 每行各字段字符数 + 逗号 + 换行。别用 len(str(dict))，那会把键名也算进去。
        raw_bytes += sum(sum(len(str(v)) for v in r.values()) + len(r) for r in rows)
        by_ep = {}
        for r in rows:
            by_ep.setdefault(int(r["ep"]), []).append(r)

        for ep, rs in by_ep.items():
            if ep in all_eps:
                bad.append("ep%d 在多个分片里重复出现" % ep)
            all_eps[ep] = rs

            # 1) 血量单调不增
            for side in ("hp0", "hp1"):
                v = np.array([float(x[side]) for x in rs])
                j = np.diff(v)
                if j.size:
                    max_hp_jump = max(max_hp_jump, float(j.max()))
                    if j.max() > 1e-9:
                        bad.append("ep%d %s 血量回升 %.3f" % (ep, side, j.max()))

            # 2) 场地边界
            x0 = np.abs([float(x["x0"]) for x in rs])
            x1 = np.abs([float(x["x1"]) for x in rs])
            oob += int((x0 > D.ARENA_HALF + 1e-6).sum() + (x1 > D.ARENA_HALF + 1e-6).sum())

            # 3) 射程用 hdist（判定当时的距离），不是 dist（积分后的观测值）
            for r in rs:
                mv = int(r["mv"])
                if mv:
                    n_hit += 1
                    n_blk += int(r["blk"])
                    hd = float(r["hdist"])
                    if hd > REACH[mv] + 1e-3:
                        reach_viol += 1
                        if reach_viol <= 3:
                            bad.append("ep%d 招式%d 在 %.3fm 命中（上限 %.2f）"
                                       % (ep, mv, hd, REACH[mv]))
            act[0] += sum(int(x["a0"]) == 0 for x in rs)
            act[1] += sum(int(x["a0"]) == 1 for x in rs)
            act[2] += sum(int(x["a0"]) == 2 for x in rs)
            act[3] += sum(int(x["a0"]) == 3 for x in rs)
            act[4] += sum(int(x["a0"]) == 4 for x in rs)
            act[5] += sum(int(x["a0"]) == 5 for x in rs)

            # 4) 伤害对账：本局掉血 vs meta 记录的 damage
            m = eps_meta.get(ep)
            if m is not None:
                drop = 2 * D.MAX_HP - (float(rs[-1]["hp0"]) + float(rs[-1]["hp1"]))
                err = abs(drop - sum(m["damage"]))
                max_bal_err = max(max_bal_err, err)
                if err > 5e-3:
                    bad.append("ep%d 掉血 %.3f 与记录 %.3f 不符" % (ep, drop, sum(m["damage"])))

    # ---- 分片连续性 ----
    gaps = [b - a for a, b in zip(idx, idx[1:]) if b - a != 1]
    if gaps:
        bad.append("分片编号不连续，缺口 %d 处（%s）" % (len(gaps), gaps[:5]))

    # ---- 覆盖账本 ----
    cov_path = os.path.join(os.path.dirname(args.dir.rstrip("/\\")) or ".", "coverage.json")
    cov = {}
    if os.path.exists(cov_path):
        with open(cov_path, encoding="utf-8") as fh:
            cov = json.load(fh)
        for rb in range(len(REACT_EDGES) - 1):
            react_h[rb] = cov["react_hist"][rb]
        for ab in range(len(AGGR_EDGES) - 1):
            aggr_h[ab] = cov["aggr_hist"][ab]
        for rb in range(len(REACT_EDGES) - 1):
            for ab in range(len(AGGR_EDGES) - 1):
                grid[rb][ab] = cov["grid"][rb][ab]

    lens = np.array([len(v) for v in all_eps.values()])
    act_sum = act.sum()
    ratio = (raw_bytes / comp_bytes) if comp_bytes else 0

    # ---- 局面覆盖：比"打法原型覆盖"更硬的一层 ----
    # 20 格的 react×aggr 矩阵 300 局就填满了，它只能说明"手都上过场"，
    # 不能说明"局面都被走过"。这里看的是真正进模型的状态分布。
    DIST_EDGES = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 99]
    dist_h = np.zeros(len(DIST_EDGES) - 1, dtype=np.int64)
    wall = 0          # 贴墙帧（|x|>3.5，被边界卡住的局面）
    clash = 0         # 双方同帧出招（对撞）
    idle_both = 0     # 双方同帧都不动（无效帧，越多说明数据越水）
    ev_lost = 0       # 同帧两次命中被挤掉的次数（一行只能装一个事件）
    for ep, rs in all_eps.items():
        d = np.array([float(x["dist"]) for x in rs])
        for i in range(len(DIST_EDGES) - 1):
            dist_h[i] += int(((d >= DIST_EDGES[i]) & (d < DIST_EDGES[i + 1])).sum())
        x0 = np.abs([float(x["x0"]) for x in rs])
        x1 = np.abs([float(x["x1"]) for x in rs])
        wall += int((np.maximum(x0, x1) > 3.5).sum())
        a0 = np.array([int(x["a0"]) for x in rs])
        a1 = np.array([int(x["a1"]) for x in rs])
        clash += int(((a0 >= 2) & (a0 <= 3) & (a1 >= 2) & (a1 <= 3)).sum())
        idle_both += int(((a0 == 5) & (a1 == 5)).sum())
        # 一行只能装一个事件：若某帧双方同时命中，必有一条被挤到下一帧或丢掉。
        # 用 meta 记录的 hits 和 CSV 里的事件行数对一下，差额就是丢的事件。
        m = eps_meta.get(ep)
        if m is not None:
            ev_lost += max(0, int(m["hits"]) - int((np.array([int(x["atk"]) for x in rs]) >= 0).sum()))

    print("=" * 66)
    print("分片 %d/%d 片  局数 %d  帧数 %d" % (len(paths), total_paths, len(all_eps), lens.sum()))
    print("压缩率 %.1f:1   落盘 %.2f MB   未压缩约 %.1f MB"
          % (ratio, comp_bytes / 2**20, raw_bytes / 2**20))
    print("-" * 66)
    print("血量单调回升      %.2e      （必须 0）" % max_hp_jump)
    print("场地越界          %d 次" % oob)
    print("射程违规          %d 次" % reach_viol)
    print("伤害对账最大差    %.2e      （必须 < 5e-3）" % max_bal_err)
    print("命中 %d  格挡率 %.0f%%" % (n_hit, 100 * n_blk / max(1, n_hit)))
    print("局长度 p50/p95    %d / %d 帧" % (np.percentile(lens, 50), np.percentile(lens, 95)))
    print("-" * 66)
    print("动作分布（a0 侧）：", end="")
    for i, n in enumerate(D.ACTION_NAMES):
        print(" %s %.0f%%" % (n, 100 * act[i] / max(1, act_sum)), end="")
    print()
    if kinds:
        print("打法覆盖：", end="")
        tot = sum(kinds.values())
        for k in sorted(kinds, key=lambda x: -kinds[x]):
            print(" %s %.0f%%" % (k, 100 * kinds[k] / tot), end="")
        print()

    # ---- 局面覆盖 ----
    tot_frames = max(1, int(lens.sum()))
    print("-" * 66)
    print("局面覆盖（这才是模型真正见到的东西）")
    print("  间距分布：", end="")
    for i in range(len(DIST_EDGES) - 1):
        p = 100 * dist_h[i] / max(1, dist_h.sum())
        lab = "%g-%g" % (DIST_EDGES[i], DIST_EDGES[i + 1]) if DIST_EDGES[i + 1] < 99 else "6+"
        print(" %s:%.0f%%" % (lab, p), end="")
    print()
    print("  贴墙帧 %.1f%%   同帧对撞 %.1f%%   双方都发呆 %.1f%%"
          % (100 * wall / tot_frames, 100 * clash / tot_frames, 100 * idle_both / tot_frames))
    if ev_lost:
        print("  ⚠ 同帧双命中被挤掉 %d 次（占命中 %.2f%%）——一行只装得下一个事件，"
              "这是记录格式的先天缺陷" % (ev_lost, 100 * ev_lost / max(1, n_hit + ev_lost)))
    else:
        print("  同帧双命中 0 次（或都被完整记录）")

    if cov:
        print("-" * 66)
        print("覆盖矩阵（行=反应延迟档，列=攻击性档；数字是该格出现的人次）")
        rlab = ["%d-%d帧" % (REACT_EDGES[i], REACT_EDGES[i + 1] - 1)
                for i in range(len(REACT_EDGES) - 1)]
        alab = ["%.2f-%.2f" % (AGGR_EDGES[j], AGGR_EDGES[j + 1])
                for j in range(len(AGGR_EDGES) - 1)]
        print("%9s" % "" + "".join("%11s" % a for a in alab))
        for i, rl in enumerate(rlab):
            print("%9s" % rl + "".join("%11d" % grid[i][j] for j in range(len(alab))))
        zero = int((grid == 0).sum())
        print("空格 %d / %d 格%s" % (zero, grid.size,
                                  "  ← 还有没碰过的组合，继续跑" if zero else "  ← 全格都盖到了"))
        if cov.get("bytes") and cov.get("episodes"):
            per_ep = cov["bytes"] / cov["episodes"]
            print("-" * 66)
            print("累计 %d 局 / %.2f GB，平均 %.1f KB/局 → 每天 1 局/秒 约 %.1f GB/天"
                  % (cov["episodes"], cov["bytes"] / 2**30, per_ep / 1024,
                     per_ep * 86400 / 2**30))
            print("首片 %s   最新 %s" % (cov.get("first_ts"), cov.get("last_ts")))
    print("=" * 66)
    if bad:
        print("发现 %d 处问题：" % len(bad))
        for b in bad[:10]:
            print("  ✗", b)
    else:
        print("✓ 分片连续、规则一致、账对得上，没有一条违规")


if __name__ == "__main__":
    main()
