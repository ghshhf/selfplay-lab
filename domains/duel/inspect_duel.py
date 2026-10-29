"""
inspect_duel.py —— 从落盘 CSV 反算校验（不看环境内部状态，只看数据和规则能不能对上）

这才是"数据可信"的证据：如果规则说拳只能打 1.15m，那么 CSV 里不该出现
1.4m 的拳命中；如果规则说格挡只吃 28% 伤害，那么 -7 × 0.28 必须对得上血量变化。
"""

import csv
import json
import numpy as np

import duel as D

ARENA = D.ARENA_HALF
REACH = {1: D.MOVES[D.PUNCH][3], 2: D.MOVES[D.KICK][3]}
DMG = {1: D.MOVES[D.PUNCH][4], 2: D.MOVES[D.KICK][4]}
BLK_MUL = D.BLOCK_DAMAGE_MUL


def load(path="duel.csv"):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(path="duel.csv"):
    rows = load(path)
    global meta_dmg
    meta_dmg = {}
    try:
        with open(path.replace(".csv", "_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        for e in meta.get("episode_list", []):
            meta_dmg[e["ep"]] = e["damage"]
        print("已载入采集时记录的逐局伤害（%d 局）用于精确对账" % len(meta_dmg))
    except FileNotFoundError:
        print("没有找到 meta 文件，跳过精确对账")
    eps = {}
    for r in rows:
        eps.setdefault(int(r["ep"]), []).append(r)

    bad = []
    n_hit = n_blk = 0
    max_hp_jump = 0.0
    max_bal_err = 0.0
    worst_reach = []
    act_cnt = np.zeros(6, dtype=np.int64)
    mv_cnt = {1: 0, 2: 0}
    ep_lens = []
    ko = 0

    for ep, rs in eps.items():
        ep_lens.append(len(rs))
        # ---- 1) 血量必须单调不增 ----
        for side in ("hp0", "hp1"):
            v = np.array([float(x[side]) for x in rs])
            jumps = np.diff(v)
            if jumps.size:
                max_hp_jump = max(max_hp_jump, float(jumps.max()))
                if jumps.max() > 1e-9:
                    bad.append("ep%d %s 血量回升 %.2f" % (ep, side, jumps.max()))

        # ---- 2) 位置/动作合法性 ----
        x = np.array([[float(a["x0"]), float(a["x1"])] for a in rs])
        if np.abs(x).max() > ARENA + 1e-6:
            bad.append("ep%d 位置越界 |x|max=%.3f" % (ep, np.abs(x).max()))
        for a in rs:
            a0, a1 = int(a["a0"]), int(a["a1"])
            if not (0 <= a0 < 6 and 0 <= a1 < 6):
                bad.append("ep%d frame%s 动作编号非法 %d/%d" % (ep, a["frame"], a0, a1))
            act_cnt[a0] += 1
            act_cnt[a1] += 1

        # ---- 3) 逐次命中：射程、伤害、格挡倍率 ----
        hits = [a for a in rs if int(a["atk"]) >= 0]
        hp0 = np.array([float(a["hp0"]) for a in rs])
        hp1 = np.array([float(a["hp1"]) for a in rs])
        for a in hits:
            n_hit += 1
            mv = int(a["mv"])
            vic = int(a["vic"])
            blk = int(a["blk"])
            dist = float(a["hdist"])
            mv_cnt[mv] = mv_cnt.get(mv, 0) + 1
            if blk:
                n_blk += 1
            # 射程校验
            if dist > REACH[mv] + 1e-3:
                worst_reach.append((dist, REACH[mv], mv))
                bad.append("ep%d frame%s 招式%d 在 %.3fm 命中，射程只有 %.2fm"
                           % (ep, a["frame"], mv, dist, REACH[mv]))
            # 伤害校验：用下一帧的血量差
            i = rs.index(a)
            if i + 1 < len(rs):
                key = "hp%d" % vic
                drop = float(rs[i][key]) - float(rs[i + 1][key])
                want = DMG[mv] * (BLK_MUL if blk else 1.0)
                # 同一帧可能被截断到 0，允许 drop <= want
                if drop > want + 1e-3:
                    bad.append("ep%d frame%s 掉血 %.2f 超过招式上限 %.2f"
                               % (ep, a["frame"], drop, want))

        # ---- 4) 总账对平：CSV 反算的掉血 vs 采集时记的 damage ----
        dmg_real = 2 * D.MAX_HP - (float(rs[-1]["hp0"]) + float(rs[-1]["hp1"]))
        rec = meta_dmg.get(ep)
        if rec is not None:
            err = abs(dmg_real - sum(rec))
            max_bal_err = max(max_bal_err, err)
            if err > 5e-3:
                bad.append("ep%d 掉血 %.2f 与记录伤害 %.2f 不符" % (ep, dmg_real, sum(rec)))
        # 上限校验：重算的招式总量不该小于真实掉血
        dmg_cap = sum(DMG[int(a["mv"])] * (BLK_MUL if int(a["blk"]) else 1.0) for a in hits)
        if dmg_real > dmg_cap + 1e-3:
            bad.append("ep%d 掉血 %.2f 超过招式伤害上限 %.2f" % (ep, dmg_real, dmg_cap))

        if float(rs[-1]["hp0"]) <= 0 or float(rs[-1]["hp1"]) <= 0:
            ko += 1

    print("=" * 62)
    print("CSV 反算校验  %s" % path)
    print("=" * 62)
    print("行数 %d   回合 %d   平均每局 %.0f 帧 (%.1f 秒)" %
          (len(rows), len(eps), np.mean(ep_lens), np.mean(ep_lens) / D.FPS))
    print()
    print("① 血量单调性      单帧最大回升 %.2e  （必须 <= 0）" % max_hp_jump)
    print("② 命中总数        %d 次，其中被格挡 %d 次 (%.0f%%)" %
          (n_hit, n_blk, 100 * n_blk / max(1, n_hit)))
    print("③ 招式使用        拳 %d 次 / 踢 %d 次  （比例 %.0f%% : %.0f%%）" %
          (mv_cnt[1], mv_cnt[2], 100 * mv_cnt[1] / max(1, n_hit), 100 * mv_cnt[2] / max(1, n_hit)))
    print("④ 射程违规        %d 次  （拳 %.2fm / 踢 %.2fm）" %
          (len(worst_reach), REACH[1], REACH[2]))
    print("⑤ KO 回合         %d/%d (%.0f%%)" % (ko, len(eps), 100 * ko / len(eps)))
    print("⑥ 伤害对账        CSV 反算 vs 采集记录，最大差额 %.2e  （必须为 0）" % max_bal_err)
    print()
    print("⑦ 动作帧占比：")
    tot = act_cnt.sum()
    for i, n in enumerate(D.ACTION_NAMES):
        p = 100 * act_cnt[i] / tot
        print("   %-6s %6.1f%%  %s" % (n, p, "█" * int(p / 2)))
    print()
    if bad:
        print("✗ 发现 %d 个问题，前 8 条：" % len(bad))
        for b in bad[:8]:
            print("   -", b)
    else:
        print("✓ 规则与数据完全一致，没有一条违规")


if __name__ == "__main__":
    main()
