"""五子棋自进化：权重自己迭代，规则当铁面裁判

    python evolve.py --gens 40 --pop 20

它做的三件事：
  1. 自对弈 —— 不喂任何人类棋谱
  2. 自我迭代 —— 变异 + 锦标赛选育，一代比一代强（或者不，如实记录）
  3. 环境反馈 —— 输赢由规则判定，赖不掉

为什么这条能验证"自进化能不能走通"：
  每 3 代回头跟「初始权重」打一次对照。
  胜率从 50% 涨上去 = 真变强；原地不动 = 进化没效果，不粉饰。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

from gomoku import W_INIT, W_DIM, Board, clip_w


# ---------------------------------------------------------------- 对局
def _pick(b: Board, me: int, w: np.ndarray, depth: int, width: int) -> int:
    opp = 3 - me
    cands = b.candidates()
    if len(cands) == 1:
        return int(cands[0])

    ordered = b.rank(cands, me, w)
    top = ordered[:width]
    if depth <= 1:
        return int(top[0])

    best_v, best_c = -1e30, int(top[0])
    alpha, beta = -1e30, 1e30
    for c in top:
        c = int(c)
        if b.put(c, me):        # 直接成五
            b.undo()
            return c
        v = _mm(b, opp, me, w, depth - 1, alpha, beta, width)
        b.undo()
        if v > best_v:
            best_v, best_c = v, c
        if v > alpha:
            alpha = v
        if alpha >= beta:
            break
    return best_c


def _mm(b: Board, player: int, root_me: int, w: np.ndarray,
        depth: int, alpha: float, beta: float, width: int) -> float:
    """极小化极大，返回值统一是 root_me 视角。"""
    if depth <= 0:
        return b.evaluate(root_me, w)
    cands = b.candidates()
    if len(cands) == 0:
        return b.evaluate(root_me, w)

    top = b.rank(cands, player, w)[:width]

    if player == root_me:
        best = -1e30
        for c in top:
            c = int(c)
            if b.put(c, player):
                b.undo()
                return 1e15 + depth
            v = _mm(b, 3 - player, root_me, w, depth - 1, alpha, beta, width)
            b.undo()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
    else:
        best = 1e30
        for c in top:
            c = int(c)
            if b.put(c, player):
                b.undo()
                return -1e15 - depth
            v = _mm(b, 3 - player, root_me, w, depth - 1, alpha, beta, width)
            b.undo()
            if v < best:
                best = v
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best


def play_one(w_black, w_white, depth: int = 2, width: int = 10) -> tuple[int, int]:
    """打到底，不设任何限制。返回 (胜者 1/2/0, 步数)。"""
    b = Board()
    ws = {1: w_black, 2: w_white}
    player = 1
    while b.moves < 225:
        cands = b.candidates()
        if len(cands) == 0:
            return 0, b.moves
        m = _pick(b, player, ws[player], depth, width)
        if b.put(m, player):
            return player, b.moves + 1
        player = 3 - player
    return 0, b.moves


def score_of(w, opp, games: int, depth: int, width: int) -> float:
    """黑白各半，赢 1 / 和 0.5，返回 0~1。"""
    pts = 0.0
    for i in range(games):
        if i % 2 == 0:
            r, _ = play_one(w, opp, depth, width)
            pts += 1.0 if r == 1 else (0.5 if r == 0 else 0.0)
        else:
            r, _ = play_one(opp, w, depth, width)
            pts += 1.0 if r == 2 else (0.5 if r == 0 else 0.0)
    return pts / max(1, games)


def mutate(w: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    """各维按自身量级做相对扰动 —— 权重值域跨好几个数量级，不能用同一个 sigma。"""
    out = w.copy()
    for i in range(1, W_DIM):
        span = max(abs(w[i]), 1.0)
        out[i] = w[i] + rng.gauss(0.0, scale * span)
    return clip_w(out)


# ---------------------------------------------------------------- 并行
_G = {}


def _init_worker(depth, width):
    _G["depth"] = depth
    _G["width"] = width


def _job(payload):
    w, opp, games, tag = payload
    s = score_of(np.asarray(w), np.asarray(opp), games, _G["depth"], _G["width"])
    return s, tag


# ---------------------------------------------------------------- 主循环
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--games", type=int, default=6, help="粗筛：每个候选跟冠军打几局")
    ap.add_argument("--confirm", type=int, default=20, help="精筛：胜者跟冠军打几局定去留")
    ap.add_argument("--margin", type=float, default=0.6, help="精筛胜率超过多少才换代")
    ap.add_argument("--check", type=int, default=20, help="体检局数（跟初始权重）")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--workers", type=int, default=0, help="0 = 自动用满 CPU")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    nw = args.workers or max(1, (os.cpu_count() or 2) - 1)
    pool = mp.Pool(nw, initializer=_init_worker,
                   initargs=(args.depth, args.width)) if nw > 1 else None

    tag = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.outdir, f"evolve_{tag}.log")
    csv_path = os.path.join(args.outdir, f"games_{tag}.csv")
    champ_path = os.path.join(args.outdir, "champion.json")
    logf = open(log_path, "w", encoding="utf-8")

    def out(msg: str):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    if args.resume and os.path.exists(args.resume):
        champion = clip_w(np.array(json.load(open(args.resume))["w"]))
        out(f"# 从存档继续：{args.resume}")
    else:
        champion = clip_w(W_INIT.copy())

    baseline = clip_w(W_INIT.copy())
    best_ever = champion.copy()
    best_ever_v = 0.5
    hall: list[np.ndarray] = []

    out("#" * 66)
    out("# 五子棋自进化 · 规则当裁判 · 无窗口 · 大规模迭代")
    out(f"# 世代 {args.gens} · 种群 {args.pop} · 粗筛 {args.games} 局 "
        f"· 精筛 {args.confirm} 局（门槛 {args.margin}）")
    out(f"# 深度 {args.depth} · 宽 {args.width} · 并行 {nw} · 体检 {args.check} 局")
    out("#" * 66)
    out(f"# 初始权重 {np.round(champion, 3).tolist()}")

    def evals(jobs):
        """jobs = [(w, opp, games, tag)] -> [(score, tag)]"""
        if pool is None:
            return [(score_of(w, o, g, args.depth, args.width), t)
                    for w, o, g, t in jobs]
        return pool.map(_job, [(w.tolist(), o.tolist(), g, t)
                               for w, o, g, t in jobs])

    t_all = time.time()
    hist = []

    for gen in range(1, args.gens + 1):
        t0 = time.time()
        scale = max(0.02, 0.35 * (0.90 ** (gen - 1)))

        cands = [champion.copy()] + [mutate(champion, rng, scale) for _ in range(args.pop - 1)]
        # 所有变异体都直接跟现任冠军打 —— 排名靠前的才真是"能赢最强者"
        jobs = [(cands[i], champion, args.games, i) for i in range(1, args.pop)]
        res = evals(jobs)
        res.sort(key=lambda x: -x[0])

        # 粗筛前两名都进精筛：粗筛只有几局，噪声足以把真强者挤掉
        finalists = [cands[i] for _, i in res[:2]]
        defend, best_w = -1.0, champion
        for fc in finalists:
            d = score_of(fc, champion, args.confirm, args.depth, args.width)
            if d > defend:
                defend, best_w = d, fc

        if defend >= args.margin:
            hall.append(champion)
            champion = best_w
            verdict = "★ 换代"
        else:
            verdict = "- 守擂"

        dt = time.time() - t0
        out(f"gen {gen:3d}  {verdict}  精筛 {defend:.2f}（{args.confirm}局）  "
            f"粗筛最高 {res[0][0]:.2f}  {dt:.1f}s")

        # 体检 + 防退化：跟初始权重的胜率是唯一绝对标尺
        if gen % 5 == 0 or gen == args.gens:
            v = score_of(champion, baseline, args.check, args.depth, args.width)
            hist.append((gen, v))
            note = ""
            if v > best_ever_v + 0.02:
                best_ever_v, best_ever = v, champion.copy()
                note = "  ← 新历史最优"
            elif v < best_ever_v - 0.12:
                champion = best_ever.copy()
                note = "  ← 退化，回退到历史最优"
            out(f"         └ 体检 {v:.0%}（{args.check}局）历史最优 {best_ever_v:.0%}{note}")
            out(f"            权重 {np.round(champion, 2).tolist()}")

    out("")
    out("#" * 66)
    if hall:
        out(f"# 擂台上还剩 {len(hall)} 个历史冠军，冠军逐个复盘：")
        for i, h in enumerate(hall[-5:]):
            out(f"   vs 历史冠军{i+1} -> {score_of(champion, h, 4, args.depth, args.width):.0%}")
    final = score_of(champion, baseline, max(24, args.check), args.depth, args.width)
    out(f"# 终检 冠军 vs 初始权重 = {final:.0%}（{max(24, args.check)} 局）")
    out(f"# 历史最优 {best_ever_v:.0%}")
    out(f"# 换代 {len(hall)} 次 / {args.gens} 代")
    out(f"# 总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    out(f"# 冠军权重 {np.round(champion, 4).tolist()}")

    json.dump({"w": champion.tolist(), "vs_init": final, "log": log_path},
              open(champ_path, "w"), indent=2)

    # 进化曲线旁存一份，方便画图
    with open(os.path.join(args.outdir, f"curve_{tag}.csv"), "w", newline="") as f:
        w2 = csv.writer(f)
        w2.writerow(["gen", "winrate_vs_init"])
        for g, v in hist:
            w2.writerow([g, v])

    if pool is not None:
        pool.close()
    out(f"# 已存 {champ_path}")
    logf.close()


if __name__ == "__main__":
    main()
