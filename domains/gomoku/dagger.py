"""DAgger：在我们自己会遇到的局面上，让 Rapfi 给标准答案

病根（实测出来的，不是猜的）
    训练 19 万条、验证准确率 73% 的网络，实战平均活 23.6 手；
    把网络整个换掉、退回最初那 7 个数字的规则权重，能活 25.3 手。
    —— 网络是负资产。

为什么
    教材全是「Rapfi 打 Rapfi」的局面。我们一旦走出 Rapfi 不会走的一步，
    后面所有局面都在教材之外，网络只能瞎猜；一个错误在 20 步里指数放大。
    这叫复合误差（compounding error），是模仿学习的经典死法。

解法
    让我们真的去走（产生我们自己的局面），每一步问 Rapfi「这局面你走哪」，
    把 (我们的局面, 他的答案) 存成教材。跑几轮，我们踩过的坑就被
    他自己的正确答案覆盖掉。
    学名 Dataset Aggregation（Ross & Bagnell, 2010）。

顺带
    board_query 还会带回引擎对这局面的评分（Eval），以后可以拿来训练胜率头，
    比「最终胜负」密得多 —— 每局 20 步就有 20 个信号，而不是 1 个。

用法
    python dagger.py --games 40 --movetime 300 --ask 500 --out results/dagger.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time

from gomoku import N, Board
from match_rapfi import PiskvorkEngine, load_net, net_pick


def play_and_ask(args, our: int = 1):
    """打一局，沿途记下「我方走之前」的局面，局后逐个问老师。

    引擎每局都重建：同一个进程连做多次 START 会失效（实测 stdin 直接断开），
    重建多花 1 秒加载 NNUE，但不会中途莫名其妙地死。

    返回 (记录列表, 胜者, 手数)
    """
    b = Board()
    seq: list[tuple[int, int, int]] = []
    marks: list[tuple] = []
    player = 1
    winner = 0
    eng = PiskvorkEngine(args.engine, movetime=args.movetime)
    eng.start()
    while b.moves < N * N:
        if player == our:
            marks.append((b.b.copy(), player, len(seq)))
            m = net_pick(b, player, None, args.depth, args.width)
        else:
            m = eng.begin() if b.moves == 0 else eng.turn(b.last)
        if not (0 <= m < N * N) or b.b[m] != 0:
            eng.end()
            eng.close()
            return [], -1, b.moves                     # 引擎犯规或我方越界
        seq.append((m % N, m // N, player))
        if b.put(m, player):
            winner = player
            eng.end()
            eng.close()
            break
        player = 3 - player
    else:
        eng.end()
        eng.close()

    recs = []
    teacher = PiskvorkEngine(args.engine, movetime=args.ask)
    teacher.start()
    for snap, pl, k in marks:
        if k == 0:                                        # 空盘没有信息量
            continue
        try:
            mv, ev = teacher.board_query(seq[:k])
        except Exception:
            continue
        if not (0 <= mv < N * N) or snap[mv] != 0:         # 老师给了非法/重复点
            continue
        recs.append({"board": snap.tolist(), "move": int(mv), "player": int(pl),
                     "no": k + 1, "winner": int(winner), "strength": 100,
                     "eval": float(ev)})
    teacher.close()
    return recs, winner, b.moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="rapfi/pbrain-rapfi-windows-avx2.exe")
    ap.add_argument("--net", default="results/net_sl.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--movetime", type=int, default=300, help="对局时 Rapfi 每手毫秒")
    ap.add_argument("--ask", type=int, default=500, help="问老师时每手毫秒")
    ap.add_argument("--out", default="results/dagger.jsonl")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=10)
    args = ap.parse_args()

    ck = load_net(args.net)
    print(f"# 学员 {args.net} (acc={ck and ck.get('acc')})", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    total, wins, strokes = 0, 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        for g in range(1, args.games + 1):
            recs, winner, n = play_and_ask(args)
            if winner == -1:
                print(f"  第 {g} 局异常，跳过", flush=True)
                continue
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            total += 1
            strokes += n
            if winner == 1:
                wins += 1
            print(f"  第 {g:3d} 局  共 {n:2d} 手  {'胜' if winner == 1 else '负'}  "
                  f"存 {len(recs):2d} 条  均 {strokes/total:.1f} 手  "
                  f"累计 {wins}胜{total-wins}负  {time.time()-t0:.0f}s", flush=True)

    print(f"# 完成 {total} 局 / {strokes} 手 / 均 {strokes/max(1,total):.1f} 手 "
          f"/ {wins} 胜 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
