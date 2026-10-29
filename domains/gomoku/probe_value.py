# -*- coding: utf-8 -*-
"""验收 value 头：给它已知胜负的局面，看它给不给对的胜率。

为什么要先验：训练日志里的「胜率 91.6%」可能只是标签不平衡的假象。
如果数据里 90% 是黑赢，模型只要学会「轮到黑就说赢」就能拿到 90%，
但它在真正需要判断的中盘均势上毫无区分度 —— 那样接进搜索只会添乱。

所以这里用三类局面直测它的分辨力：
  必胜（我方四连，下一手成五）
  必败（对方活四，堵不完）
  均势（开局几手）
"""
import os
import sys

import numpy as np
import torch

from gomoku import Board, CELLS, N
import match_rapfi as M


def val_of(b: Board, player: int) -> float:
    """value 头给出的「player 胜率」"""
    board = np.zeros(CELLS + 1, dtype=np.int8)
    board[:CELLS] = b.b
    bt = torch.from_numpy(board).view(1, -1)
    pt = torch.tensor([player], dtype=torch.long)
    with torch.no_grad():
        f = M._NET.body(M._NET._encode(bt, pt))
        v = M._NET.value(f)
    return float(torch.sigmoid(v)[0])


def build(seq):
    b = Board()
    for cell, p in seq:
        b.put(cell, p)
    return b


def main():
    net = sys.argv[1] if len(sys.argv) > 1 else "results/net_sl.pt"
    M.load_net(net)
    print(f"# 权重 {net}\n")

    cases = []

    # 1) 我方四连，轮到我 → 我必胜
    seq = [(7 * 15 + 4, 1), (0, 2), (7 * 15 + 5, 1), (1, 2), (7 * 15 + 6, 1), (2, 2)]
    b = build(seq)
    cases.append(("我四连·轮我(必胜)", b, 1, 2))

    # 2) 对方四连，轮到我 → 我必败（堵不完，但这里只有单四还能堵，故用双四）
    #    对方在 (7,4..7,7) 双端活四 --- 构造：白占据 (7,5)(7,6)(7,7)，两端 (7,4)(7,8) 都空
    seq = [(0, 1), (7 * 15 + 5, 2), (1, 1), (7 * 15 + 6, 2), (2, 1), (7 * 15 + 7, 2)]
    b = build(seq)
    cases.append(("敌活三·轮我(危险)", b, 1, 1))

    # 3) 对方活四，轮到我 → 我必败
    seq = [(0, 1), (7 * 15 + 4, 2), (1, 1), (7 * 15 + 5, 2), (2, 1), (7 * 15 + 6, 2), (14 * 15, 1), (7 * 15 + 7, 2)]
    b = build(seq)
    cases.append(("敌活四·轮我(必败)", b, 1, 0))

    # 4) 开局两手，均势
    seq = [(7 * 15 + 7, 1), (7 * 15 + 8, 2)]
    b = build(seq)
    cases.append(("开局两手(均势)", b, 1, 1))

    # 5) 我方双活三，攻势
    seq = [(7 * 15 + 7, 1), (0, 2), (7 * 15 + 8, 1), (1, 2), (6 * 15 + 7, 1), (2, 2), (8 * 15 + 7, 1)]
    b = build(seq)
    cases.append(("我双活三·轮我(优势)", b, 2, 2))

    print(f"{'局面':24s} {'我方胜率':>8s}  {'规则分':>8s}")
    print("-" * 52)
    import evolve
    for name, bd, me, _ in cases:
        v = val_of(bd, me)
        r = bd.evaluate(me, evolve.W0) if hasattr(evolve, "W0") else 0.0
        print(f"{name:24s} {v:8.3f}  {r:8.1f}")

    # 6) 同一局面换视角：黑白各看一次，两者应接近互补
    print("\n# 视角一致性（黑胜率 + 白胜率 应 ≈ 1）")
    seq = [(7 * 15 + 7, 1), (7 * 15 + 8, 2), (8 * 15 + 8, 1)]
    b = build(seq)
    vb = val_of(b, 1)
    vw = val_of(b, 2)
    print(f"  黑 {vb:.3f} / 白 {vw:.3f} → 和 {vb + vw:.3f}")


if __name__ == "__main__":
    main()
