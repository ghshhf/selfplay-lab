# -*- coding: utf-8 -*-
"""严格验收 value 头：只用中央区域的对称局面，排除角落分布外干扰。

判据：同一局面黑白各看一次，两个胜率之和应 ≈ 1（视角自洽）；
且「我方活三」应明显高于「对方活三」。
"""
import sys

import numpy as np
import torch

from gomoku import Board, CELLS
import match_rapfi as M


def val_of(b, player):
    board = np.zeros(CELLS + 1, dtype=np.int8)
    board[:CELLS] = b.b
    bt = torch.from_numpy(board).view(1, -1)
    pt = torch.tensor([player], dtype=torch.long)
    with torch.no_grad():
        f = M._NET.body(M._NET._encode(bt, pt))
        return float(torch.sigmoid(M._NET.value(f))[0])


def mk(moves):
    b = Board()
    for x, y, p in moves:
        b.put(y * 15 + x, p)
    return b


def main():
    net = sys.argv[1] if len(sys.argv) > 1 else "results/net_best.pt"
    M.load_net(net)
    print(f"# {net}\n")
    print(f"{'测试':22s} {'黑胜率':>7s} {'白胜率':>7s} {'和':>6s}  判定")
    print("-" * 62)

    tests = []

    # A 黑中央活三（横），轮白
    tests.append(("黑活三·轮白", [(6, 7, 1), (0, 0, 2), (7, 7, 1), (0, 1, 2), (8, 7, 1)]))

    # B 白中央活三（横），轮黑
    tests.append(("白活三·轮黑", [(0, 0, 1), (6, 7, 2), (0, 1, 1), (7, 7, 2), (0, 2, 1), (8, 7, 2)]))

    # C 黑双活三（横 + 竖），轮白 —— 黑大优
    tests.append(("黑双活三", [(6, 7, 1), (0, 0, 2), (7, 7, 1), (0, 1, 2), (8, 7, 1),
                              (0, 2, 2), (7, 6, 1), (1, 0, 2), (7, 5, 1)]))

    # D 均势：双方各两子分散
    tests.append(("均势", [(7, 7, 1), (7, 8, 2), (8, 8, 1), (6, 6, 2)]))

    # E 黑已成四（活四：6..9 中间缺一个），轮白 —— 黑必胜
    tests.append(("黑活四", [(6, 7, 1), (0, 0, 2), (7, 7, 1), (0, 1, 2), (8, 7, 1), (0, 2, 2), (9, 7, 1), (1, 0, 2)]))

    bad = 0
    for name, mv in tests:
        b = mk(mv)
        vb, vw = val_of(b, 1), val_of(b, 2)
        s = vb + vw
        ok = abs(s - 1.0) < 0.30
        if not ok:
            bad += 1
        print(f"{name:22s} {vb:7.3f} {vw:7.3f} {s:6.3f}  {'OK' if ok else '不自洽'}")

    print(f"\n黑活三 {val_of(mk(tests[0][1]),1):.3f}  白活三 {val_of(mk(tests[1][1]),1):.3f}"
          f"  → 应 前者 > 后者")
    print(f"不自洽 {bad}/{len(tests)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
