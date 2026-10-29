"""单局诊断：打印每一步和棋盘，看清到底死在哪

不猜。只看。
"""
import sys
import numpy as np
from gomoku import Board, CELLS, N
import match_rapfi as M
from vcf import threats_at, danger_points

ENGINE = "rapfi/pbrain-rapfi-windows-avx2.exe"

SYM = " .XO"


def render(b: Board, marks=()):
    rows = []
    for r in range(N):
        line = ""
        for c in range(N):
            idx = r * N + c
            v = b.b[idx]
            ch = SYM[v + 1] if v else ("·" if idx in b.candidates() else " ")
            line += ch
        rows.append(f"{r:2d} {line}")
    return "\n".join(rows) + "\n    " + "".join(str(c % 10) for c in range(N))


def main():
    ck = M.load_net("results/net_sl.pt")
    print(f"# 网络 {ck and ck.get('acc')} ch={ck and ck.get('ch')}", flush=True)
    eng = M.PiskvorkEngine(ENGINE, movetime=300)
    b = Board()
    our = 1
    player = 1
    eng.start()
    for step in range(1, 60):
        if player == our:
            plan = None
            try:
                from vcf import find_vcf
                plan = find_vcf(b, player, depth=M.VCF_DEPTH, budget=M.VCF_BUDGET)
            except Exception as e:
                print("vcf err", e)
            hot = danger_points(b, 3 - player, topk=4, min_level=11)
            m = M.net_pick(b, player, None, 2, 10)
            tag = f"VCF杀{plan[:3] if plan else '-'} 敌双四{hot} 选中{m}"
        else:
            m = eng.begin() if b.moves == 0 else eng.turn(b.last)
            tag = f"Rapfi{m}"
        if b.b[m] != 0:
            print("非法落点", m)
            break
        won = b.put(m, player)
        print(f"\n第{step:2d}手 {'我' if player == our else '敌'} 落 {m//N},{m%N}  {tag}")
        print(render(b))
        if won:
            print(f"\n>>> {'我' if player == our else '敌'} 成五，共 {b.moves} 手")
            break
        player = 3 - player
    eng.end()


if __name__ == "__main__":
    main()
