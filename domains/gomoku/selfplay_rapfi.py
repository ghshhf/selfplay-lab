#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rapfi 自对弈：让世界第一跟它自己下，专门生产教材

为什么这么做
  我们现在只有 2000 来条教材，而 AlphaGo 监督学习用了 3000 万条。
  靠自己跟 Rapfi 打去攒，一局 5 秒才收 20 条，太慢。
  但 Rapfi 自己打自己，一局 4 秒能收 40 条 —— 而且是双方都是顶级着法，
  没有"我方乱走带偏局面"的污染，样本质量比我们的对局高得多。

产出
  results/rapfi_selfplay.jsonl   格式与 rapfi_moves.jsonl 完全一致，
  训练脚本直接一起吃。

用法
  python selfplay_rapfi.py --games 0 --movetime 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

from gomoku import N, Board
from match_rapfi import PiskvorkEngine


def play_one(exe: str, movetime: int, rec: list, sb: int = 100, sw: int = 100) -> int:
    """两个引擎互下一局，返回胜者（1 黑 / 2 白 / 0 平 / -1 作废）。rec 收集双方着法
    sb/sw = 黑方/白方的棋力上限（Rapfi INFO strength，100 为满）
    """
    eb = PiskvorkEngine(exe, movetime=movetime, strength=sb)
    ew = PiskvorkEngine(exe, movetime=movetime, strength=sw)
    try:
        eb.start()
        ew.start()
        b = Board()
        player = 1
        while b.moves < N * N:
            before = b.b.copy()
            if player == 1:
                m = eb.begin() if b.moves == 0 else eb.turn(b.last)
                ev = eb.last_eval
            else:
                m = ew.turn(b.last)
                ev = ew.last_eval
            if b.b[m] != 0:
                return -1                      # 引擎给了非法点，这局作废
            # eval 是引擎"走这步之前"对局面的评分，视角正是该引擎 = 落子方，
            # 和 winner 标签一样不用换算。它比"最终胜负"密 20 倍且带梯度，
            # 是胜率头唯一可靠的标签源（终局胜负在强弱悬殊局里是伪信号）。
            rec.append({"board": before, "move": int(m), "player": player,
                        "no": b.moves + 1, "eval": float(ev),
                        "strength": sb if player == 1 else sw})
            won = b.put(m, player)
            if won:
                return player
            player = 3 - player
        return 0
    finally:
        eb.close()
        ew.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=r"E:\selfevo\rapfi\pbrain-rapfi-windows-avx2.exe")
    ap.add_argument("--games", type=int, default=0, help="0 = 一直跑")
    ap.add_argument("--movetime", type=int, default=100, help="每手毫秒（100 够强，也够快）")
    ap.add_argument("--out", default="results/rapfi_selfplay.jsonl")
    ap.add_argument("--log", default="results/selfplay_rapfi.log")
    ap.add_argument("--jitter", type=int, default=40,
                    help="每局思考时间上下抖动毫秒，让局面自然分叉，避免每局一模一样")
    ap.add_argument("--strong", type=int, default=100, help="强方棋力上限 0-100")
    ap.add_argument("--weak", type=int, default=45,
                    help="弱方棋力上限 0-100。差距要小：弱方一点不赢，胜率标签就没信息")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log = open(args.log, "a", encoding="utf-8")

    def emit(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    emit(f"# Rapfi 自对弈启动 {time.strftime('%m-%d %H:%M:%S')}  "
         f"每手 {args.movetime}±{args.jitter}ms")

    fout = open(args.out, "a", encoding="utf-8")
    total = ok = bad = 0
    moves = 0
    t0 = time.time()

    g = 0
    while args.games == 0 or g < args.games:
        rec = []
        t = time.time()
        mt = max(30, args.movetime + random.randint(-args.jitter, args.jitter))
        # 强弱方轮流执黑：这样 winner 与先后手解耦，
        # 标签才反映"这方局面占优"，而不是"这方是黑棋"
        if g % 2 == 0:
            sb, sw = args.strong, args.weak
        else:
            sb, sw = args.weak, args.strong
        try:
            winner = play_one(args.engine, mt, rec, sb, sw)
        except Exception as e:
            emit(f"  第 {g+1} 局异常: {e}")
            bad += 1
            g += 1
            continue

        if winner < 0:
            bad += 1
        else:
            ok += 1
            for r in rec:
                # winner 是给胜率头用的标签：这局谁赢是确定的，不需要人来标
                fout.write(json.dumps({"board": [int(v) for v in r["board"]],
                                       "move": r["move"], "player": r["player"],
                                       "no": r["no"], "winner": winner,
                                       "eval": r.get("eval", 0.0),
                                       "strength": r.get("strength", 100)}) + "\n")
            fout.flush()
            moves += len(rec)

        total += 1
        g += 1
        if g % 10 == 0:
            el = time.time() - t0
            emit(f"  第 {g:4d} 局  有效 {ok} 废局 {bad}  教材 +{moves}  ✅ {moves/max(1,el)*60:.0f} 条/分钟"
                 f"  {el/60:.1f} 分钟  本局 {time.time()-t:.1f}s")

    emit(f"# 结束 共 {total} 局（有效 {ok} 废 {bad}）教材 +{moves} 条，用时 {(time.time()-t0)/60:.1f} 分钟")
    fout.close()
    log.close()


if __name__ == "__main__":
    main()
