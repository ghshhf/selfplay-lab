#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
和 Rapfi 对局（Piskvork 协议，stdin/stdout，无窗口）

用途
  1) 给自己定级：我们这套 7 维权重 + 深度2/宽10 的怪东西，对世界第一差多少
  2) 攒教材：把 Rapfi 每一手记下来，当监督学习的标签（学习他 → 再超越他）

用法
  python match_rapfi.py --engine rapfi/rapfi.exe --games 20 --movetime 1000
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np

from gomoku import N, CELLS, Board
from evolve import _pick, _mm
from gpu_selfplay import PolicyNet
from vcf import danger_points, find_vcf, threats_at

MOVE_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")
_EVAL_RE = re.compile(r"Eval\s+(-?\d+)")


class PiskvorkEngine:
    """Piskvork 协议客户端：START / BEGIN / TURN / BOARD / END
    关键点：INFO 是单向配置命令，引擎不回话 —— 只能发不能等
    """

    def __init__(self, exe: str, cwd: str | None = None, movetime: int = 1000,
                 memory_mb: int = 512, strength: int = 100):
        self.p = subprocess.Popen(
            [exe], cwd=cwd or os.path.dirname(os.path.abspath(exe)),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
            creationflags=0x08000000,          # CREATE_NO_WINDOW
        )
        self.movetime = movetime
        self.memory = memory_mb
        self.strength = max(0, min(100, int(strength)))
        self.last_eval = 0.0        # 最近一次 MESSAGE 里的 Eval（见 _read）
        self.q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.about = self._cmd("ABOUT")

    def _reader(self):
        """独立线程读，主线程用队列带超时取 —— 避免 readline 永久阻塞"""
        try:
            for line in self.p.stdout:
                self.q.put(line.rstrip("\r\n"))
        except Exception:
            pass
        self.q.put(None)                       # 进程结束标记

    # ---------- 底层 ----------
    def _send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def _read(self, timeout=30.0) -> str:
        """跳过 MESSAGE/DEBUG 噪声，返回真正有意义的应答

        顺手把 MESSAGE 里的 Eval 存进 self.last_eval —— 引擎对每个局面的
        评分是白送的训练信号（比"最终胜负"密 20 倍，且天然带梯度），
        不接下它就浪费了。
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("引擎已退出")
            line = line.strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith(("MESSAGE", "DEBUG", "INFO")):
                m = _EVAL_RE.search(line)
                if m:
                    self.last_eval = float(m.group(1))
                continue
            return line
        raise TimeoutError("引擎无响应")

    def _cmd(self, line: str, timeout=30.0) -> str:
        self._send(line)
        return self._read(timeout)

    # ---------- 协议 ----------
    def start(self):
        r = self._cmd(f"START {N}", timeout=20)
        if not r.upper().startswith("OK"):
            raise RuntimeError(f"START 失败: {r}")
        self._send(f"INFO timeout_turn {self.movetime}")      # 单向，不回话
        self._send("INFO timeout_match 900000")
        self._send(f"INFO max_memory {self.memory * 1024 * 1024}")
        self._send(f"INFO strength {self.strength}")          # 100=满强度，越低越让对方
        return r

    def begin(self) -> int:
        line = self._cmd("BEGIN", timeout=self.movetime / 1000 + 30)
        return self._parse_move(line)

    def turn(self, cell: int) -> int:
        x, y = cell % N, cell // N
        line = self._cmd(f"TURN {x},{y}", timeout=self.movetime / 1000 + 30)
        return self._parse_move(line)

    def board_query(self, positions: list[tuple[int, int, int]],
                    timeout: float = 0) -> tuple[int, float]:
        """把整个局面交给引擎，让它以「最后落子方的对手」身份走一手。

        这是 DAgger 的唯一入口 —— 能问出「换成你，这局面走哪」，
        顺带还能从 MESSAGE 里捡到它对这局面的评分（Eval）。

        有个坑：BOARD 后面不能跟「15,15,5」那行尺寸 —— 会被当成坐标，
        15 越界，回 "Coord is not valid or empty"。尺寸只由 START 决定。

        positions: [(x, y, who), ...]，必须按落子先后排列，who ∈ {1 黑, 2 白}
        返回 (落子格, 最后一个 Eval)
        """
        lines = ["BOARD"] + [f"{x},{y},{w}" for x, y, w in positions] + ["DONE"]
        self._send("\n".join(lines))
        lim = timeout or (self.movetime / 1000 + 30)
        t0 = time.time()
        ev = 0.0
        while time.time() - t0 < lim:
            try:
                line = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("引擎已退出")
            s = line.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith("ERROR"):
                raise RuntimeError(f"BOARD 被拒: {s}")
            if up.startswith(("MESSAGE", "DEBUG")):
                m = _EVAL_RE.search(s)
                if m:
                    ev = float(m.group(1))
                continue
            m = MOVE_RE.match(s)
            if m:
                return int(m.group(2)) * N + int(m.group(1)), ev
        raise TimeoutError("BOARD 无落子应答")

    def end(self):
        try:
            self._cmd("END")
        except Exception:
            pass

    def close(self):
        try:
            self._send("END")
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass

    @staticmethod
    def _parse_move(line: str) -> int:
        m = MOVE_RE.match(line)
        if not m:
            raise RuntimeError(f"看不懂引擎回话: {line!r}")
        x, y = int(m.group(1)), int(m.group(2))
        if not (0 <= x < N and 0 <= y < N):
            raise RuntimeError(f"坐标越界: {line!r}")
        return y * N + x


# ----------------------------------------------------------------------
# 神经网络走子（学 Rapfi 学出来的那套权重）
# ----------------------------------------------------------------------
_NET = None
_NET_CH = 48
# 是否用胜率头评估搜索叶子。False 则退回那 7 个手写数字（b.evaluate）。
#
# 🔴 2026-10-30 实测：默认必须是 False。
#   同一份权重、同一裁判各 30 局，只切换叶子评估：
#     胜率头  21.2 手   那 7 个手写数字  44.6 手
#   差 23 手 / 17 个标准误。原因不是训练不够，是信息类型不对 ——
#   胜率头回答"形势谁好"，答不了"还有几步死"；而五子棋的胜负
#   恰恰由"几步死"决定。手写数字粗糙，但它数的是威胁等级。
#   要重新启用，必须先证明它能分辨活四 / 活三 / 活二 的紧迫度差异。
_USE_VALUE = False

# 连续冲四搜索的预算：深度 10 层、总节点 2 万。
# 冲四是强制手，每层有效分支只有几个，所以 2 万节点通常几毫秒就跑完。
VCF_DEPTH = 10
VCF_BUDGET = 20000


def load_net(path: str):
    """载入 train_sl.py 存下的权重；坏文件返回 None，让调用方退回规则版"""
    global _NET, _NET_CH
    if not path or not os.path.exists(path):
        return None
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = PolicyNet(ck.get("ch", 48))
        net.load_state_dict(ck["net"], strict=False)   # 旧权重没有胜率头，忽略缺键
        net.eval()
        _NET, _NET_CH = net, ck.get("ch", 48)
        return ck
    except Exception as e:
        print(f"# 载入网络失败（{e}），退回规则版", flush=True)
        _NET = None
        return None


def _net_winrate(b: Board, me: int) -> float:
    """胜率头对当前局面给出 me 的胜率（0~1）"""
    import torch
    board = np.zeros(CELLS + 1, dtype=np.int8)
    board[:CELLS] = b.b
    bt = torch.from_numpy(board).view(1, -1)
    pt = torch.tensor([me], dtype=torch.long)
    with torch.no_grad():
        f = _NET.body(_NET._encode(bt, pt))
        return float(torch.sigmoid(_NET.value(f))[0])


def _net_winrate_batch(snaps: list, me: int) -> list:
    """一次前向评估多个局面。搜索每层要评 width 个叶子，
    逐个前向会把 CPU 吃满；拼成一个 batch 只花一次的开销。"""
    import torch
    if not snaps:
        return []
    arr = np.zeros((len(snaps), CELLS + 1), dtype=np.int8)
    for i, s in enumerate(snaps):
        arr[i, :CELLS] = s
    bt = torch.from_numpy(arr)
    pt = torch.full((len(snaps),), me, dtype=torch.long)
    with torch.no_grad():
        f = _NET.body(_NET._encode(bt, pt))
        return [float(v) for v in torch.sigmoid(_NET.value(f)).tolist()]


def _vmm(b: Board, player: int, root_me: int, depth: int,
         alpha: float, beta: float, width: int, w) -> float:
    """极小化极大，**叶子用胜率头评估**（返回值统一是 root_me 的胜率 0~1）

    与 _mm 的区别只有一处：叶子不再是那 7 个手写数字加权，
    而是"这个局面我大概几成胜率"。这是整个改造的目的 ——
    手写数字分不清"优势"和"劣势"，只会数棋型。

    内部节点仍用规则 rank 排序：那里的排序只影响剪枝效率，不影响正确性，
    调网络纯属浪费。
    """
    cands = b.candidates()
    if depth <= 0 or len(cands) == 0:
        return _net_winrate(b, root_me)

    top = b.rank(cands, player, w)[:width] if w is not None else cands[:width]

    if player == root_me:
        best = -1.0
        for c in top:
            c = int(c)
            if b.put(c, player):          # 走出去就成五，我方必胜
                b.undo()
                return 2.0 + depth        # 比任何胜率都大，且越浅越好（早赢）
            v = _vmm(b, 3 - player, root_me, depth - 1, alpha, beta, width, w)
            b.undo()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
    else:
        best = 2.0
        for c in top:
            c = int(c)
            if b.put(c, player):          # 对手走出去就成五，我方必败
                b.undo()
                return -1.0 - depth
            v = _vmm(b, 3 - player, root_me, depth - 1, alpha, beta, width, w)
            b.undo()
            if v < best:
                best = v
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best


def net_pick(b: Board, me: int, w=None, depth: int = 2, width: int = 10) -> int:
    """决策链：先算「必杀」，再算「必死」，剩下的才交给网络 + 搜索

    为什么是这个顺序：
      前四步都是规则层的硬判断——不猜、不学、不会因为训练不够而判错。
      网络再准也是概率，而「这里能成五」是 100% 的事实。
      优先级 = 确定性优先，概率垫底。

    改造前后最大的差别是第 3、4 步：
      原来只查「一手成五」，等于完全不会算杀也看不懂对手的杀。
      现在有连续冲四搜索，攻守两端都补上了。
    """
    import torch
    cands = b.candidates()
    if len(cands) == 1:
        return int(cands[0])
    opp = 3 - me

    for c in cands:                                  # 1) 我直接成五
        c = int(c)
        if b.put(c, me):
            b.undo()
            return c
        b.undo()

    for c in cands:                                  # 2) 对手能成五，必须堵
        c = int(c)
        if b.put(c, opp):
            b.undo()
            return c
        b.undo()

    # 3) 我方连续冲四能不能直接算死（原来是完全缺失的一环）
    plan = find_vcf(b, me, depth=VCF_DEPTH, budget=VCF_BUDGET)
    if plan:
        return int(plan[0])

    # 3.5) 对手如果轮到他走就能连续冲四杀我们 —— 占掉他这条线的起点。
    #      「只防成五指」是防不住连续冲四的：他的第一步往往只是个活三，
    #      我们不理，他冲四、我们堵、他再冲四，先手全在他手上，最后成五。
    opp_plan = find_vcf(b, opp, depth=VCF_DEPTH, budget=VCF_BUDGET)
    if opp_plan:
        return int(opp_plan[0])

    # 4) 对手的威胁点，按危险度：
    #    一子成双四/活四（≥11）—— 一旦让他下出来就堵不完了，必抢
    #    一子成单四（1）—— 说明他有三连。不抢，他下一手冲四，我们还是得堵，
    #      等于白送一手先手；所以也抢。这是 20 手速死的主要窟窿。
    hot = danger_points(b, opp, topk=4, min_level=11)
    if hot:
        return int(hot[0])
    warm = danger_points(b, opp, topk=2, min_level=1)
    if warm:
        return int(warm[0])

    if _NET is None:                                 # 没有网络就退回规则版
        return _pick(b, me, w, depth, width)

    board = np.zeros(CELLS + 1, dtype=np.int8)       # 5) 网络打分
    board[:CELLS] = b.b
    bt = torch.from_numpy(board).view(1, -1)
    pt = torch.tensor([me], dtype=torch.long)
    with torch.no_grad():
        lg = _NET.logits(bt, pt)[0]
    legal = torch.from_numpy((b.b == 0).astype(bool))
    lg = lg.masked_fill(~legal, -1e9)

    k = min(width, int(legal.sum()))
    top = [int(i) for i in torch.topk(lg, k).indices.tolist()]

    if w is None or depth <= 1:
        return top[0]

    # 6) 在网络的 top-K 里做 alpha-beta
    best_v, best_c = -1e30, top[0]
    alpha, beta = -1e30, 1e30
    for c in top:
        if b.put(c, me):                             # 落下去就五连
            b.undo()
            return c
        if _USE_VALUE:
            v = _vmm(b, opp, me, depth - 1, alpha, beta, width, w)
        else:
            v = _mm(b, opp, me, w, depth - 1, alpha, beta, width)
        b.undo()
        # 网络分的微小加成：同等局面下优先它觉得像人手的
        v += 1e-4 * float(lg[c])
        if v > best_v:
            best_v, best_c = v, c
        if v > alpha:
            alpha = v
        if alpha >= beta:
            break
    return best_c


def play_game(w: np.ndarray, engine: PiskvorkEngine, our_color: int,
              depth: int = 2, width: int = 10, record: list | None = None):
    """our_color: 1 黑 2 白。返回 (胜者, 步数, 棋盘)"""
    b = Board()
    engine.start()
    player = 1
    while b.moves < N * N:
        if player == our_color:
            m = (net_pick(b, player, w, depth, width) if _NET is not None
                 else _pick(b, player, w, depth, width))
        else:
            m = engine.begin() if b.moves == 0 else engine.turn(b.last)
        if b.b[m] != 0:
            engine.end()
            return -1, b.moves, b            # 引擎给了非法/重复落点，判它犯规
        before = b.b.copy()                     # 落子前的局面，才是训练样本
        won = b.put(m, player)
        if record is not None:
            record.append({"no": b.moves, "cell": int(m), "player": player,
                           "board": before})
        if won:
            engine.end()
            return player, b.moves, b
        player = 3 - player
    engine.end()
    return 0, b.moves, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--games", type=int, default=20, help="0 = 一直打下去")
    ap.add_argument("--movetime", type=int, default=1000, help="Rapfi 每手毫秒")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=10)
    ap.add_argument("--weights", default="results/champion.json")
    ap.add_argument("--net", default="results/net_sl.pt",
                    help="神经网络权重；不存在则退回 7 维规则版")
    ap.add_argument("--ladder", default="results/ladder.json",
                    help="梯子状态：没赢过就一直执黑，赢了才开始黑白轮换")
    ap.add_argument("--out", default="results/match_vs_rapfi.json")
    ap.add_argument("--log", default=None)
    ap.add_argument("--dump-moves", default="results/rapfi_moves.jsonl",
                    help="把 Rapfi 的每一手存下来，当监督学习标签")
    ap.add_argument("--leaf", default="rule", choices=["value", "rule"],
                    help="搜索叶子用什么评估：value=胜率头，rule=那 7 个手写数字。"
                         "留这个开关是为了能干净对比两者实战强弱")
    args = ap.parse_args()

    global _USE_VALUE
    _USE_VALUE = (args.leaf == "value")

    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        _lf = open(args.log, "a", encoding="utf-8")

        class _Tee:
            def write(self, s):
                sys.__stdout__.write(s)
                _lf.write(s)

            def flush(self):
                sys.__stdout__.flush()
                _lf.flush()

        sys.stdout = _Tee()

    w = np.array(json.load(open(args.weights))["w"])
    probe = PiskvorkEngine(args.engine, movetime=args.movetime)
    print(f"# Rapfi: {probe.about}", flush=True)
    probe.close()

    ck = load_net(args.net)
    mode = f"神经网络 net_sl.pt (ch={_NET_CH})" if ck else "7 维规则版（网络未就绪）"
    if ck:
        print(f"# 我方策略 {mode}  验证 {ck.get('acc', 0)*100:.1f}%  第 {ck.get('epoch', '?')} 代", flush=True)
    else:
        print(f"# 我方策略 {mode}  权重 {np.round(w, 2).tolist()}", flush=True)

    ladder = {"unlocked": False, "black_games": 0, "black_wins": 0, "unlock_game": None}
    if os.path.exists(args.ladder):
        try:
            ladder.update(json.load(open(args.ladder)))
        except Exception:
            pass
    state = "已解锁 → 黑白轮换" if ladder["unlocked"] else "锁定 → 每局执黑先手"
    print(f"# 梯子状态 {state}（先手 {ladder['black_games']} 局 / 胜 {ladder['black_wins']}）",
          flush=True)
    print(f"# 对局 {args.games or '无限'} 局，Rapfi 每手 {args.movetime} ms", flush=True)

    def save_ladder():
        os.makedirs(os.path.dirname(args.ladder) or ".", exist_ok=True)
        with open(args.ladder, "w", encoding="utf-8") as f:
            json.dump(ladder, f, ensure_ascii=False, indent=2)

    wins = losses = draws = 0
    t0 = time.time()
    dump = open(args.dump_moves, "a", encoding="utf-8") if args.dump_moves else None
    net_mtime = os.path.getmtime(args.net) if os.path.exists(args.net) else 0

    # 实战手数滑动窗口 —— 这才是真正的裁判
    # 验证准确率涨了可能只是背下了训练局面；能活多少手才反映真实棋力
    from collections import deque
    recent = deque(maxlen=20)
    arena_csv = "results/arena.csv"
    if not os.path.exists(arena_csv):
        with open(arena_csv, "w", encoding="utf-8") as f:
            f.write("time,game,avg_strokes20,last_strokes,wins,losses,net_acc\n")

    g = 0
    while args.games == 0 or g < args.games:
        # 权重重训过就换新的 —— 后台训练一存盘，这边下一局立刻用上
        if args.net and os.path.exists(args.net):
            m = os.path.getmtime(args.net)
            if m != net_mtime:
                ck2 = load_net(args.net)
                net_mtime = m
                if ck2:
                    print(f"# 换权重：验证 {ck2.get('acc', 0)*100:.1f}% "
                          f"第 {ck2.get('epoch','?')} 代", flush=True)

        our_color = 1 if (not ladder["unlocked"] or g % 2 == 0) else 2
        rec = []
        t = time.time()
        eng = PiskvorkEngine(args.engine, movetime=args.movetime)   # 每局全新进程
        try:
            winner, n_move, board = play_game(w, eng, our_color, args.depth, args.width, rec)
        except Exception as e:
            print(f"  第 {g+1} 局异常: {e}", flush=True)
            eng.close()
            continue
        eng.close()

        if winner == our_color:
            wins += 1
            res = "胜"
        elif winner == 0:
            draws += 1
            res = "和"
        else:
            losses += 1
            res = "负"
        side = "黑" if our_color == 1 else "白"

        if not ladder["unlocked"]:
            ladder["black_games"] += 1
            if winner == our_color:
                ladder["black_wins"] += 1
                ladder["unlocked"] = True
                ladder["unlock_game"] = g + 1
                print(f"  ★ 第 {g+1} 局首次执黑取胜（{n_move} 手）→ 解锁轮换", flush=True)
            save_ladder()

        recent.append(n_move)
        avg20 = sum(recent) / len(recent)
        print(f"  第 {g+1:3d} 局  我方执{side}  {res}  共{n_move:3d}手  {time.time()-t:5.1f}s"
              f"  累计 {wins}胜{losses}负{draws}和  近{len(recent)}局均 {avg20:.1f}手", flush=True)
        try:
            with open(arena_csv, "a", encoding="utf-8") as fa:
                acc_now = 0.0
                if _NET is not None:
                    try:
                        import torch as _t
                        ck3 = _t.load(args.net, map_location="cpu", weights_only=False)
                        acc_now = float(ck3.get("acc", 0))
                    except Exception:
                        pass
                fa.write(f"{time.strftime('%m-%d %H:%M:%S')},{g+1},{avg20:.2f},{n_move},"
                         f"{wins},{losses},{acc_now:.4f}\n")
        except Exception:
            pass

        # 存教材：只收 Rapfi 落的那几手，落子前局面 + 它选的格 = 一条监督样本
        if dump is not None:
            for r in rec:
                if r["player"] != our_color:
                    dump.write(json.dumps({
                        "board": [int(v) for v in r["board"]],
                        "move": r["cell"],
                        "player": r["player"],
                        "no": r["no"],
                    }) + "\n")
            dump.flush()

        json.dump({"wins": wins, "losses": losses, "draws": draws,
                   "score": (wins + 0.5 * draws) / max(1, wins + losses + draws),
                   "unlocked": ladder["unlocked"],
                   "movetime": args.movetime, "mode": "net" if _NET is not None else "rule"},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        g += 1

    total = max(1, wins + losses + draws)
    score = (wins + 0.5 * draws) / total
    print(f"# 结果 {wins}胜 {losses}负 {draws}和  得分率 {score*100:.1f}%  "
          f"耗时 {time.time()-t0:.0f}s", flush=True)
    if 0 < score < 1:
        import math
        gap = -400 * math.log10(1 / score - 1)
        print(f"# 与 Rapfi 差距约 {gap:.0f} Elo", flush=True)
    elif score == 0:
        print("# 全负 —— 差距超出可测范围（按 0 胜 N 负估，至少差 800+ Elo）", flush=True)

    if dump is not None:
        dump.close()


if __name__ == "__main__":
    main()
