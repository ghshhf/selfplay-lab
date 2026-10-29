#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五子棋 · GPU 批量自对弈（无窗口 / 无渲染 / 纯并行）

设计要点
  1) 状态全在显存：一次 B 盘同时推进，一步棋 = 一次散点写入 + 一次窗口查表
  2) 裁判 = 规则本身（连五查表），不可作弊，reward 可信 —— 这是能自进化的前提
  3) 策略 = 全卷积小网，REINFORCE 更新权重
  4) 采样阶段不留计算图（否则 225 步 × 大量卷积会爆显存），
     改为存 int8 局面快照 → 训练时随机抽样 → 重算 logprob 拿梯度
  5) GPU 只干一件事：把 B 盘前向压成一次 kernel，靠并行出吞吐
     —— 不画棋盘、不开窗口、不做渲染

用法
  python gpu_selfplay.py --boards 4096 --iters 2000
  python gpu_selfplay.py --boards 256 --iters 30 --device cpu     # 无卡自测
"""

import argparse
import copy
import csv
import os
import time
from collections import deque

import torch
import torch.nn as nn

W = 15
CELLS = W * W
SENTINEL = CELLS          # 越界占位，恒为空


# ----------------------------------------------------------------------
# 连五索引表：[225, 20, 5]
# 每格在 4 个方向上，所有能覆盖它的 5 连窗口；不足 20 条用 SENTINEL 补
# 判定：窗口内本方棋子数 == 5
# ----------------------------------------------------------------------
def build_win_index():
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    table = torch.full((CELLS, 20, 5), SENTINEL, dtype=torch.long)
    for r in range(W):
        for c in range(W):
            m = r * W + c
            w = 0
            for dr, dc in dirs:
                for k in range(5):
                    cells, ok = [], True
                    for t in range(5):
                        rr, cc = r + (t - k) * dr, c + (t - k) * dc
                        if not (0 <= rr < W and 0 <= cc < W):
                            ok = False
                            break
                        cells.append(rr * W + cc)
                    if ok:
                        table[m, w] = torch.tensor(cells)
                        w += 1
            assert w <= 20
    return table


WIN_IDX = build_win_index()


class PolicyNet(nn.Module):
    """全卷积，输出 225 格 logits；膨胀卷积撑开感受野覆盖全盘"""

    def __init__(self, ch=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(ch, 1, 1)
        # 胜率头：回答"当前落子方赢不赢"，输出 1 个 logit
        # 规模刻意与主体相当（约 6 万参数），不让它反压策略头
        self.value = nn.Sequential(
            nn.Conv2d(ch, 8, 1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * CELLS, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def _encode(self, board, player):
        """board [b,226] int8 ｜ player [b] → [b,3,15,15]
        转成 我方/对方/常量 三通道，消除先后手偏置
        """
        b = board[:, :CELLS].view(-1, W, W)
        p = player.view(-1, 1, 1)
        mine = (b == p).float()
        opp = ((b != 0) & (b != p)).float()
        ones = torch.ones_like(mine)
        return torch.stack([mine, opp, ones], dim=1)

    def logits(self, board, player):
        """board [b,226] int8 ｜ player [b] → [b,225]"""
        return self.head(self.body(self._encode(board, player))).view(-1, CELLS)

    def forward2(self, board, player):
        """一次前向同时给出（着法 logits, 胜率 logit），body 只算一次"""
        f = self.body(self._encode(board, player))
        return (self.head(f).view(-1, CELLS), self.value(f).view(-1))


# ----------------------------------------------------------------------
class BatchGomoku:
    def __init__(self, n, device, keep_snap=True):
        self.n = n
        self.dev = device
        self.win_idx = WIN_IDX.to(device)
        self.keep_snap = keep_snap
        self.reset()

    def reset(self):
        self.board = torch.zeros(self.n, CELLS + 1, dtype=torch.int8, device=self.dev)
        self.player = torch.ones(self.n, dtype=torch.long, device=self.dev)
        self.done = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        self.winner = torch.zeros(self.n, dtype=torch.long, device=self.dev)

    def legal(self, idx):
        return self.board[idx, :CELLS] == 0

    def apply(self, idx, move):
        p = self.player[idx]
        self.board[idx, move] = p.to(torch.int8)
        widx = self.win_idx[move]
        vals = self.board[idx].gather(1, widx.reshape(-1, 100)).view(-1, 20, 5)
        hit = (vals.to(torch.long) == p.view(-1, 1, 1)).sum(dim=2) == 5
        return hit.any(dim=1)

    # ---------------- 自我对弈 ----------------
    def selfplay(self, net, max_steps=CELLS):
        """跑完一轮全部盘，返回 (快照, 落子, 执子方, 掩码, 胜负)"""
        T, B = max_steps, self.n
        snap = torch.zeros(T, B, CELLS + 1, dtype=torch.int8, device=self.dev)
        moves = torch.zeros(T, B, dtype=torch.long, device=self.dev)
        plr = torch.zeros(T, B, dtype=torch.long, device=self.dev)
        mask = torch.zeros(T, B, dtype=torch.bool, device=self.dev)

        active = torch.arange(B, device=self.dev)
        t = 0
        while active.numel() > 0 and t < max_steps:
            b = self.board[active]
            p = self.player[active]
            lg = net.logits(b, p)
            lg = lg.masked_fill(~(b[:, :CELLS] == 0), -1e9)
            dist = torch.distributions.Categorical(logits=lg)
            mv = dist.sample()

            snap[t, active] = b
            moves[t, active] = mv
            plr[t, active] = p
            mask[t, active] = True

            won = self.apply(active, mv)
            if won.any():
                w = active[won]
                self.done[w] = True
                self.winner[w] = self.player[w]
            self.player[active] = 3 - self.player[active]

            full = self.board[active, :CELLS].ne(0).all(dim=1)
            if full.any():
                self.done[active[full & ~self.done[active]]] = True

            active = active[~self.done[active]]
            t += 1

        return snap[:t], moves[:t], plr[:t], mask[:t], self.winner.clone()


# ----------------------------------------------------------------------
@torch.no_grad()
def eval_vs_random(net, n, dev, max_steps=CELLS, temp_seed=None):
    """前一半盘拿黑，后一半拿白；对手走合法格随机"""
    net.eval()
    env = BatchGomoku(n, dev, keep_snap=False)
    net_side = torch.ones(n, dtype=torch.long, device=dev)
    net_side[n // 2:] = 2

    active = torch.arange(n, device=dev)
    t = 0
    while active.numel() > 0 and t < max_steps:
        b = env.board[active]
        p = env.player[active]
        legal = b[:, :CELLS] == 0

        lg = net.logits(b, p).masked_fill(~legal, -1e9)
        net_mv = lg.argmax(dim=1)
        rnd_mv = torch.distributions.Categorical(probs=legal.float()).sample()

        mv = torch.where(net_side[active] == p, net_mv, rnd_mv)

        won = env.apply(active, mv)
        if won.any():
            w = active[won]
            env.done[w] = True
            env.winner[w] = env.player[w]
        env.player[active] = 3 - env.player[active]

        full = env.board[active, :CELLS].ne(0).all(dim=1)
        if full.any():
            env.done[active[full & ~env.done[active]]] = True
        active = active[~env.done[active]]
        t += 1

    w = env.winner
    win = int((w == net_side).sum())
    lose = int(((w != 0) & (w != net_side)).sum())
    draw = int((w == 0).sum())
    net.train()
    return win, lose, draw


@torch.no_grad()
def eval_match(a, b, n, dev, max_steps=CELLS, seed=1234):
    """a 对 b：前一半 a 执黑，后一半 a 执白
    用固定种子采样落子 —— 贪心对贪心在两套几乎相同的权重之间只比谁执黑，测不出强弱
    """
    a.eval(); b.eval()
    gen = torch.Generator(device=dev).manual_seed(seed)
    env = BatchGomoku(n, dev, keep_snap=False)
    a_side = torch.ones(n, dtype=torch.long, device=dev)
    a_side[n // 2:] = 2

    active = torch.arange(n, device=dev)
    t = 0
    while active.numel() > 0 and t < max_steps:
        board = env.board[active]
        p = env.player[active]
        legal = board[:, :CELLS] == 0

        pa = torch.softmax(a.logits(board, p).masked_fill(~legal, -1e9), dim=1)
        pb = torch.softmax(b.logits(board, p).masked_fill(~legal, -1e9), dim=1)
        la = torch.multinomial(pa, 1, generator=gen).squeeze(1)
        lb = torch.multinomial(pb, 1, generator=gen).squeeze(1)
        mv = torch.where(a_side[active] == p, la, lb)

        won = env.apply(active, mv)
        if won.any():
            w = active[won]
            env.done[w] = True
            env.winner[w] = env.player[w]
        env.player[active] = 3 - env.player[active]
        full = env.board[active, :CELLS].ne(0).all(dim=1)
        if full.any():
            env.done[active[full & ~env.done[active]]] = True
        active = active[~env.done[active]]
        t += 1

    w = env.winner
    win = int((w == a_side).sum())
    lose = int(((w != 0) & (w != a_side)).sum())
    a.train(); b.train()
    return win, lose, int((w == 0).sum())


def policy_grad_step(net, opt, snap, moves, plr, mask, winner, samples, micro):
    """从快照里抽 samples 个 (步,盘) 记录，按胜负加权求策略梯度"""
    T, B = moves.shape
    snap_flat = snap.reshape(T * B, CELLS + 1)
    mv_flat = moves.reshape(-1)
    pl_flat = plr.reshape(-1)
    mk_flat = mask.reshape(-1)

    sign = torch.where(winner == 0, torch.zeros_like(winner),
                       torch.where(plr == winner.view(1, -1),
                                   torch.ones_like(plr), -torch.ones_like(plr)))
    sg_flat = sign.reshape(-1)

    avail = mk_flat.nonzero().flatten()
    take = min(samples, avail.numel())
    sel = avail[torch.randperm(avail.numel(), device=moves.device)[:take]]

    total_loss, seen = 0.0, 0
    for start in range(0, take, micro):
        part = sel[start:start + micro]
        b = snap_flat[part]
        p = pl_flat[part]
        lg = net.logits(b, p).masked_fill(~(b[:, :CELLS] == 0), -1e9)
        logp = torch.log_softmax(lg, dim=1).gather(1, mv_flat[part].unsqueeze(1)).squeeze(1)
        loss = -(logp * sg_flat[part]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        total_loss += float(loss.detach()) * part.numel()
        seen += part.numel()
    return total_loss / max(1, seen)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--samples", type=int, default=32768, help="每轮抽样训练的记录数")
    ap.add_argument("--micro", type=int, default=8192)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-boards", type=int, default=512)
    ap.add_argument("--gate", type=float, default=0.55, help="对擂主胜率超过它才换擂主")
    ap.add_argument("--save", default="results/gpu_policy.pt")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    net = PolicyNet(args.ch).to(dev)
    ref = copy.deepcopy(net).eval()          # 守擂基准，赢了才换
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    logpath = args.log or time.strftime("results/gpu_selfplay_%Y%m%d-%H%M%S.log")
    os.makedirs(os.path.dirname(logpath) or ".", exist_ok=True)
    csvpath = logpath.replace(".log", ".csv")

    def emit(s):
        print(s, flush=True)
        with open(logpath, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    emit("#" * 68)
    emit("# 五子棋 GPU 批量自对弈 · 无窗口 · 纯并行")
    emit(f"# 设备 {dev} ｜ 盘数 {args.boards} ｜ 迭代 {args.iters} ｜ 参数量 {nparam:,}")
    if dev.type == "cuda":
        emit(f"# 显卡 {torch.cuda.get_device_name(0)}")
    emit("#" * 68)

    with open(csvpath, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["iter", "games_per_sec", "loss", "vs_random", "vs_ref"])

    env = BatchGomoku(args.boards, dev)
    tail = deque(maxlen=10)
    t_all = time.time()

    for it in range(1, args.iters + 1):
        env.reset()
        t0 = time.time()
        snap, moves, plr, mask, winner = env.selfplay(net)
        t_play = time.time() - t0

        loss = policy_grad_step(net, opt, snap, moves, plr, mask, winner,
                                args.samples, args.micro)
        del snap, moves, plr, mask

        dt = time.time() - t0
        gps = args.boards / max(dt, 1e-6)
        tail.append(gps)

        if it % args.eval_every == 0 or it == 1:
            w, l, d = eval_vs_random(net, args.eval_boards, dev)
            rate = w / max(1, w + l + d)
            rw, rl, rd = eval_match(net, ref, args.eval_boards, dev)
            rrate = rw / max(1, rw + rl + rd)
            flag = ""
            if rrate >= args.gate:
                ref = copy.deepcopy(net).eval()
                flag = "  ★ 换擂主"
            emit(f"iter {it:5d}  采样 {t_play:5.2f}s  合计 {dt:5.2f}s  "
                 f"{gps:6.0f} 盘/秒  loss {loss:+.4f}  对随机 {rate*100:5.1f}%  "
                 f"对擂主 {rrate*100:5.1f}% ({rw}胜{rl}负{rd}平){flag}")
            with open(csvpath, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([it, f"{gps:.0f}", f"{loss:.4f}",
                                        f"{rate:.3f}", f"{rrate:.3f}"])
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            torch.save({"net": net.state_dict(), "ref": ref.state_dict(),
                        "iter": it, "boards": args.boards, "ch": args.ch}, args.save)

    emit(f"# 完成 {args.iters} 轮 ｜ 总耗时 {time.time()-t_all:.1f}s ｜ "
         f"平均 {sum(tail)/max(1,len(tail)):.0f} 盘/秒")
    emit(f"# 权重 → {args.save}")


if __name__ == "__main__":
    main()
