#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU 监督学习：把 Rapfi 的着法当老师，训练 PolicyNet 更新权重

思路
  教材 = results/rapfi_moves.jsonl  (局面 → Rapfi 落的那一格)
  目标 = 交叉熵最小化，即"看到这种局面，我也下同一格"
  增强 = 棋盘 8 重对称变换，样本量直接 ×8（902 条 → 7216 条）

为什么是 CPU
  网络只有 112 万参数、样本几千条，GPU 的传输开销比计算还大；
  真正该吃 GPU 的是"几千盘同时自对弈"，不是训练这个尺寸的网。

用法
  python train_sl.py --data results/rapfi_moves.jsonl --epochs 60 --threads 4
  → 产出 results/net_sl.pt（match_rapfi.py 用 --net 载入）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

# Eval → 胜率的映射尺度。实测 Rapfi 的 Eval 分布：均势 0~500、明显优劣 1000~1300，
# 单局内标准差约 500。取 1000 使 500 分 ≈ 0.73、1000 分 ≈ 0.88、2000 分 ≈ 0.98。
# 之前取 2000 是错的：它把整批标签压进 0.5±0.12，梯度几乎为零，
# 值头直接退化成"永远输出 0.55"的常数预测（值损 80 代固定在 ln2≈0.685）。
EVAL_SCALE = 1000.0

from gpu_selfplay import CELLS, W, PolicyNet

# ----------------------------------------------------------------------
# 8 重对称：坐标变换 → 新棋盘第 (nr,nc) 格取自旧棋盘 (r,c) 格
# ----------------------------------------------------------------------
def _transforms():
    out = []
    for rot in range(4):
        for flip in (False, True):
            def f(r, c, rot=rot, flip=flip):
                if flip:
                    c = W - 1 - c
                for _ in range(rot):
                    r, c = c, W - 1 - r
                return r, c
            # src[新格] = 旧格
            src = np.empty(CELLS, dtype=np.int64)
            for r in range(W):
                for c in range(W):
                    nr, nc = f(r, c)
                    src[nr * W + nc] = r * W + c
            # 同时给出正向映射，用来搬走子标签
            fwd = np.empty(CELLS, dtype=np.int64)
            for r in range(W):
                for c in range(W):
                    nr, nc = f(r, c)
                    fwd[r * W + c] = nr * W + nc
            out.append((src, fwd))
    return out


TRANS = _transforms()


def load_data(path):
    """path 可以是单个文件，也可以是逗号分隔的多个（我们的对局 + Rapfi 自对弈）

    每项后面可以跟 `*倍数` 来抬高该数据源的采样权重，例如
        results/rapfi_selfplay.jsonl,results/dagger_1.jsonl*12
    —— 必要之举：自对弈有 19 万条，DAgger 纠错样本只有几千条，
    不加权就被淹掉，等于没加。倍数靠复制样本实现，代价只是内存。
    """
    specs = [p.strip() for p in path.split(",") if p.strip()]
    boards, moves, players, values, games = [], [], [], [], []
    pols = []
    bad = 0
    gid = 0
    prev_no = -1
    for spec in specs:
        mult = 1
        if "*" in spec:
            spec, m = spec.rsplit("*", 1)
            try:
                mult = max(1, int(m))
            except ValueError:
                mult = 1
        fp = spec
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                mv = d.get("move", None)
                bd = d.get("board", None)
                pl = d.get("player", None)
                if mv is None or bd is None or pl is None:
                    bad += 1
                    continue
                if len(bd) != CELLS or not (0 <= int(mv) < CELLS):
                    bad += 1
                    continue
                if not any(bd):
                    gid += 1             # 空盘 = 新的一局。必须按局分组，
                                         # 否则同局的两个局面会同时落进训练和验证 → 标签泄漏
                elif "eval" in d and int(d.get("no", 0)) <= prev_no:
                    gid += 1             # DAgger 样本里没有空盘（故意跳过），
                                         # 手数往回跳就说明换了一局
                prev_no = int(d.get("no", prev_no))
                # 胜率标签：始终相对"落子方"。这局谁赢是裁判定的，不用人标
                if "eval" in d:
                    # DAgger 样本带 Rapfi 的 Eval —— 这是当前局面的连续评估，
                    # 比"最终胜负"密 20 倍，而且天然带梯度。
                    # 为什么必须换掉硬标签：硬标签（0/1）只有两种值，网络会一路
                    # 推到饱和 —— 实测它把均势局面判成 0.11/0.99，搜索拿它当叶子
                    # 评估等于没有分辨力。软标签保住"差多少"这个信息。
                    ev = float(d["eval"])
                    if ev == 0.0 and any(bd):
                        # 引擎在必胜/必败局面会秒回、不给分，方向未知 → 不教胜率
                        val = -1.0
                    else:
                        # Eval 是"当前落子方视角"，正中 0.5，±EVAL_SCALE 走完 tanh
                        val = 0.5 + 0.5 * math.tanh(ev / EVAL_SCALE)
                else:
                    wn = d.get("winner", None)
                    if wn in (1, 2):
                        val = 1.0 if int(pl) == int(wn) else 0.0
                    else:
                        val = -1.0       # 旧数据没标签，只算着法损失
                # 棋力被削弱的着法是垃圾，不能拿去学着法；但它的胜负标签仍然有效
                pl_ok = 1 if int(d.get("strength", 100)) >= 60 else 0
                for _ in range(mult):
                    boards.append(bd)
                    moves.append(int(mv))
                    players.append(int(pl))
                    games.append(gid)
                    values.append(val)
                    pols.append(pl_ok)
    x = np.asarray(boards, dtype=np.int8)
    y = np.asarray(moves, dtype=np.int64)
    p = np.asarray(players, dtype=np.int64)
    v = np.asarray(values, dtype=np.float32)
    g = np.asarray(games, dtype=np.int32)
    pol = np.asarray(pols, dtype=np.int8)
    return x, y, p, v, g, pol, bad


def augment(x, y, power=1):
    """每个样本随机取一种对称（比全量 8 倍省内存，效果等同随机增强）"""
    n = x.shape[0]
    out_x = x.copy()
    out_y = y.copy()
    for i in range(n):
        src, fwd = TRANS[np.random.randint(len(TRANS))]
        out_x[i] = x[i][src]
        out_y[i] = fwd[y[i]]
    return out_x, out_y


def to_tensor(x_np, y_np, p_np, dev):
    """board 要补到 226 列（第 226 位是哨兵，PolicyNet 只取前 225）"""
    b = torch.zeros(len(x_np), CELLS + 1, dtype=torch.int8)
    b[:, :CELLS] = torch.from_numpy(x_np)
    return b.to(dev), torch.from_numpy(y_np).to(dev), torch.from_numpy(p_np).to(dev)


def accuracy(net, x, y, p, dev, bs=2048):
    """返回 (top1, top5)。top5 更平滑，样本少的时候比 top1 可靠"""
    net.eval()
    h1 = h5 = tot = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            b = torch.zeros(len(x[i:i + bs]), CELLS + 1, dtype=torch.int8)
            b[:, :CELLS] = torch.from_numpy(x[i:i + bs])
            b = b.to(dev)
            pl = torch.from_numpy(p[i:i + bs]).to(dev)
            tgt = torch.from_numpy(y[i:i + bs]).to(dev)
            legal = b[:, :CELLS] == 0
            legal[torch.arange(legal.shape[0]), tgt] = True      # 标签格也算合法
            lg = net.logits(b, pl).masked_fill(~legal, -1e9)
            h1 += int((lg.argmax(dim=1) == tgt).sum())
            k = min(5, lg.shape[1])
            h5 += int((torch.topk(lg, k, dim=1).indices == tgt.view(-1, 1)).any(dim=1).sum())
            tot += len(tgt)
    return h1 / max(1, tot), h5 / max(1, tot)


def value_acc(net, x, y, p, v, dev, bs=2048):
    """胜率头的方向准确率（只算有标签的样本）。
    随机基线 = 50%，持续高于 55% 才说明它真的看懂了局面好坏。
    """
    net.eval()
    hit = tot = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            vb = v[i:i + bs]
            m = torch.from_numpy((vb >= 0)).to(dev)
            if not bool(m.any()):
                continue
            b = torch.zeros(len(x[i:i + bs]), CELLS + 1, dtype=torch.int8)
            b[:, :CELLS] = torch.from_numpy(x[i:i + bs])
            b = b.to(dev)
            pl = torch.from_numpy(p[i:i + bs]).to(dev)
            vt = torch.from_numpy(vb).to(dev)
            _, vl = net.forward2(b, pl)
            pred = (vl > 0).long()
            # 标签可能是 0/1 硬标签，也可能是来自 Eval 的软标签 → 统一比方向
            gt = (vt > 0.5).long()
            hit += int((pred[m] == gt[m]).sum())
            tot += int(m.sum())
    return hit / max(1, tot), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/rapfi_moves.jsonl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4, help="CPU 线程（别吃满，留给进化）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="训练设备：auto 有卡就用 CUDA")
    ap.add_argument("--val", type=float, default=0.12)
    ap.add_argument("--vw", type=float, default=0.5, help="胜率损失的权重（相对着法损失）")
    ap.add_argument("--out", default="results/net_sl.pt")
    ap.add_argument("--init", default=None,
                    help="从已有权重续训（教材增加了就接着练，不从头来）")
    ap.add_argument("--log", default=None)
    ap.add_argument("--vonly", action="store_true",
                    help="只训胜率头：冻结 body 与着法头，别的权重一个字节都不动。"
                         "用来单独升级胜率头而不破坏已经打出手数的策略头")
    ap.add_argument("--minutes", type=float, default=0,
                    help="跑满这么多分钟就停（0 = 按 epochs）")
    args = ap.parse_args()

    # emit / 日志句柄先就位，后面的设备信息才打得出
    logf = open(args.log, "a", encoding="utf-8") if args.log else None

    def emit(s):
        print(s, flush=True)
        if logf:
            logf.write(s + "\n")
            logf.flush()

    torch.set_num_threads(max(1, args.threads))
    # 训练是"几千样本同时算"的并行活，正好是显卡的形状；下棋那步才是串行的。
    # 所以造数据/对战留 CPU，训练走 GPU。
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        emit("# 想要 CUDA 但不可用，回落 CPU")
        dev = torch.device("cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        emit(f"# 设备 CUDA · {torch.cuda.get_device_name(0)}")
    else:
        emit(f"# 设备 CPU · {args.threads} 线程")
    np.random.seed(int(time.time()) & 0xFFFF)

    x, y, p, v, g, pol, bad = load_data(args.data)
    if len(x) < 50:
        emit(f"# 教材太少（{len(x)} 条），先多打几局再来")
        return
    nlab = int((v >= 0).sum())
    npol = int(pol.sum())
    ngames = int(len(np.unique(g)))
    emit(f"# 教材 {len(x)} 条 / {ngames} 局（跳过 {bad} 条脏数据）｜"
         f"着法可用 {npol} 条｜带胜负标签 {nlab} 条"
         f"（{nlab/max(1,len(x))*100:.1f}%）  线程 {args.threads}")

    # 按"局"划分，不按行 —— 同一局的局面必须在同一侧，否则胜率头会背答案
    gids = np.unique(g)
    rng = np.random.permutation(len(gids))
    nvalg = max(1, int(len(gids) * args.val))
    pick = np.zeros(len(gids), dtype=bool)
    pick[rng[:nvalg]] = True
    vmask = pick[np.searchsorted(gids, g)]
    vi = np.where(vmask)[0]
    ti = np.where(~vmask)[0]
    xv, yv, pv, vv = x[vi], y[vi], p[vi], v[vi]
    xt, yt, pt, vt = x[ti], y[ti], p[ti], v[ti]
    polt = pol[ti]
    polv = pol[vi] == 1
    emit(f"# 训练 {len(xt)} 条 / 验证 {len(xv)} 条（按局切分，验证 {nvalg} 局）；"
         f"每代 8 重对称随机增强")

    net = PolicyNet(args.ch).to(dev)
    warm = 0.0
    if args.init and os.path.exists(args.init):
        try:
            ck0 = torch.load(args.init, map_location="cpu", weights_only=False)
            if ck0.get("ch", 48) == args.ch:
                # strict=False：旧权重没有胜率头，那些键让它随机初始化
                miss = net.load_state_dict(ck0["net"], strict=False)
                if miss.missing_keys:
                    emit(f"# 胜率头 {len(miss.missing_keys)} 个键从随机初始化开始")
                warm = float(ck0.get("acc", 0))
                emit(f"# 续训自 {args.init}（上一轮验证 {warm*100:.1f}%，第 {ck0.get('epoch','?')} 代）")
            else:
                emit(f"# {args.init} 的通道数 {ck0.get('ch')} 与 --ch {args.ch} 不符，从头训")
        except Exception as e:
            emit(f"# 续训载入失败（{e}），从头训")
    nparam = sum(q.numel() for q in net.parameters())
    emit(f"# 网络 {nparam} 参数  ch={args.ch}")

    if args.vonly:
        # 只放开胜率头。body 冻结后 forward2 里策略分支的梯度自然为 0，
        # 但显式冻结更保险 —— 绝不能因为一次误更新把 47.5 手的策略头弄坏。
        for q in net.body.parameters():
            q.requires_grad_(False)
        for q in net.head.parameters():
            q.requires_grad_(False)
        trainable = [q for q in net.parameters() if q.requires_grad]
        emit(f"# 只训胜率头：可训练参数 {sum(q.numel() for q in trainable)} / {nparam}")
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # 🔴 不再拿历史准确率当门槛：教材每轮都在涨（4k → 36k），验证集难度也跟着变，
    #    两个百分比根本不可比。旧的 `warm - 0.08` 门槛把训练锁死了整整 7 轮（差 0.02 个点）。
    #    现在只在本轮内挑最高，落盘交给实战手数去裁判。
    best, best_acc5, best_state, best_ep, best_vacc = -1.0, 0.0, None, 0, 0.0
    a1, a5 = accuracy(net, xv, yv, pv, dev)
    emit(f"# 初始验证 top1 {a1*100:.1f}%  top5 {a5*100:.1f}%（随机 0.44% / 2.2%）")

    t0 = time.time()
    step_total = 0
    for ep in range(1, args.epochs + 1):
        net.train()
        xa, ya = augment(xt, yt)
        perm = np.random.permutation(len(xa))
        loss_sum, nb = 0.0, 0
        vloss_sum, vnb = 0.0, 0
        for i in range(0, len(perm), args.bs):
            sel = perm[i:i + args.bs]
            if len(sel) < 4:
                continue
            b, tgt, pl = to_tensor(xa[sel], ya[sel], pt[sel], dev)
            pm = torch.from_numpy(polt[sel]).to(dev).bool()
            if not bool(pm.any()):
                continue                            # 这一批全是弱方垃圾着法，跳过
            legal = b[:, :CELLS] == 0
            lg, vl = net.forward2(b, pl)          # body 只算一次，两个头共用
            lg = lg.masked_fill(~legal, -1e9)
            lp = nn.functional.cross_entropy(lg[pm], tgt[pm])
            vlab = torch.from_numpy(vt[sel]).to(dev)
            msk = vlab >= 0                        # 没有胜负标签的样本不参与胜率损失
            if bool(msk.any()):
                lv = nn.functional.binary_cross_entropy_with_logits(
                    vl[msk], vlab[msk].float())
                # vonly 时不掺着法损失：那部分对冻结参数没有梯度，
                # 掺进来只会让"本轮最好"的判据被无关数字带偏
                loss = (args.vw * lv) if args.vonly else (lp + args.vw * lv)
                vloss_sum += float(lv.detach())
                vnb += 1
            else:
                if args.vonly:
                    continue          # 这批没有胜率标签，vonly 下无事可做；
                                      # 不能退回 lp —— 策略头已冻结，反传会报无梯度
                loss = lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            loss_sum += float(lp.detach())
            nb += 1
            step_total += 1
        sched.step()

        acc, acc5 = accuracy(net, xv[polv], yv[polv], pv[polv], dev)
        vacc, vn = value_acc(net, xv, yv, pv, vv, dev)
        dt = time.time() - t0
        emit(f"ep {ep:3d}  着法损 {loss_sum/max(1,nb):.4f}  验证 {acc*100:5.1f}%/top5 {acc5*100:5.1f}%"
             f"  胜率 {vacc*100:5.1f}%(标{vn})  值损 {vloss_sum/max(1,vnb):.4f}"
             f"  {dt:6.1f}s  ({step_total} 步)")
        # 挑"本轮最好"的判据跟着目标走：vonly 看胜率方向准确率，否则看着法准确率
        key = vacc if args.vonly else acc
        if key > best:
            best, best_acc5, best_ep, best_vacc = key, acc5, ep, vacc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            emit(f"        └ 本轮新高 {acc*100:.1f}%（留在内存，本轮结束统一落盘）")

        if args.minutes and dt > args.minutes * 60:
            emit(f"# 跑满 {args.minutes} 分钟，提前结束")
            break

    # ---------- 本轮结束：无条件落盘（同时留一份 .prev 备份，手数掉了能回滚）----------
    if best_state is not None:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        if os.path.exists(args.out):
            try:
                import shutil
                shutil.copy2(args.out, args.out + ".prev")
            except Exception:
                pass
        tmp = args.out + ".tmp"
        torch.save({"net": best_state, "ch": args.ch, "acc": best, "acc5": best_acc5,
                    "vacc": best_vacc,
                    "epoch": best_ep, "samples": int(len(x)),
                    "data": os.path.basename(str(args.data))}, tmp)
        os.replace(tmp, args.out)
        emit(f"# 落盘 {args.out}（本轮最好 {best*100:.1f}%@ep{best_ep}，样本 {len(x)}，"
             f"上一版备份为 .prev）")

    emit(f"# 训练结束，本轮最好验证准确率 {best*100:.1f}%，用时 {time.time()-t0:.0f}s")
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
