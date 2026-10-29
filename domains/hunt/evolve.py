# -*- coding: utf-8 -*-
"""hunt 域：两个种群的对抗共进化 + 信念质量追踪。

与既有域的三点不同
------------------
1. **两个独立种群**（猎手 / 逃犯），不是同一种群自战。
   这样两侧各走各的，天然形成军备竞赛 —— 也顺带治了 _mini_duel 那种
   「全种群塌成一个流派」的毛病：一方变强，另一方必须跟着变，否则被吃掉。

2. **除了胜率，还记信念质量**。这是本域存在的理由：
      belief_entropy  信念熵，有没有收敛到一个判断
      belief_top1     信念峰值是否真的指向逃犯（「有判断」≠「判断对」）
      belief_nll      真值处的负对数似然（连续版命中率）
   只有 hunter 种群真正学会了建模对手，这三个数才会一起变好。

3. **完全信息对照组**：每隔若干代，让同一批冠军在 visible 模式（直接拿到
   真实坐标）下重跑。完全信息胜率 − 不完全信息胜率 = 不完全信息的成本。
   有了这个差额，才能判断 hunter 的进步究竟来自「推断变准」还是
   「本来azio就很强」。

随机基线（有意为之地压到 50% 附近）
----------------------------------
图规模 16、线索间隔 4、线索噪声 0.8、回合上限 15 → 实测 0.515。
再放宽回合数就变成「图上两个随机游走必然相遇」，逃犯再聪明也活不下来，
只剩猎手单边变强，军备竞赛不起来。
"""
import argparse
import json
import os
import random
import time

import numpy as np

from env import (BOUNDS_E, BOUNDS_H, GENE_E, GENE_H, EvaderBot, HuntEnv,
                 HunterBot)

DIMENSIONS = ["nodes", "hidden", "press", "machines"]
DIM_LABEL = {"nodes": "图规模", "hidden": "线索稀疏", "press": "回合压迫",
             "machines": "通电门槛"}
MAX_LEVEL = 3


def rng_from():
    return np.random.default_rng()


def denorm(g, bounds):
    """把 [0,1]^d 的基因型映射到各维的取值范围。"""
    return [bounds[k][0] + g[k] * (bounds[k][1] - bounds[k][0])
            for k in range(len(g))]


def random_gene(rng, d):
    return list(rng.random(d))


def diversity(pop):
    """种群多样性：各基因维标准差的均值（归一化到 [0,1] 空间）。"""
    a = np.array(pop)
    return float(np.mean(np.std(a, axis=0)))


# ---------------------------------------------------------------------------
# 世界难度
# ---------------------------------------------------------------------------
class World:
    def __init__(self):
        # 起步参数全部实测校准过，不靠直觉。修机-开门版的标定过程：
        #   先试 16 节点 / 线索每 4 回合 / 猎手全速 → 猎手 0.948，逃犯被压死；
        #   加 hunter_slow 与稀线索两个方向各扫一遍，再补上「修机有动静」
        #   （修机每回合都泄一次位置，而不是等 obs_every），最终取
        #   nodes=20 / obs_every=10 / max_turns=75 / tick=2 / sabotage=0.22
        #   → 随机种群互搏 猎手 0.725，信念峰值命中 0.184（随机 1/20=0.05）。
        # 这个域的信念指标第一次像样：修机时必须静止，猎手才有稳定的东西可推。
        # 刻意**没有**压到 0.50：逃犯还有「选远机/挑时机/贪进度」三条没进化的轴，
        # 从 0.745 往下追同样是真梯度；硬调到五五开反而会把两端都压平。
        self.levels = {"nodes": 0, "hidden": 0, "press": 0, "machines": 0}
        self._last_upgrade = None

    def params(self):
        lv = self.levels
        return dict(
            nodes=20 + 3 * lv["nodes"],
            extra=8 + lv["nodes"],
            # 数值越小 = 线索越频繁 = 越难
            obs_every=max(4, 10 - 2 * lv["hidden"]),
            obs_noise=0.4 + 0.2 * lv["hidden"],
            max_turns=75 - 6 * lv["press"],
            n_machines=5,
            # 门槛从 3 抬到 4：逃犯要多冒一台机器的静止时间
            need=3 + (1 if lv["machines"] >= 2 else 0),
            machine_ticks=2,
            gate_ticks=3,
            sabotage=0.22,
            hunter_slow=2,
        )

    def level_no(self):
        return sum(self.levels.values())

    def describe(self):
        parts = [DIM_LABEL[k] for k in DIMENSIONS if self.levels[k] > 0]
        return " ".join(parts) if parts else "Lv0"

    def try_upgrade(self, recent_h, recent_e):
        """只有**双方势均力敌且都压不动了**，才认为这一档榨干。

        早期版本的判据是「任一方 best >= 0.80 就升级」，在对抗共进化里这是错的：
        一方长期 0.80 是**失衡**信号，不是饱和信号。实测从 gen 120 起 best_e 恒为
        1.000，于是每代都触发升级，正反馈到 —— 逃犯越强 → 越升级 → 逃犯更强，
        level 从 3 一路飙到 9，猎手被压到 0.20 再也起不来。

        正确的饱和判据得看两侧是否处于**胶着**：都在中位区间且波动很小。
        """
        if len(recent_h) < 12:
            return None
        tail_h, tail_e = recent_h[-12:], recent_e[-12:]
        contended = (all(0.45 <= x <= 0.85 for x in tail_h)
                     and all(0.45 <= x <= 0.85 for x in tail_e))
        calm = (max(tail_h) - min(tail_h) < 0.15
                and max(tail_e) - min(tail_e) < 0.15)
        if not (contended and calm):
            return None
        if self._last_upgrade is not None and \
                len(recent_h) - self._last_upgrade < 20:
            return None            # 冷却：升级之后要给种群时间适应
        for k in DIMENSIONS:
            if self.levels[k] < MAX_LEVEL:
                self.levels[k] += 1
                self._last_upgrade = len(recent_h)
                return k
        return None


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def play(gh, ge, P, seed, visible=None):
    h = HunterBot(denorm(gh, BOUNDS_H))
    e = EvaderBot(denorm(ge, BOUNDS_E))
    return HuntEnv(seed=seed, visible=visible, **P).run(h, e)


def evaluate(pop_h, pop_e, P, rng, k=10, reps=3, seed0=0,
             want_visible=False):
    """交叉对战。返回 (fit_h, fit_e, metrics, fit_h_vis)。

    fit_h_vis：同一批 pair、**同一个 seed**（也就是同一张图、同一个起始局面）
    下再跑一次完全信息版本。这样两个数字才可比 ——
    早期版本里 visible 对照用的是另一套 seed，而 seed 决定图，
    于是比的是两张不同的图，差额毫无意义（还出过负数）。

    fit_h[i] = 该猎手的抓捕率；fit_e[j] = 该逃犯的逃脱率。
    metrics 里是这一代所有对局的信念质量均值。
    """
    nh, ne = len(pop_h), len(pop_e)
    ks = min(k, ne)
    kh = min(k, nh)

    caught = np.zeros((nh, ne))
    cnt = np.zeros((nh, ne))
    caught_vis = np.zeros((nh, ne))
    ent_sum = nll_sum = top1_sum = 0.0
    mach_sum = still_sum = 0.0
    n_sample = 0

    for i in range(nh):
        js = rng.choice(ne, size=ks, replace=False)
        for j in js:
            s = 0.0
            for r in range(reps):
                seed = seed0 * 1000003 + i * 7919 + j * 104729 + r
                res = play(pop_h[i], pop_e[j], P, seed)
                # 按 winner 判，不是按 caught 判：超时也算猎手赢（逃犯没逃出去）
                s += 1.0 if res["winner"] == "hunter" else 0.0
                if want_visible:
                    resv = play(pop_h[i], pop_e[j], P, seed, visible=1)
                    caught_vis[i, j] += 1.0 if resv["winner"] == "hunter" else 0.0
                if res.get("belief_entropy") is not None:
                    ent_sum += res["belief_entropy"]
                    nll_sum += res["belief_nll"]
                    top1_sum += res["belief_top1"]
                    mach_sum += res["machines_done"]
                    still_sum += res["still_turns"]
                    n_sample += 1
            caught[i, j] = s
            cnt[i, j] = float(reps)

    m = cnt > 0
    rate = np.where(m, caught / np.maximum(cnt, 1e-9), 0.0)

    fit_h = rate.sum(axis=1) / np.maximum((cnt > 0).sum(axis=1), 1)
    fit_e = 1.0 - (rate.sum(axis=0) / np.maximum((cnt > 0).sum(axis=0), 1))

    # 分子是多次 reptitions 的累加，分母就得是总对局数（cnt 的总和），
    # 不能只数采到了几个对手 —— 那样会得到一个 > 1 的「胜率」。
    fit_h_vis = (caught_vis.sum(axis=1) / np.maximum(cnt.sum(axis=1), 1)
                 if want_visible else None)

    metrics = {
        "belief_entropy": round(ent_sum / max(1, n_sample), 4),
        "belief_nll": round(nll_sum / max(1, n_sample), 4),
        "belief_top1": round(top1_sum / max(1, n_sample), 4),
        # 修机/静止是两个新的过程指标：逃犯到底修到了几台、
        # 为了进度把自己钉在原地多少回合（暴露时长）。
        "machines_done": round(mach_sum / max(1, n_sample), 3),
        "still_turns": round(still_sum / max(1, n_sample), 2),
    }
    # kh 目前未用于 fit_e 的采样（fit_e 用所有 hunter 的结果反推），保留对称
    _ = kh
    return fit_h, fit_e, metrics, fit_h_vis


def visible_control(pop_h_i, pop_e_j, P, n=120):
    """完全信息对照：同一对策略，在猎手能直接看到真实坐标时能拿几分。

    注意：主循环已经不用它了（改用 evaluate 内部同 seed 同 pair 的真对照）。
    留在这里是给手工排查用 —— 单独调用时 seed 序列与正式评估不同，
    结果是**不可比**的，别拿它的数字做结论。
    """
    rng = random.Random(20261030)
    win = 0
    for t in range(n):
        res = play(pop_h_i, pop_e_j, P, seed=t + 1, visible=1)
        win += 1 if res["winner"] == "hunter" else 0
    return win / max(1, n)


# ---------------------------------------------------------------------------
# 遗传
# ---------------------------------------------------------------------------
def tournament(fits, rng, size=3):
    idx = rng.choice(len(fits), size=size, replace=False)
    return int(idx[int(np.argmax(fits[idx]))])


def next_gen(fits, pop, rng, elite=2, sigma=0.12):
    order = np.argsort(-fits)
    new = [list(pop[i]) for i in order[:elite]]
    n = len(pop)
    while len(new) < n:
        a = tournament(fits, rng)
        b = tournament(fits, rng)
        pa, pb = pop[a], pop[b]
        child = [pa[k] if rng.random() < 0.5 else pb[k] for k in range(len(pa))]
        child = [min(1.0, max(0.0, c + float(rng.normal(0.0, sigma))))
                 for c in child]
        new.append(child)
    return new


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="evo")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--k", type=int, default=10, help="每方采样对手数")
    ap.add_argument("--reps", type=int, default=3, help="每个对手跑几个 seed")
    ap.add_argument("--generations", type=int, default=0, help="0=无限")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)
    LOG = os.path.join(OUT, "evolve.log")
    TSV = os.path.join(OUT, "progress.tsv")
    CJSON = os.path.join(OUT, "champions.jsonl")
    BEST = os.path.join(OUT, "best_now.json")
    STATE = os.path.join(OUT, "state.json")

    rng = np.random.default_rng(args.seed)
    world = World()
    pop_h = [random_gene(rng, len(GENE_H)) for _ in range(args.pop)]
    pop_e = [random_gene(rng, len(GENE_E)) for _ in range(args.pop)]
    gen = 0
    hist_h, hist_e = [], []

    if args.resume and os.path.exists(STATE):
        st = json.load(open(STATE, encoding="utf-8"))
        world.levels = st.get("levels", world.levels)
        pop_h = st.get("pop_h", pop_h)
        pop_e = st.get("pop_e", pop_e)
        gen = int(st.get("gen", 0))
        hist_h = st.get("hist_h", [])
        hist_e = st.get("hist_e", [])

    if not os.path.exists(TSV) or gen == 0:
        with open(TSV, "w", encoding="utf-8") as f:
            f.write("gen\tbest_h\tmean_h\tbest_e\tmean_e\tdiv_h\tdiv_e\t"
                    "belief_entropy\tbelief_nll\tbelief_top1\t"
                    "visible_ctrl\tlevel\tsec\n")

    print("hunt 共进化启动: pop=%d k=%d reps=%d" % (args.pop, args.k, args.reps))

    while args.generations == 0 or gen < args.generations:
        t0 = time.time()
        P = world.params()
        want_vis = (gen % 10 == 0)
        fit_h, fit_e, met, fit_h_vis = evaluate(
            pop_h, pop_e, P, rng, k=args.k, reps=args.reps, seed0=gen,
            want_visible=want_vis)
        hist_h.append(float(np.max(fit_h)))
        hist_e.append(float(np.max(fit_e)))

        hist_h = hist_h[-40:]
        hist_e = hist_e[-40:]

        ctrl = None
        if want_vis and fit_h_vis is not None:
            # 同 pair、同 seed 的完全信息抓捕率 —— 差额才是「推断的代价」
            ctrl = float(np.max(fit_h_vis))

        sec = time.time() - t0
        with open(TSV, "a", encoding="utf-8") as f:
            f.write("%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t"
                    "%s\t%d\t%.1f\n"
                    % (gen, np.max(fit_h), np.mean(fit_h),
                       np.max(fit_e), np.mean(fit_e),
                       diversity(pop_h), diversity(pop_e),
                       met["belief_entropy"], met["belief_nll"],
                       met["belief_top1"],
                       "" if ctrl is None else "%.3f" % ctrl,
                       world.level_no(), sec))
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("[gen %04d] %s best_h %.3f best_e %.3f div %.3f/%.3f "
                    "信念 熵%.3f NLL%.3f 峰值%.3f (%.1fs)\n"
                    % (gen, world.describe(), np.max(fit_h), np.max(fit_e),
                       diversity(pop_h), diversity(pop_e),
                       met["belief_entropy"], met["belief_nll"],
                       met["belief_top1"], sec))

        champ_h = pop_h[int(np.argmax(fit_h))]
        champ_e = pop_e[int(np.argmax(fit_e))]
        json.dump({"gen": gen,
                   "hunter": denorm(champ_h, BOUNDS_H),
                   "evader": denorm(champ_e, BOUNDS_E),
                   "fit_h": float(np.max(fit_h)),
                   "fit_e": float(np.max(fit_e)),
                   "metrics": met,
                   "levels": dict(world.levels)},
                  open(BEST, "w", encoding="utf-8"), ensure_ascii=False)
        with open(CJSON, "a", encoding="utf-8") as f:
            f.write(json.dumps({"gen": gen, "params": P,
                                "hunter": [round(x, 3) for x in champ_h],
                                "evader": [round(x, 3) for x in champ_e],
                                "fit_h": round(float(np.max(fit_h)), 3),
                                "fit_e": round(float(np.max(fit_e)), 3),
                                "metrics": met}, ensure_ascii=False) + "\n")
        json.dump({"gen": gen + 1, "levels": world.levels,
                   "pop_h": pop_h, "pop_e": pop_e,
                   "hist_h": hist_h, "hist_e": hist_e},
                  open(STATE, "w", encoding="utf-8"))

        up = world.try_upgrade(hist_h, hist_e)
        if up:
            print("[gen %04d] 难度升级：%s → Lv%d %s"
                  % (gen, DIM_LABEL[up], world.level_no(), world.describe()))

        pop_h = next_gen(fit_h, pop_h, rng)
        pop_e = next_gen(fit_e, pop_e, rng)
        gen += 1


if __name__ == "__main__":
    main()
