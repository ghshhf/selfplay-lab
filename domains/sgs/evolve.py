"""sgs/evolve.py —— 三国杀 1v1 不完全信息 self-play 进化驱动（第二个并行实例）。

复用 _mini_duel 的「世界生成(外层) + 进化(内层) + EMA 平台期自动升级」骨架，
但难度维换成三国杀相关，环境是卡牌回合制而非物理对弈。

基因 gene = [aggr, guard, heal, dist] ∈ [0,1]^4 （见 env.py）
难度维 DIMENSIONS = hand/sha/tao/hp/hero
  · 开局只开 hand（初始手牌），其余关着
  · EMA 连 10 代 spread<0.01 → 随机开一个未开的维 → 自动升级
  · hero(武将池) 留到最后：待 clone 无名杀接入真实武将技能后再开

输出（与 _mini_duel 同构，无窗口、日志驱动）：
  evo/evolve.log  evo/progress.tsv  evo/champions.jsonl
  evo/state.json  evo/levels.jsonl  evo/best_now.json
"""
import os
import sys
import time
import json
import random
import numpy as np
from env import SgsEnv, Bot

# ---------------------------------------------------------------- 基因 / 难度
GENE_NAMES = ("aggr", "guard", "heal", "dist")
N = len(GENE_NAMES)
BOUNDS = np.array([[0.0, 1.0]] * N, float)
SPAN = BOUNDS[:, 1] - BOUNDS[:, 0]

DIMENSIONS = ("hand", "sha", "tao", "hp", "hero")
DIM_LABEL = {"hand": "初始手牌", "sha": "杀密度", "tao": "桃密度",
             "hp": "起始血量", "hero": "武将池"}
START_LEVELS = {"hand": 1}            # 开局只开一维


def clamp(g):
    g = np.asarray(g, float).copy()
    for i in range(N):
        g[i] = min(BOUNDS[i, 1], max(BOUNDS[i, 0], g[i]))
    return g


def random_gene(rng):
    return clamp(rng.random(N))


class World:
    def __init__(self, levels=None):
        self.levels = {k: 0 for k in DIMENSIONS}
        base = dict(START_LEVELS)
        if levels:
            for k, v in levels.items():
                if k in self.levels:
                    base[k] = int(v)
        self.levels.update(base)

    def enabled(self):
        return [k for k in DIMENSIONS if self.levels[k] > 0]

    def level_no(self):
        return sum(1 for k in DIMENSIONS if self.levels[k] > 0)

    def describe(self):
        parts = [DIM_LABEL[k] for k in DIMENSIONS if self.levels[k] > 0]
        return " ".join(parts) if parts else "基础"

    def make_template(self, gen):
        rng = np.random.default_rng((gen * 2654435761 + 777) & 0x7FFFFFFF)
        hp, start_hand, draw = 4, 4, 2
        sha, shan, tao = 12, 12, 6
        if self.levels.get("hand", 0) > 0:
            start_hand = int(rng.integers(3, 3 + self.levels["hand"] * 2))
        if self.levels.get("sha", 0) > 0:
            sha = int(rng.integers(8, 8 + self.levels["sha"] * 6))
        if self.levels.get("tao", 0) > 0:
            tao = int(rng.integers(3, 3 + self.levels["tao"] * 4))
        if self.levels.get("hp", 0) > 0:
            hp = int(rng.integers(3, 3 + self.levels["hp"] * 2))
        # hero 维暂占位，clone 无名杀后接真实武将池
        return {"hp": hp, "draw": draw,
                "start_hand": start_hand, "deck_comp": (sha, shan, tao)}


# ---------------------------------------------------------------- 遗传操作
def tournament(pop, fits, rng, k=3):
    idx = rng.integers(0, len(pop), size=k)
    return pop[int(idx[np.argmax(fits[idx])])]


def crossover(a, b, rng):
    mask = rng.random(N) < 0.5
    child = np.where(mask, a, b).astype(float)
    blend = rng.random(N) < 0.25
    child[blend] = 0.5 * (a[blend] + b[blend])
    return clamp(child)


def mutate(g, rng, sigma=1.0):
    g = g.copy()
    m = rng.random(N) < 0.55
    g[m] += rng.normal(0.0, 0.08 * sigma, N)[m] * SPAN[m]
    return clamp(g)


def diversity(pop):
    arr = np.asarray(pop, float)
    return float(np.mean(np.std(arr, axis=0) / SPAN))


def next_gen(pop, fits, rng, sigma=1.0):
    n = len(pop)
    order = np.argsort(-fits)
    new = [pop[order[0]]]                      # 保精英
    for _ in range(int(0.15 * n)):             # 15% 移民
        new.append(random_gene(rng))
    while len(new) < n:
        p = tournament(pop, fits, rng)
        q = tournament(pop, fits, rng)
        new.append(mutate(crossover(p, q, rng), rng, sigma))
    return np.array(new[:n])


# ---------------------------------------------------------------- 评估
def evaluate(pop, ec, rng, reps=2):
    """3 人身份局评估。
    每局从 pop 随机抽 3 个 bot（可重复），坐 3 个座位，身份由 env 随机分配
    （座位 0 主公，1/2 随机反贼/内奸）。fit = 该 bot 出场胜率。
    """
    n = len(pop)
    score = np.zeros(n)
    count = np.zeros(n)
    for _ in range(reps * n):
        seats = rng.integers(0, n, size=3)
        bots = {s: Bot(pop[seats[s]]) for s in range(3)}
        env = SgsEnv(seed=int(rng.integers(0, 1 << 30)), **ec)
        r = env.run(bots)
        w = r["winner"]
        if w >= 0:
            score[seats[w]] += 1
        for s in range(3):
            count[seats[s]] += 1
    return np.divide(score, count, out=np.full(n, 0.5), where=count > 0)


def try_upgrade(world, hist, gen, rng, log):
    if len(hist) < 10:
        return False
    spread = max(hist[-10:]) - min(hist[-10:])
    if spread < 0.01:
        disabled = [k for k in DIMENSIONS
                    if world.levels[k] == 0 and k != "hero"]   # hero 留到最后
        if not disabled:
            return False
        pick = rng.choice(disabled)
        world.levels[pick] += 1
        log.append((gen, "UPGRADE", "开启难度维: %s" % DIM_LABEL[pick]))
        return True
    return False


# ---------------------------------------------------------------- 主循环
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="evo")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--generations", type=int, default=0, help="0=无限常驻")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true", help="从 state.json 续跑")
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "evolve.log")
    tsv_path = os.path.join(out, "progress.tsv")
    ch_path = os.path.join(out, "champions.jsonl")
    lvl_path = os.path.join(out, "levels.jsonl")
    state_path = os.path.join(out, "state.json")
    best_path = os.path.join(out, "best_now.json")

    rng = np.random.default_rng(args.seed)
    world = World()
    hof = []                                  # 名人堂（历代冠军基因）
    hist = []                                  # 历代 best EMA
    ema = None
    gen = 0
    pop = None

    # 续跑：从 state.json 恢复代/种群/难度/EMA（常驻进程万一被杀可接着跑）
    if args.resume and os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            st = json.load(f)
        gen = int(st["gen"])
        pop = np.array(st["pop"])
        world.levels = dict(st["world"])
        ema = st.get("ema")
        print("[resume] 从 gen %d 续跑，难度=%s" % (gen, world.describe()))

    if pop is None:
        pop = np.array([random_gene(rng) for _ in range(args.pop)])

    # 日志头：全新跑覆盖写，续跑追加（不重复写标题/表头）
    mode = "w" if not args.resume else "a"
    with open(log_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("三国杀 1v1 不完全信息 self-play 进化  (seed=%d pop=%d reps=%d)\n"
                    % (args.seed, args.pop, args.reps))
    with open(tsv_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("gen\tbest\tema\tmean\tdiv\t" + "\t".join(GENE_NAMES) +
                    "\tlevel\tsec\n")

    while (args.generations == 0) or (gen < args.generations):
        t0 = time.time()
        ec = world.make_template(gen)
        fits = evaluate(pop, ec, rng, reps=args.reps)
        best_i = int(np.argmax(fits))
        best_fit = float(fits[best_i])
        ema = best_fit if ema is None else 0.85 * ema + 0.15 * best_fit
        hist.append(ema)
        div = diversity(pop)
        champ = pop[best_i]

        # 冠军赛硬度检验：对名人堂最近 8 个
        hof_score = 0.0
        if hof:
            for hg in hof[-8:]:
                env = SgsEnv(seed=int(rng.integers(0, 1 << 30)), **ec)
                bots = {0: Bot(champ), 1: Bot(hg), 2: Bot(hg)}
                r = env.run(bots)
                hof_score += 1 if r["winner"] == 0 else (0.5 if r["winner"] == -1 else 0)
            hof_score /= len(hof[-8:])

        # 新冠军入堂
        is_new = (not hof) or np.linalg.norm(champ - hof[-1]) > 1e-6
        if is_new:
            hof.append(champ.copy())
            with open(ch_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"gen": gen, "gene": champ.tolist(),
                                    "fit": best_fit, "ema": ema,
                                    "difficulty": world.describe()},
                                   ensure_ascii=False) + "\n")

        # 自动升级
        upg = try_upgrade(world, hist, gen, rng, [])
        if upg:
            with open(lvl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"gen": gen, "levels": world.levels,
                                    "describe": world.describe()},
                                   ensure_ascii=False) + "\n")

        sec = time.time() - t0
        mean_fit = float(np.mean(fits))
        # 日志
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[gen %04d] 难度Lv%d(%s) best %.3f ema %.3f mean %.3f div %.3f "
                    "hof %.2f 冠军 %s (%.1fs)\n"
                    % (gen, world.level_no(), world.describe(), best_fit, ema,
                       mean_fit, div, hof_score,
                       " ".join("%.2f" % x for x in champ), sec))
        with open(tsv_path, "a", encoding="utf-8") as f:
            f.write("%d\t%.4f\t%.4f\t%.4f\t%.4f\t%s\t%d\t%.1f\n"
                    % (gen, best_fit, ema, mean_fit, div,
                       "\t".join("%.4f" % x for x in champ),
                       world.level_no(), sec))

        # 状态 / 最优
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"gen": gen, "seed": args.seed, "pop": pop.tolist(),
                       "world": world.levels, "ema": ema}, f)
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump({"gene_names": list(GENE_NAMES), "gene": champ.tolist(),
                       "params": {"aggr": float(champ[0]), "guard": float(champ[1]),
                                  "heal": float(champ[2]), "dist": float(champ[3])},
                       "fit": best_fit, "ema": ema,
                       "difficulty": world.describe(),
                       "difficulty_dims": world.levels}, f, ensure_ascii=False)

        pop = next_gen(pop, fits, rng)
        gen += 1

    print("done: %d gens, last difficulty = %s" % (gen, world.describe()))


if __name__ == "__main__":
    main()
