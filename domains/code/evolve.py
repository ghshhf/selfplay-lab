"""编程 world model 自博弈进化：环境=多文件代码库/测试，agent 能生成代码。"""
import os, json, time, random, numpy as np
from env import (GENE_NAMES, BOUNDS, DIMENSIONS, DIM_LABEL, CodeEnv, Bot, random_gene)

OUT = "evo"
LOG = os.path.join(OUT, "evolve.log")
TSV = os.path.join(OUT, "progress.tsv")
CH = os.path.join(OUT, "champions.jsonl")
LVL = os.path.join(OUT, "levels.jsonl")
STATE = os.path.join(OUT, "state.json")
BEST = os.path.join(OUT, "best_now.json")

class World:
    def __init__(self):
        self.levels = {k: 0 for k in DIMENSIONS}
        self.levels["size"] = 1  # 开局只开函数体积
        # 上一代冠军基因 {基因名: 原始值}；None = 随机出题（对照模式）。
        # 世界据此定向出题：冠军不爱读码，就多出必须通读才改得对的题。
        self.champ = None
    def make_template(self, gen):
        ec = {k: self.levels[k] for k in DIMENSIONS}
        ec["seed"] = gen * 31 + 1
        ec["champ"] = self.champ
        return ec
    def describe(self):
        parts = [DIM_LABEL[k] for k in DIMENSIONS if self.levels[k] > 0]
        return " ".join(parts) if parts else "Lv0"
    def level_no(self):
        return sum(self.levels.values())
    def try_upgrade(self, hist, gen, rng, best_hist=None):
        if len(hist) < 10:
            return False
        tail = hist[-10:]
        spread = max(tail) - min(tail)

        # 平台判据有两条，任一成立就升：
        #
        # ① EMA 平稳（原判据）—— 适用于"还在慢慢涨"的域。
        #
        # ② 冠军饱和（新增）—— 最近 10 代里 ≥8 代 best_fit 顶到 1.0。
        #    补这条的原因：只用①会漏判。实测 code 域 best 几乎恒为 1.000，
        #    但中间偶尔夹一两代掉到 0.9x，EMA 就在 0.95~1.00 之间抖，
        #    spread 永远 > 0.01，于是 2300 代一次都没升过 —— 不是没平台，
        #    是判据只盯着"均值稳不稳"，没盯"冠军是不是已经打穿评估器"。
        #    这也和 RLHF 里的 reward saturation 同形：代理奖励顶到天花板时，
        #    继续优化只会学到噪声。
        saturated = False
        if best_hist and len(best_hist) >= 10:
            recent = best_hist[-10:]
            saturated = sum(1 for x in recent if x >= 0.999) >= 8

        if spread >= 0.01 and not saturated:
            return False

        # 维度可升到 MAX_LEVEL（让隐藏/体积等维度真正拉开梯度，而非停在 1）
        MAX_LEVEL = 3
        candidates = [k for k in DIMENSIONS if self.levels[k] < MAX_LEVEL]
        if not candidates:
            return False
        # 优先把尚未开启(=0)的维度开起来，开完后再随机挑一个往上加
        first_time = [k for k in candidates if self.levels[k] == 0]
        pool = first_time if first_time else candidates
        pick = rng.choice(pool)
        self.levels[pick] += 1
        return True

def evaluate(pop, ec, rng, reps=2):
    """评估种群。

    ec["champ"] 非 None 时，出题走对抗模式（见 adversarial.py）。
    返回 (fits, counter_txt)：counter_txt 记录这一代世界在针对冠军的什么倾向，
    用于事后核账——对抗到底生效了没有，不能只靠嘴说。
    """
    fits = []
    counter_txt = ""
    for g in pop:
        bot = Bot(g)
        bot.rng = random.Random(int(rng.integers(0, 1 << 30)))
        env = CodeEnv(seed=ec["seed"], size=ec["size"], depth=ec["depth"],
                      hidden=ec["hidden"], scaffold=ec["scaffold"],
                      resource=ec["resource"], champ=ec.get("champ"))
        counter_txt = getattr(env, "counter", "") or ""
        # 多次重试取最好（模拟 agent 多次尝试修复）
        best_passed = False
        best_error = ""
        best_ms = 0
        for _ in range(max(1, min(reps, int(bot.retry_cap)))):
            passed, error, ms, output, files = bot.submit(env)
            if passed:
                best_passed = True
                best_error = ""
                best_ms = ms
                break
            best_error = error
            best_ms = ms
        fit = 1.0 if best_passed else 0.0
        if best_error:
            fit -= 0.1
        if best_ms > 3000:
            fit -= 0.05
        fits.append(fit)
    return np.array(fits), counter_txt

def next_gen(pop, fits, rng, elite=0.25, immigrant=0.15, mutation=1.0):
    n = len(pop)
    order = np.argsort(fits)[::-1]
    elite_n = max(1, int(n * elite))
    imm_n = max(1, int(n * immigrant))
    new = []
    for i in range(elite_n):
        new.append(pop[order[i]].copy())
    while len(new) < n - imm_n:
        a, b = rng.choice(order[:elite_n + 2], size=2, replace=False)
        child = (pop[a] + pop[b]) / 2
        for i in range(len(GENE_NAMES)):
            if rng.random() < 0.3:
                child[i] += rng.normal(0, 0.1 * mutation)
            lo, hi = BOUNDS[i]
            child[i] = max(lo, min(hi, child[i]))
        new.append(child)
    for _ in range(imm_n):
        new.append(random_gene(rng))
    return np.array(new[:n])

def diversity(pop):
    return float(np.mean(np.std(pop, axis=0)))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="evo")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--generations", type=int, default=0, help="0=无限")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    global OUT, LOG, TSV, CH, LVL, STATE, BEST
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)
    LOG = os.path.join(OUT, "evolve.log")
    TSV = os.path.join(OUT, "progress.tsv")
    CH = os.path.join(OUT, "champions.jsonl")
    LVL = os.path.join(OUT, "levels.jsonl")
    STATE = os.path.join(OUT, "state.json")
    BEST = os.path.join(OUT, "best_now.json")

    rng = np.random.default_rng(args.seed)
    world = World()
    hof = []
    hist = []
    best_hist = []   # 每代冠军原始 fit，判「饱和」用（ema 会掩盖它）
    ema = None
    gen = 0
    pop = None

    if args.resume and os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            st = json.load(f)
        gen = int(st["gen"])
        pop = np.array([Bot(g) for g in st["pop"]])
        world.levels = dict(st["world"])
        ema = st.get("ema")
        print("[resume] 从 gen %d 续跑，难度=%s" % (gen, world.describe()))

    if pop is None:
        pop = [Bot.random(rng) for _ in range(args.pop)]

    mode = "w" if not args.resume else "a"
    with open(LOG, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("编程 world model self-play 进化  (seed=%d pop=%d reps=%d)\n"
                    % (args.seed, args.pop, args.reps))
    with open(TSV, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("gen\tbest\tema\tmean\tdiv\t" + "\t".join(GENE_NAMES) +
                    "\tlevel\tsec\n")

    while (args.generations == 0) or (gen < args.generations):
        t0 = time.time()
        ec = world.make_template(gen)
        # Bot 列表转基因数组供 evaluate 读取（兼容旧接口）
        genes = np.array([b.gene for b in pop])
        fits, counter_txt = evaluate(genes, ec, rng, reps=args.reps)
        best_i = int(np.argmax(fits))
        best_fit = float(fits[best_i])
        ema = best_fit if ema is None else 0.85 * ema + 0.15 * best_fit
        hist.append(ema)
        best_hist.append(best_fit)
        div = diversity(genes)
        champ = pop[best_i]

        # 世界记住这一代冠军，下一代据此定向出题（对抗模式）。
        # 放在名人堂之前：本代冠军一旦确定，下一代的世界就要针对它。
        world.champ = {GENE_NAMES[i]: float(champ.gene[i])
                       for i in range(len(GENE_NAMES))}

        # 名人堂
        is_new = (not hof) or np.linalg.norm(np.array(champ.gene) - np.array(hof[-1].gene)) > 1e-6
        if is_new:
            hof.append(champ)
            with open(CH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"gen": gen, "gene": list(champ.gene),
                                    "fit": best_fit, "ema": ema,
                                    "difficulty": world.describe()},
                                   ensure_ascii=False) + "\n")

        # 自动升级
        upg = world.try_upgrade(hist, gen, rng, best_hist=best_hist)
        if upg:
            with open(LVL, "a", encoding="utf-8") as f:
                f.write(json.dumps({"gen": gen, "levels": world.levels,
                                    "describe": world.describe()},
                                   ensure_ascii=False) + "\n")

        sec = time.time() - t0
        mean_fit = float(np.mean(fits))
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("[gen %04d] 难度Lv%d(%s) best %.3f ema %.3f mean %.3f div %.3f "
                    "hof %d 冠军 %s (%.1fs)%s\n"
                    % (gen, world.level_no(), world.describe(), best_fit, ema,
                       mean_fit, div, len(hof),
                       " ".join("%.2f" % x for x in champ.gene), sec,
                       (" | 针对: " + counter_txt) if counter_txt else ""))
        with open(TSV, "a", encoding="utf-8") as f:
            f.write("%d\t%.4f\t%.4f\t%.4f\t%.4f\t%s\t%d\t%.1f\n"
                    % (gen, best_fit, ema, mean_fit, div,
                       "\t".join("%.4f" % x for x in champ.gene),
                       world.level_no(), sec))

        with open(STATE, "w", encoding="utf-8") as f:
            json.dump({"gen": gen, "seed": args.seed,
                       "pop": [list(b.gene) for b in pop],
                       "world": world.levels, "ema": ema}, f)
        with open(BEST, "w", encoding="utf-8") as f:
            json.dump({"gene_names": list(GENE_NAMES), "gene": list(champ.gene),
                       "params": {GENE_NAMES[i]: float(champ.gene[i]) for i in range(len(GENE_NAMES))},
                       "fit": best_fit, "ema": ema,
                       "difficulty": world.describe(),
                       "difficulty_dims": world.levels}, f, ensure_ascii=False)

        pop = [Bot(g) for g in next_gen(genes, fits, rng)]
        gen += 1

    print("done: %d gens, last difficulty = %s" % (gen, world.describe()))

if __name__ == "__main__":
    main()