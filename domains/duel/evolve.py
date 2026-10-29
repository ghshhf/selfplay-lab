"""
evolve.py —— 自我博弈进化循环（无窗口、日志驱动、可断点续跑、难度自迭代）

和 stream_duel.py 的分工：
  stream_duel.py  随机撒点采数据 —— 广，但每个"人"都很弱
  evolve.py       强者互相打，赢的留下生下一代 —— 深，越打越强

为什么要有它（v2 升级）
-----------------------
v1 只用了 6 个状态机参数，33 代就把"贴身 + 见招就挡 + 只用拳"这个唯一最优解榨干了。
主人点破了根子：**环境是写死的，所以规则太简单，最优解一下就被固定住**。

v2 的核心改动是"随机到底 + 自动升级"：
  1. 基因扩到 9 维：6 原参数 + 武器（离散：徒手/长矛/盾/双刃/投枪）+ 捡道具倾向 + 绕陷阱倾向。
  2. 环境每代随机生成（World 模板）：场地宽度、障碍墙、道具雨、陷阱、物理参数。
     武器由基因决定，环境由难度等级决定 —— 两套随机互不干扰。
  3. 自动升级（迭代算法）：监控 EMA，一旦摸顶就从难度池随机抽一个维度启用，
     强制重新进化。进化循环嵌套在环境生成循环里，最优解永远在动。

日志就是唯一输出（没有窗口）
---------------------------
  evo/evolve.log      人读。每代一小段，带难度、世界、漂移、趋势、诊断
  evo/gens.jsonl      机读。每代一行完整记录
  evo/progress.tsv    表格。一行一代，可直接丢进 Excel 画曲线
  evo/champions.jsonl 冠军档案
  evo/levels.jsonl    难度阶梯。每次自动升级记一行（哪一代谢顶、启用了什么）
  evo/state.json      断点。种群 / 代数 / Elo / 难度 / rng
  evo/best_now.json   新冠军导出。手机侧填进 Bot(**params) 即可复现

设计上刻意避开的三件事：
  1. 不做神经网络。基因是 9 个可读参数 —— 漂移我看得见，权重我看不见。
  2. 不在当代内部循环太狠。全对全 2×2 局就够排序，多打只是把噪声当信号。
  3. 不留"绝对最优"的口子。每代有随机新生个体进来，多样性低时变异加倍。
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

import duel as D
from bot import Bot

# ---------------------------------------------------------------- 基因定义（v2：9 维）

# weapon 是离散字段（索引 WEAPON_NAMES），react 是离散帧数，其余连续。
GENE_NAMES = ("weapon", "aggr", "guard", "kick", "dist", "react", "idle", "loot", "avoid")
WEAPON_IDX = 0          # 第 0 维是武器（离散）
REACT_IDX = 5           # 第 5 维是 react（离散）
INT_IDX = (WEAPON_IDX, REACT_IDX)
N_GENE = len(GENE_NAMES)

# 上下界（顺序对应 GENE_NAMES）
BOUNDS = np.array([
    [0.0, 4.0],     # weapon      武器索引 0..4
    [0.02, 0.99],   # aggr        攻击性
    [0.00, 0.99],   # guard       格挡倾向
    [0.00, 0.95],   # kick        踢招比例
    [0.50, 4.20],   # dist        偏好交战距离（扩到 4.20：投枪射程 4.0）
    [1.0, 20.0],    # react       反应延迟（帧）
    [0.00, 0.65],   # idle        发呆比例
    [0.00, 0.95],   # loot        捡道具倾向
    [0.00, 0.95],   # avoid       绕陷阱倾向
])
SPAN = BOUNDS[:, 1] - BOUNDS[:, 0]

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print("[evo] 收到信号 %s，当前代跑完后退出" % signum, flush=True)


def clamp(g):
    g = np.clip(np.asarray(g, dtype=float), BOUNDS[:, 0], BOUNDS[:, 1])
    g[WEAPON_IDX] = int(round(g[WEAPON_IDX]))
    g[REACT_IDX] = int(round(g[REACT_IDX]))
    return g


def random_genome(rng, weapon_on=True):
    g = clamp(BOUNDS[:, 0] + rng.random(N_GENE) * SPAN)
    if not weapon_on:
        g[WEAPON_IDX] = 0      # 武器维度未启用时强制徒手
    return g


def genome_key(g):
    """把基因变成可当字典键的字符串。离散维取整后才能当 key。"""
    parts = []
    for i, v in enumerate(g):
        parts.append("%d" % round(v) if i in INT_IDX else "%.3f" % v)
    return "/".join(parts)


def genome_kwargs(g):
    return {
        "weapon": D.WEAPON_NAMES[int(round(g[WEAPON_IDX]))],
        "aggression": float(g[1]), "guard_bias": float(g[2]),
        "kick_bias": float(g[3]), "prefer_dist": float(g[4]),
        "react": int(round(g[REACT_IDX])), "idle_bias": float(g[6]),
        "loot_bias": float(g[7]), "avoid_bias": float(g[8]),
    }


def make_bot(g, rng):
    b = Bot(rng=rng, **genome_kwargs(g))
    b.kind = "evo"
    return b


def fmt_gene(g):
    return ("%s  aggr %.2f  guard %.2f  kick %.2f  dist %.2f  react %2d  "
            "idle %.2f  loot %.2f  avoid %.2f"
            % (D.WEAPON_NAMES[int(round(g[0]))], g[1], g[2], g[3], g[4],
               int(round(g[5])), g[6], g[7], g[8]))


# ---------------------------------------------------------------- 难度世界

# 难度维度池。每个维度可独立启用（level>0）。自动升级时从"未启用"里随机抽一个开。
DIMENSIONS = ("weapon", "arena", "wall", "loot", "trap", "physics", "adversarial")
DIM_LABEL = {
    "weapon": "武器进化", "arena": "地图变体", "wall": "障碍墙",
    "loot": "道具雨", "trap": "陷阱", "physics": "物理扰动", "adversarial": "强敌池",
}
DIM_DESC = {
    "weapon": "解锁武器基因：个体可以进化出长矛/盾/双刃/投枪，非对称对抗",
    "arena": "场地宽度每代随机（6~12m），出生点随机",
    "wall": "随机生成障碍墙（挡移动也挡攻击，矮墙变掩体）",
    "loot": "随机降落道具（疾行/狂暴/护甲/治疗），捡到获整局 buff",
    "trap": "随机布陷阱（泥沼/电网/尖刺），踩到掉血或硬直",
    "physics": "摩擦/击退每代随机微调，物理手感在变",
    "adversarial": "对手池扩容：不仅跟当前种群打，还抽历史名人堂 + 随机原型",
}


class World:
    """当前难度配置。make_template(gen) 按代生成确定性世界模板（场地/墙/道具/陷阱/物理）；
    武器不在这里 —— 武器是基因，由个体决定。"""

    def __init__(self, levels=None):
        self.levels = {k: 0 for k in DIMENSIONS}
        if levels:
            for k, v in levels.items():
                if k in self.levels:
                    self.levels[k] = int(v)

    def enabled(self):
        return [k for k in DIMENSIONS if self.levels[k] > 0]

    def weapon_on(self):
        return self.levels.get("weapon", 0) > 0

    def describe(self):
        parts = [DIM_LABEL[k] for k in DIMENSIONS if self.levels[k] > 0]
        return " ".join(parts) if parts else "基础(徒手8m)"

    def level_no(self):
        """难度总级数 = 已启用维度数。写日志用。"""
        return sum(1 for k in DIMENSIONS if self.levels[k] > 0)

    def make_template(self, gen):
        """按 gen 生成确定性世界模板。同一代所有对局共享，保证 win_rate 可比；
        每代不同，进化面对的环境持续变化 —— 这就是"最优解永远在动"的来源。"""
        rng = np.random.default_rng((gen * 2654435761 + 12345) & 0x7FFFFFFF)
        ec = {}
        if self.levels.get("arena", 0) > 0:
            ec["half"] = float(rng.uniform(3.0, 6.0))
        half = ec.get("half", D.ARENA_HALF)
        if self.levels.get("wall", 0) > 0:
            k = int(rng.integers(1, 1 + self.levels["wall"] + 1))
            walls = []
            for _ in range(k):
                wx = float(rng.uniform(-half * 0.55, half * 0.55))
                ww = float(rng.uniform(0.3, 0.7))
                walls.append((wx, ww))
            ec["walls"] = walls
        if self.levels.get("loot", 0) > 0:
            n = int(rng.integers(1, 1 + self.levels["loot"] * 2))
            loot = []
            for _ in range(n):
                x = float(rng.uniform(-half * 0.85, half * 0.85))
                t = D.LOOT_TYPES[int(rng.integers(len(D.LOOT_TYPES)))]
                loot.append((x, t))
            ec["loot"] = loot
        if self.levels.get("trap", 0) > 0:
            m = int(rng.integers(1, 1 + self.levels["trap"] * 2))
            traps = []
            for _ in range(m):
                x = float(rng.uniform(-half * 0.85, half * 0.85))
                t = D.TRAP_TYPES[int(rng.integers(len(D.TRAP_TYPES)))]
                traps.append((x, t))
            ec["traps"] = traps
        if self.levels.get("physics", 0) > 0:
            ec["friction"] = float(rng.uniform(4.0, 9.0))
            ec["knockback"] = float(rng.uniform(1.6, 3.0))
        return ec

    def template_summary(self, ec):
        if not ec:
            return "基础(8m 无墙无道具无陷阱)"
        w = []
        if "half" in ec:
            w.append("场地%.1fm" % (ec["half"] * 2))
        if "walls" in ec:
            w.append("墙%d" % len(ec["walls"]))
        if "loot" in ec:
            w.append("道具%d" % len(ec["loot"]))
        if "traps" in ec:
            w.append("陷阱%d" % len(ec["traps"]))
        if "friction" in ec or "knockback" in ec:
            w.append("物理")
        return " ".join(w)


# ---------------------------------------------------------------- 遗传操作

def crossover(a, b, rng):
    """混合交叉：每个基因一半概率取父一半取母，再以 0.25 概率做算术平均。"""
    mask = rng.random(len(a)) < 0.5
    child = np.where(mask, a, b).astype(float)
    blend = rng.random(len(a)) < 0.25
    child[blend] = 0.5 * (a[blend] + b[blend])
    # 武器离散维：交叉后取整
    child[WEAPON_IDX] = round(child[WEAPON_IDX])
    return clamp(child)


def mutate(g, rng, sigma, weapon_on=True):
    """按量程比例扰动。sigma=1.0 时每维约 0.08 个量程。
    weapon 维度只在 weapon_on 时按 8% 概率随机换一种武器。"""
    g = g.copy()
    m = rng.random(len(g)) < 0.55
    for i in INT_IDX:
        m[i] = False          # 离散维不走高斯，单独处理
    g[m] += rng.normal(0.0, 0.08 * sigma, m.sum()) * SPAN[m]
    # react 离散维：小步扰动后取整（normal(0, 0.3*sigma) 够让它偶尔 ±1）
    g[REACT_IDX] += rng.normal(0.0, 0.3 * sigma)
    if weapon_on and rng.random() < 0.08:
        g[WEAPON_IDX] = int(rng.integers(0, len(D.WEAPON_NAMES)))
    return clamp(g)


def tournament(pop, fits, rng, k=3):
    idx = rng.integers(0, len(pop), size=k)
    return pop[int(idx[np.argmax(fits[idx])])]


def diversity(pop):
    """种群多样性 = 各维归一化标准差的均值（不含武器离散维）。0 表示全一样。"""
    arr = np.asarray(pop, dtype=float)
    cols = [i for i in range(N_GENE) if i != WEAPON_IDX]
    return float(np.mean(np.std(arr[:, cols], axis=0) / SPAN[cols]))


def style_of(g):
    """把基因粗分成五种打法（只看连续 6 维，武器单列）。"""
    aggr, guard, kick, dist = g[1], g[2], g[3], g[4]
    if dist > 2.20:
        return "游走"
    if aggr > 0.60 and guard < 0.55:
        return "猛攻"
    if guard > 0.60 and aggr < 0.55:
        return "铁壁"
    if aggr > 0.50 and guard > 0.60:
        return "反击"
    return "中庸"


def style_counts(pop):
    c = {}
    for g in pop:
        s = style_of(g)
        c[s] = c.get(s, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def human_gene_hint(g):
    """一句话说清这个打法（手机界面给人看的就是这句）。"""
    wp = D.WEAPON_NAMES[int(round(g[0]))]
    aggr, guard, kick, dist, react, loot, avoid = g[1], g[2], g[3], g[4], g[5], g[7], g[8]
    a = "贴身猛攻" if aggr > 0.8 else ("主动找架打" if aggr > 0.55 else
                                   ("等对方先动" if aggr > 0.3 else "几乎不主动"))
    b = "见招就挡" if guard > 0.8 else ("一半概率挡" if guard > 0.45 else "基本不挡")
    c = "爱用踢" if kick > 0.6 else ("拳踢各半" if kick > 0.35 else "几乎只用拳")
    d = ("站 %.1fm 外不进去" % dist) if dist > 2.2 else ("保持 %.1fm" % dist)
    extra = ""
    if loot > 0.5:
        extra += " 爱捡道具"
    if avoid > 0.5:
        extra += " 爱绕陷阱"
    return ("%s持%s / %s / %s / %s / 反应%d帧%s"
            % (D.WEAPONS[wp]["label"], a, b, c, d, int(round(react)), extra))


# ---------------------------------------------------------------- 一局

def play_one(env, a0, a1, max_frames):
    """打一局，只回结算需要的量，不落帧。走 step_raw 而不是 step：
    观测数组在进化里一次都没被读过，构造它纯属浪费（cProfile 占 37%）。"""
    env.reset()
    while True:
        if env.step_raw(a0(env.p0, env.p1), a1(env.p1, env.p0))[2]:
            break
    p0, p1 = env.p0, env.p1
    return {
        "winner": 0 if (p0.alive and not p1.alive) else (
            1 if (p1.alive and not p0.alive) else (
                0 if (not p0.alive and not p1.alive) else
                (0 if p0.hp > p1.hp else (1 if p1.hp > p0.hp else -1)))),
        "frames": env.frames,
        "hp": (p0.hp, p1.hp),
        "damage": tuple(env.damage_dealt),
        "hits": len(env.events),
        "blocked": sum(1 for e in env.events if e[3]),
        "swings": list(env.n_swings),
        "ko": (not p0.alive) or (not p1.alive),
    }


# ---------------------------------------------------------------- Elo

class EloTable:
    """跨代可比的绝对强度尺。当代胜率只能排序，Elo 才能回答"真的变强了吗"。"""

    K = 12.0

    def __init__(self, table=None):
        self.table = dict(table or {})

    def get(self, key):
        return self.table.get(key, 1200.0)

    def update(self, ka, kb, score_a):
        ra, rb = self.get(ka), self.get(kb)
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        self.table[ka] = ra + self.K * (score_a - ea)
        self.table[kb] = rb + self.K * ((1.0 - score_a) - (1.0 - ea))

    def top(self, keys, n=5):
        return sorted(((self.get(k), k) for k in keys), reverse=True)[:n]


# ---------------------------------------------------------------- 一代

def evaluate(pop, cfg, rng, hof_entries, elo, env_cfg, world):
    """全对全循环赛 + 名人堂跨代检验。env_cfg 是本代世界模板（所有对局共享）。"""
    N = len(pop)
    wins = np.zeros((N, N))
    games = np.zeros((N, N))

    g = {"frames": 0, "eps": 0, "hits": 0, "blocked": 0,
         "swings": 0, "ko": 0, "draw": 0, "dmg": [0.0, 0.0]}
    per = [{"w": 0, "g": 0, "dmg": 0.0, "kohit": 0, "frames": 0,
            "hits": 0, "blocked": 0, "swings": 0} for _ in range(N)]

    keys = [genome_key(p) for p in pop]

    seeds = [int(rng.integers(1 << 30)) for _ in range(cfg.reps)]
    gaps = [float(rng.uniform(cfg.gap_lo, cfg.gap_hi)) for _ in range(cfg.reps)]

    def _match(gi, gj, i, j, reps, weight_wins=True):
        w0 = D.WEAPON_NAMES[int(round(gi[WEAPON_IDX]))]
        w1 = D.WEAPON_NAMES[int(round(gj[WEAPON_IDX]))]
        for r in range(reps):
            s, gap = seeds[r % len(seeds)], gaps[r % len(seeds)]
            for swap in (0, 1):
                left_g, right_g = (gi, gj) if swap == 0 else (gj, gi)
                env = D.Duel(seed=s, max_frames=cfg.max_frames, start_gap=gap,
                             env_cfg=env_cfg, weapons=(w0, w1) if swap == 0 else (w1, w0))
                bL = make_bot(left_g, np.random.default_rng(s * 2 + 11 + r))
                bR = make_bot(right_g, np.random.default_rng(s * 2 + 12 + r))
                res = play_one(env, bL, bR, cfg.max_frames)

                li, ri = (i, j) if swap == 0 else (j, i)
                w = res["winner"]
                ko = res["ko"]
                if w < 0:
                    sL, sR = 0.05, 0.05
                else:
                    win_pts = 1.0 if ko else 0.5
                    sL, sR = (win_pts, 0.0) if w == 0 else (0.0, win_pts)

                g["eps"] += 1
                g["frames"] += res["frames"]
                g["hits"] += res["hits"]
                g["blocked"] += res["blocked"]
                g["swings"] += int(res["swings"][0] + res["swings"][1])
                g["ko"] += 1 if ko else 0
                g["draw"] += 1 if w < 0 else 0
                g["dmg"][0] += res["damage"][0]
                g["dmg"][1] += res["damage"][1]

                if li is not None:
                    per[li]["g"] += 1
                    per[li]["w"] += sL
                if ri is not None:
                    per[ri]["g"] += 1
                    per[ri]["w"] += sR
                if li is not None and ri is not None and weight_wins:
                    wins[li][ri] += sL
                    games[li][ri] += 1.0

                for idx, side in ((li, 0), (ri, 1)):
                    if idx is None:
                        continue
                    p = per[idx]
                    p["dmg"] += res["damage"][side] - res["damage"][1 - side]
                    p["frames"] += res["frames"]
                    p["hits"] += res["hits"]
                    p["blocked"] += res["blocked"]
                    p["swings"] += int(res["swings"][side])
                    if res["ko"] and ((w == 0 and side == 0) or (w == 1 and side == 1)):
                        p["kohit"] += 1

                elo.update(genome_key(left_g), genome_key(right_g), sL)

    # ---- 1) 当代全对全 ----
    for i in range(N):
        for j in range(i + 1, N):
            _match(pop[i], pop[j], i, j, cfg.reps)

    # ---- 2) 名人堂跨代检验 + 强敌池 ----
    opponents = list(hof_entries)
    if world.levels.get("adversarial", 0) > 0:
        # 抽几个历史名人堂（已在 opponents）之外，再补随机原型，制造非平稳对手
        for _ in range(min(3, cfg.hof_opp)):
            opponents.append(random_genome(rng, world.weapon_on()))
    if opponents:
        n_opp = min(cfg.hof_opp, len(opponents))
        for i in range(N):
            picks = rng.choice(len(opponents), size=n_opp, replace=False)
            for pi in picks:
                _match(pop[i], opponents[int(pi)], i, None, 1)

    # ---- 3) 算适应度 ----
    fits = np.zeros(N)
    detail = []
    for i in range(N):
        p = per[i]
        gg = max(1, p["g"])
        win_rate = p["w"] / gg
        dmg_diff = p["dmg"] / gg / 100.0
        ko_rate = p["kohit"] / gg
        time_eff = 1.0 - (p["frames"] / gg) / cfg.max_frames
        hit_rate = p["hits"] / max(1, p["swings"])
        # 权重刻意让"打得凶"多拿分（KO 0.12、时间效率 0.12），
        # 只给 win_rate 会让进化找到"拖时间摸鱼"的捷径（v1 实测）。
        fit = (1.00 * win_rate
               + 0.15 * float(np.tanh(dmg_diff))
               + 0.12 * ko_rate
               + 0.12 * max(0.0, time_eff))
        fits[i] = fit
        detail.append({
            "win_rate": round(win_rate, 4),
            "dmg_diff": round(p["dmg"] / gg, 2),
            "ko_rate": round(ko_rate, 4),
            "frames": round(p["frames"] / gg, 1),
            "hit_rate": round(hit_rate, 4),
            "block_rate": round(p["blocked"] / max(1, p["hits"]), 4),
            "fit": round(float(fit), 5),
            "games": p["g"],
        })

    glob = {
        "eps": g["eps"],
        "avg_frames": round(g["frames"] / max(1, g["eps"]), 1),
        "ko_rate": round(g["ko"] / max(1, g["eps"]), 4),
        "draw_rate": round(g["draw"] / max(1, g["eps"]), 4),
        "hit_rate": round(g["hits"] / max(1, g["swings"]), 4),
        "block_rate": round(g["blocked"] / max(1, g["hits"]), 4),
        "dmg_total": round(sum(g["dmg"]), 1),
    }
    return wins, games, fits, detail, glob


def around(x):
    return round(float(x), 1)


# ---------------------------------------------------------------- 冠军战

def champions_round(champ_idx, pop, cfg, rng, env_cfg):
    """最强的几个互相打，重复次数加倍。数据是质量最高的一段，也是"当代之王"的唯一来源。"""
    order = list(champ_idx)
    score = {i: [0.0, 0] for i in order}
    pairs = {}
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            wi = wj = 0.0
            n = 0
            w0 = D.WEAPON_NAMES[int(round(pop[i][WEAPON_IDX]))]
            w1 = D.WEAPON_NAMES[int(round(pop[j][WEAPON_IDX]))]
            for r in range(cfg.reps * 2):
                s = int(rng.integers(1 << 30))
                gap = float(rng.uniform(cfg.gap_lo, cfg.gap_hi))
                for swap in (0, 1):
                    gi, gj = (pop[i], pop[j]) if swap == 0 else (pop[j], pop[i])
                    env = D.Duel(seed=s, max_frames=cfg.max_frames, start_gap=gap,
                                 env_cfg=env_cfg,
                                 weapons=(w0, w1) if swap == 0 else (w1, w0))
                    bL = make_bot(gi, np.random.default_rng(s + 21))
                    bR = make_bot(gj, np.random.default_rng(s + 22))
                    res = play_one(env, bL, bR, cfg.max_frames)
                    i_is_left = (swap == 0)
                    if res["winner"] < 0:
                        si = sj = 0.05
                    else:
                        win_i = res["winner"] == (0 if i_is_left else 1)
                        pts = 1.0 if res["ko"] else 0.5
                        si, sj = (pts, 0.0) if win_i else (0.0, pts)
                    wi += si
                    wj += sj
                    n += 1
            score[i][0] += wi
            score[i][1] += n
            score[j][0] += wj
            score[j][1] += n
            pairs["%d-%d" % (i, j)] = (wi, wj, n)

    def rate(i):
        return score[i][0] / max(1, score[i][1])

    ranked = sorted(order, key=lambda i: -rate(i))
    king = ranked[0]
    return king, rate(king), pairs, score


# ---------------------------------------------------------------- 日志

def diagnose(glob, pop, fits, prev_best, hist, cfg, world):
    """把"我看到的数字"翻译成"下一步该改哪里"。"""
    msgs = []
    arr = np.asarray(pop, dtype=float)
    cols = [1, 2, 3, 4, 5, 6, 7, 8]     # 连续维索引（跳过 weapon）
    means = arr[:, cols].mean(axis=0)
    aggr_m, guard_m, kick_m, dist_m, react_m, idle_m, loot_m, avoid_m = means
    div = diversity(pop)

    # 连续维饱和诊断（和 v1 同样的逻辑，只是索引变了）
    if react_m < 3.2:
        msgs.append("react 均值 %.1f 快贴下限 1 —— 反应速度这一维已饱和" % react_m)
    if glob["hit_rate"] < 0.30:
        msgs.append("命中率 %.0f%% 偏低 —— 大量空挥，可考虑给空挥加硬直"
                    % (100 * glob["hit_rate"]))
    if glob["ko_rate"] < 0.55:
        msgs.append("KO 率 %.0f%% —— 太多局靠时间到判胜，进化方向是「拖」不是「打」"
                    % (100 * glob["ko_rate"]))
    if glob["block_rate"] > 0.55:
        msgs.append("格挡吃掉 %.0f%% 的命中 —— 防守太便宜，试试让格挡也微量掉血"
                    % (100 * glob["block_rate"]))
    if dist_m < 0.85:
        msgs.append("dist 均值 %.2f 挤在贴身 —— 远距离那一段规则不起作用" % dist_m)
    if dist_m > 2.20:
        msgs.append("dist 均值 %.2f 偏外 —— 进化在学「不交战」，检查是不是逃避收益过高"
                    % dist_m)
    if guard_m < 0.08:
        msgs.append("guard 均值 %.2f —— 几乎没人格挡，格挡动作正在变死代码" % guard_m)
    if aggr_m > 0.92:
        msgs.append("aggr 均值 %.2f 顶到上界 —— 攻击性这一维失去区分度" % aggr_m)
    if div < 0.14:
        msgs.append("多样性 %.3f 偏低 —— 早熟收敛，变异系数已上调" % div)

    # 武器分布（启用后才看）
    if world.weapon_on():
        wc = {}
        for g in pop:
            wn = D.WEAPON_NAMES[int(round(g[WEAPON_IDX]))]
            wc[wn] = wc.get(wn, 0) + 1
        msgs.append("武器分布  " + "  ".join("%s×%d" % (k, v) for k, v in
                                            sorted(wc.items(), key=lambda kv: -kv[1])))

    # 平台期：最近 20 代 best 涨幅
    if len(hist) >= 20:
        recent = [h["best_fit"] for h in hist[-20:]]
        if max(recent) - min(recent) < 0.012:
            msgs.append("best 连续 20 代几乎不动（%.3f~%.3f）—— 当前难度下的上限已摸到，"
                        "继续只是刷分，该升级难度" % (min(recent), max(recent)))

    # 同质化判定：mean 在涨、best 不动
    if len(hist) >= 12:
        h = hist[-12:]
        d_best = h[-1]["best_fit"] - h[0]["best_fit"]
        d_mean = h[-1]["mean_fit"] - h[0]["mean_fit"]
        gap0 = h[0]["best_fit"] - h[0]["mean_fit"]
        gap1 = h[-1]["best_fit"] - h[-1]["mean_fit"]
        if d_mean > 0.02 and d_best < 0.005 and gap1 < gap0 * 0.7:
            msgs.append("best 停滞但 mean 涨了 %.3f，头尾差距从 %.3f 收到 %.3f —— "
                        "种群在把最强者追平，不是在超过他" % (d_mean, gap0, gap1))
    return msgs, div


def log_block(gen, el, glob, champ_i, champ_g, champ_fit, king_wr, best_ema,
              drift, trend, diag_msgs, div, best_elo, styles, styles_n,
              pop_max_elo, world, world_summary):
    bar = "─" * 72
    L = [bar]
    L.append("[gen %04d] %s  用时 %.1fs  %d 局  难度 Lv%d %s"
             % (gen, time.strftime("%H:%M:%S"), el, glob["eps"],
                world.level_no(), world.describe()))
    L.append("  世界   " + world_summary)
    L.append("  best %.3f (ema %.3f)  mean %.3f  worst %.3f  div %.3f"
             % (champ_fit, best_ema, glob["_mean_fit"], glob["_worst_fit"], div))
    L.append("  局时 %.0f 帧(%.1fs)  KO %.0f%%  平 %.0f%%  命中率 %.0f%%  "
             "格挡 %.0f%%  总伤害 %.0f"
             % (glob["avg_frames"], glob["avg_frames"] / D.FPS,
                100 * glob["ko_rate"], 100 * glob["draw_rate"],
                100 * glob["hit_rate"], 100 * glob["block_rate"],
                glob["dmg_total"]))
    L.append("  冠军 #%02d  %s" % (champ_i, fmt_gene(champ_g)))
    L.append("        fit %.3f  冠军赛得分率 %.2f  Elo %.0f%s"
             % (champ_fit, king_wr, best_elo,
                ("   |  种群最高 %.0f" % pop_max_elo) if pop_max_elo else ""))
    L.append("  漂移   " + drift)
    L.append("  流派   " + styles + ("   ← 只剩一种了，早熟" if styles_n <= 1 else ""))
    L.append("  趋势   " + trend)
    for m in diag_msgs:
        L.append("  诊断   ⚠ " + m)
    return "\n".join(L)


# ---------------------------------------------------------------- 主循环

def try_upgrade(world, hist, gen, rng, log):
    """自动升级：EMA 压平（连续 plateau_gens 代几乎不动）且还有未启用维度 →
    从难度池随机抽一个启用，记 levels.jsonl，返回 True。这是"迭代算法"的核心。"""
    plateau_gens = 10
    plateau_eps = 0.008
    if len(hist) < plateau_gens:
        return False
    recent_ema = [h.get("best_ema") for h in hist[-plateau_gens:] if "best_ema" in h]
    if len(recent_ema) < plateau_gens:
        return False
    if max(recent_ema) - min(recent_ema) >= plateau_eps:
        return False          # 还在动，没摸顶
    disabled = [k for k in DIMENSIONS if world.levels[k] == 0]
    if not disabled:
        return False          # 全开，没得升了
    # 随机抽一个启用。武器优先早期，但这里全随机也行（主人要"随机到底"）
    pick = disabled[int(rng.integers(len(disabled)))]
    world.levels[pick] += 1
    log("[升级] gen %04d 摸顶（EMA %.3f~%.3f 连续 %d 代 < %.3f）→ 启用 [%s]：%s"
        % (gen, min(recent_ema), max(recent_ema), plateau_gens, plateau_eps,
           DIM_LABEL[pick], DIM_DESC[pick]))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="evo")
    ap.add_argument("--pop", type=int, default=24, help="种群大小，偶数")
    ap.add_argument("--generations", type=int, default=0, help="0=无限")
    ap.add_argument("--reps", type=int, default=2, help="每对每方向局数")
    ap.add_argument("--elite", type=float, default=0.25, help="精英直通比例")
    ap.add_argument("--immigrant", type=float, default=0.15, help="随机新生比例")
    ap.add_argument("--hof-size", type=int, default=24, help="名人堂容量")
    ap.add_argument("--hof-opp", type=int, default=4, help="每个个体每代打几个名人堂")
    ap.add_argument("--max-frames", type=int, default=1800)
    ap.add_argument("--gap-lo", type=float, default=1.5)
    ap.add_argument("--gap-hi", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "evolve.log"
    gens_path = out / "gens.jsonl"
    prog_path = out / "progress.tsv"
    champ_path = out / "champions.jsonl"
    state_path = out / "state.json"
    hof_path = out / "hof.json"
    levels_path = out / "levels.jsonl"

    def log(msg, also_print=True):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        if also_print:
            print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    signal.signal(signal.SIGINT, _on_signal)
    try:
        signal.signal(signal.SIGTERM, _on_signal)
    except (AttributeError, ValueError):
        pass

    seed = args.seed if args.seed is not None else int(time.time()) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)

    # ---------------- 恢复或新建 ----------------
    GENE_VER = 2
    gen = 0
    hist = []
    hof = []
    elo = EloTable()
    restored_ema = None
    world = World()
    pop = None
    if args.resume and state_path.exists():
        with open(state_path, encoding="utf-8") as fh:
            st = json.load(fh)
        if st.get("gene_ver") != GENE_VER:
            # 旧版（v1 6 维）不能直接续跑：备份后从零开始，旧成果不丢。
            backup = out.with_name(out.name + "_v1")
            if not backup.exists():
                import shutil
                shutil.move(str(out), str(backup))
                log("[evo] 检测到 v1 旧数据，已备份到 %s/，本进程从零开始（v2 9 维基因）"
                    % backup.name)
            out.mkdir(parents=True, exist_ok=True)
            for nm in ("evolve.log", "gens.jsonl", "progress.tsv",
                       "champions.jsonl", "state.json", "hof.json", "levels.jsonl"):
                p = out / nm
                if p.exists():
                    p.unlink()
        else:
            gen = st["gen"]
            hist = st.get("hist", [])
            hof = [np.array(g, dtype=float) for g in st.get("hof", [])]
            elo = EloTable(st.get("elo", {}))
            world = World(st.get("levels", {}))
            pop = [np.array(g, dtype=float) for g in st["pop"]]
            rng = np.random.default_rng(st.get("rng_seed", seed) + gen)
            restored_ema = st.get("best_ema")
            log("[evo] 续跑：从 gen %d 接着，种群 %d，名人堂 %d，历史 %d 代，难度 Lv%d"
                % (gen, len(pop), len(hof), len(hist), world.level_no()))

    if pop is None:
        pop = [random_genome(rng, world.weapon_on()) for _ in range(args.pop)]
        # 第一批塞几个已知的极端原型当种子，省几十代
        seeds_known = [
            [0, 0.95, 0.05, 0.50, 0.75, 3, 0.02, 0.3, 0.3],   # 莽夫
            [0, 0.10, 0.90, 0.40, 1.30, 3, 0.10, 0.2, 0.4],   # 龟壳
            [0, 0.55, 0.45, 0.35, 1.20, 4, 0.12, 0.4, 0.3],   # 中庸
            [0, 0.70, 0.35, 0.75, 1.60, 2, 0.05, 0.5, 0.3],   # 长腿
        ]
        for i, g in enumerate(seeds_known[:max(0, args.pop // 6)]):
            pop[i] = clamp(g)
        log("[evo] 开工 seed=%d 种群 %d 每对 %d×2 局 每片上限 %d 帧"
            % (seed, len(pop), args.reps, args.max_frames))
        log("[evo] 初始难度 Lv%d %s" % (world.level_no(), world.describe()))

    prog_header = ("gen\ttime\tbest_fit\tmean_fit\tworst_fit\tdiv\tavg_frames\t"
                   "ko_rate\tdraw_rate\thit_rate\tblock_rate\tbest_elo\t"
                   + "\t".join(GENE_NAMES) + "\n")
    if not prog_path.exists():
        prog_path.write_text(prog_header, encoding="utf-8")

    prev_means = np.asarray(pop).mean(axis=0)
    prev_best = None
    prev_ema = restored_ema
    best_ema = restored_ema
    up_streak = 0
    down_streak = 0
    t_all = time.time()
    stop_file = Path("STOP_EVO")
    immigrant_boost = 0.0      # 升级后临时提高随机新生比例，缓降回基线

    # ---------------- 主循环 ----------------
    while True:
        if _STOP or stop_file.exists():
            log("[evo] 收工信号，停在第 %d 代" % gen)
            break
        if args.generations and gen >= args.generations:
            log("[evo] 到 --generations %d，收工" % args.generations)
            break

        t0 = time.time()
        env_cfg = world.make_template(gen)     # 本代世界模板（确定性、全代共享）
        wins, games, fits, detail, glob = evaluate(
            pop, args, rng, hof, elo, env_cfg, world)

        top_k = min(6, len(pop))
        top_idx = list(np.argsort(-fits)[:top_k])
        king, king_wr, king_pairs, king_score = champions_round(
            top_idx, pop, args, rng, env_cfg)

        champ_i = int(king)
        champ_g = pop[champ_i]
        champ_fit = float(fits[champ_i])
        champ_key = genome_key(champ_g)

        best_ema = champ_fit if best_ema is None else 0.7 * best_ema + 0.3 * champ_fit

        glob["_mean_fit"] = float(np.mean(fits))
        glob["_worst_fit"] = float(np.min(fits))
        div = diversity(pop)

        means = np.asarray(pop).mean(axis=0)
        drift = "  ".join("%s %+.3f" % (GENE_NAMES[i], means[i] - prev_means[i])
                          for i in range(N_GENE))
        if prev_best is not None:
            de = best_ema - prev_ema
            if de > 0.004:
                up_streak += 1
                down_streak = 0
            elif de < -0.004:
                down_streak += 1
                up_streak = 0
            arrow = "↑" if de > 0.004 else ("↓" if de < -0.004 else "→")
            trend = ("ema %+.3f %s  连升 %d 代 / 连降 %d 代"
                     % (de, arrow, up_streak, down_streak))
        else:
            trend = "首代基准"
        prev_best = champ_fit
        prev_ema = best_ema

        diag_msgs, div2 = diagnose(glob, pop, fits, prev_best, hist, args, world)

        el = time.time() - t0
        sc = style_counts(pop)
        styles = "  ".join("%s×%d" % (k, v) for k, v in sc.items())
        pop_max_elo = max(elo.get(genome_key(p)) for p in pop)
        block = log_block(gen, el, glob, champ_i, champ_g, champ_fit,
                          king_wr, best_ema, drift, trend, diag_msgs, div,
                          elo.get(champ_key), styles, len(sc), pop_max_elo,
                          world, world.template_summary(env_cfg))
        log(block, also_print=True)

        # ---- 结构化落盘 ----
        rec = {
            "gen": gen, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(el, 2), "episodes": glob["eps"], "glob": glob,
            "difficulty": world.level_no(), "difficulty_dims": world.enabled(),
            "world": world.template_summary(env_cfg),
            "pop_max_elo": round(pop_max_elo, 1),
            "champion": {"idx": champ_i, "gene": [round(float(v), 4) for v in champ_g],
                         "key": champ_key, "fit": round(champ_fit, 5),
                         "champ_round_winrate": round(king_wr, 4),
                         "elo": round(elo.get(champ_key), 1),
                         "weapon": D.WEAPON_NAMES[int(round(champ_g[WEAPON_IDX]))]},
            "population": [
                {"idx": i, "gene": [round(float(v), 4) for v in pop[i]],
                 "key": genome_key(pop[i]),
                 "elo": round(elo.get(genome_key(pop[i])), 1), **detail[i]}
                for i in range(len(pop))],
            "means": {GENE_NAMES[i]: round(float(means[i]), 4) for i in range(N_GENE)},
            "diversity": round(div, 4), "styles": sc, "diagnosis": diag_msgs,
        }
        with open(gens_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(prog_path, "a", encoding="utf-8") as fh:
            fh.write("%d\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.0f\t"
                     % (gen, rec["ts"], champ_fit, glob["_mean_fit"],
                        glob["_worst_fit"], div, glob["avg_frames"],
                        glob["ko_rate"], glob["draw_rate"], glob["hit_rate"],
                        glob["block_rate"], elo.get(champ_key))
                     + "\t".join("%s" % round(means[i], 4) for i in range(N_GENE))
                     + "\n")

        hist.append({"gen": gen, "best_fit": champ_fit, "mean_fit": glob["_mean_fit"],
                     "div": div, "best_ema": best_ema})

        if not hof or genome_key(hof[-1]) != champ_key:
            hof.append(champ_g.copy())
            hof = hof[-args.hof_size:]
            with open(champ_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "gen": gen, "key": champ_key,
                    "gene": [round(float(v), 4) for v in champ_g],
                    "gene_readable": genome_kwargs(champ_g),
                    "weapon": D.WEAPON_NAMES[int(round(champ_g[WEAPON_IDX]))],
                    "fit": round(champ_fit, 5),
                    "champ_round_winrate": round(king_wr, 4),
                    "elo": round(elo.get(champ_key), 1),
                    "top_rounds": {k: [round(v[0], 2), round(v[1], 2), v[2]]
                                   for k, v in king_pairs.items()},
                    "diagnosis": diag_msgs,
                }, ensure_ascii=False) + "\n")

        best_now = {
            "gen": gen, "ts": rec["ts"],
            "gene": [round(float(v), 5) for v in champ_g],
            "gene_names": list(GENE_NAMES),
            "params": genome_kwargs(champ_g),
            "weapon": D.WEAPON_NAMES[int(round(champ_g[WEAPON_IDX]))],
            "style": style_of(champ_g),
            "style_readable": human_gene_hint(champ_g),
            "fit": round(champ_fit, 5),
            "champ_round_winrate": round(king_wr, 4),
            "elo": round(elo.get(champ_key), 1),
            "pop_max_elo": round(pop_max_elo, 1),
            "difficulty": world.level_no(),
            "difficulty_dims": world.enabled(),
            "note": "直接填进 bot.Bot(**params) 即可复现；gene 是同份参数的原始浮点，手机侧按需取。",
        }
        tmp_bn = (out / "best_now.json").with_suffix(".json.tmp")
        with open(tmp_bn, "w", encoding="utf-8") as fh:
            json.dump(best_now, fh, ensure_ascii=False, indent=1)
        os.replace(tmp_bn, out / "best_now.json")

        # ---- 自动升级：摸顶就从难度池抽一个维度启用 ----
        upgraded = try_upgrade(world, hist, gen, rng, log)
        if upgraded:
            # 升级后环境变了，强制提高随机新生比例（缓降回基线），逼种群重新探索
            immigrant_boost = 0.30
            # 若启用的是武器维度，给一部分个体随机换武器，引入武器多样性
            if world.weapon_on():
                for p in pop:
                    if rng.random() < 0.5:
                        p[WEAPON_IDX] = int(rng.integers(0, len(D.WEAPON_NAMES)))
            with open(levels_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "gen": gen, "ts": rec["ts"], "level": world.level_no(),
                    "enabled": world.enabled(), "ema": round(best_ema, 4),
                }, ensure_ascii=False) + "\n")

        # ---------------- 繁殖 ----------------
        n = len(pop)
        n_elite = max(1, int(round(n * args.elite)))
        imm_rate = min(0.5, args.immigrant + immigrant_boost)
        n_imm = max(1, int(round(n * imm_rate)))
        immigrant_boost = max(0.0, immigrant_boost - 0.05)
        order = np.argsort(-fits)
        new_pop = [pop[i].copy() for i in order[:n_elite]]

        sigma = 1.0 if div > 0.20 else (1.6 if div > 0.12 else 2.4)
        while len(new_pop) < n - n_imm:
            a = tournament(pop, fits, rng)
            b = tournament(pop, fits, rng)
            new_pop.append(mutate(crossover(a, b, rng), rng, sigma, world.weapon_on()))
        while len(new_pop) < n:
            new_pop.append(random_genome(rng, world.weapon_on()))
        pop = new_pop[:n]
        prev_means = means

        st = {
            "gen": gen + 1, "gene_ver": GENE_VER,
            "pop": [[round(float(v), 5) for v in p] for p in pop],
            "hof": [[round(float(v), 5) for v in p] for p in hof],
            "elo": {k: round(v, 2) for k, v in elo.table.items()},
            "levels": world.levels,
            "hist": hist[-200:],
            "best_ema": round(best_ema, 5) if best_ema is not None else None,
            "rng_seed": seed, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = state_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(st, fh, ensure_ascii=False)
        os.replace(tmp, state_path)

        gen += 1

    log("[evo] 停。共 %d 代，总用时 %.1f 分钟" % (gen, (time.time() - t_all) / 60.0))


if __name__ == "__main__":
    main()
