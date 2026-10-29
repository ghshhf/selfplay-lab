# -*- coding: utf-8 -*-
"""追捕博弈 hunt —— 图上的不完全信息 pursue-evade。

为什么做这个域
--------------
已有的两个游戏域都跑到了各自的尽头：

- duel：完全信息、对称、连续动作，已收敛到单一流派，多样性崩了。
- sgs：加了身份之后有了隐藏信息，但 Bot 是**纯行为策略**，
  没有地方存放「我认为座位 1 是反贼」这种推断。于是我们只能看胜率
  这一个黑箱数字，**无法分辨它到底是学会了建模对手，还是学会了某种
  恰好够用的动作序列**。

hunt 域把这件事翻到明面上：猎手维护一个「逃犯在哪个节点」的概率分布
（belief），每一回合都在更新。这个分布是可观测的，于是能量化三件事：

1. 信念的**熵**有没有随进化下降 —— 有没有真的在收敛到某个判断；
2. 信念的**峰值**是不是真的指向逃犯 —— 「知道」不等于「知道对了」；
3. 抓住率提高，究竟来自**更好的推断**还是**更强的搜索**。
   办法是用 `visible=1`（完全信息）当对照 —— 让同一套决策逻辑在完全信息
   下能拿多少分，差额就是「不完全信息带来的那部分难度」。

另一个刻意的选择：**零随机**（只有策略本身的随机）。sgs 有牌运，评估预算
被噪声吃掉一大截（实测 mean 0.147，远低于三方对称基线 0.333）。这里给定
同一对手和同一 seed，结果是确定的。

规则
----
- 一张 n 个节点的连通图（环 + 若干弦）。
- 逃犯每回合沿边移动或停留；猎手同理。
- 猎手**看不见**逃犯；每隔 k 回合，逃犯的真实位置泄露一次（"痕迹"）。
- 另外：若两人距离 <= 1，逃犯察觉到危险（知道猎手方位）；距离更远时
  逃犯也不知道猎手在哪 —— 双方各有私域信息，不是单方面的隐藏信息。
- 抓到（同节点）= 猎手胜；撑过 T 回合 = 逃犯胜。
"""

import math
import random

# --- 猎手基因：怎么**推断**对手，以及怎么基于推断行动 ---
GENE_H = [
    "stay_assume",   # 对手模型：假设逃犯每回合停留的概率
    "obs_decay",     # 拿到线索后，其余位置的信念衰减多少
    "prior_noise",   # 信念每回合重注入多少均匀噪声（防过早塌缩）
    "predict_step",  # 向前预测几步（>=1）
    "chase",         # 向信念高处靠拢的权重
    "recency",       # 惩罚最近走过的格子（防原地兜圈）
    "hub",           # 偏好度数高的枢纽节点
    "rand",           # 随机探索率
    "intercept",     # 去**预测的未来位置**拦截 vs 去当前信念峰值
    "persist",       # 锁定一个候选不动摇的程度
    "obs_trust",     # 多大程度相信线索是逃犯本尊 vs 可能偏到邻居
    "model_target",  # 对手模型里，多大程度认为逃犯在朝补给点走
]

BOUNDS_H = [
    (0.0, 0.9), (0.02, 1.0), (0.0, 0.30), (1.0, 4.0),
    (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 0.40),
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
]

# --- 逃犯基因：怎么**让自己难以被推断** ---
GENE_E = [
    "stay",          # 停留概率
    "flee",          # 远离上次被发现的地点
    "hub",           # 偏好枢纽(+1) vs 偏好死角(-1)
    "rand",          # 随机性
    "loop_home",     # 周期性回到起点（可预测的坏习惯）
    "anticipate",    # 预判猎手朝旧脚印来，反向走
    "backtrack",     # 避免重复刚才的边（防被抓到规律）
    "panic",         # 被刚发现过之后立刻改变行为的强度
    "crowd",         # 偏好度数大的节点（借枢纽混淆视听）
    "edge_run",      # 偏好贴着图的边缘走
]

BOUNDS_E = [
    (0.0, 0.8), (0.0, 1.0), (-1.0, 1.0), (0.0, 0.40),
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    (-1.0, 1.0), (0.0, 1.0),
]


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def make_graph(rng, n=12, extra=4):
    """返回一个连通无向图 {node: [邻居]}：一个环，再随机加几条弦。"""
    adj = {i: [] for i in range(n)}

    def link(a, b):
        if a != b and b not in adj[a]:
            adj[a].append(b)
            adj[b].append(a)

    for i in range(n):
        link(i, (i + 1) % n)

    tries = 0
    added = 0
    while added < extra and tries < 400:
        tries += 1
        a = rng.randrange(n)
        b = rng.randrange(n)
        if b in adj[a]:
            continue
        link(a, b)
        added += 1
    return adj


def bfs_dist(adj, src):
    """从 src 出发到各节点的步数（图连通，均能到达）。"""
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def supply_dist(adj, supplies):
    """多源 BFS：每个节点到**最近**补给点的步数。"""
    if not supplies:
        return {}
    dist = {}
    frontier = []
    for s in supplies:
        dist[s] = 0
        frontier.append(s)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def entropy_of(p):
    """香农熵（nat）。p 是 {node: prob}，已归一化。"""
    h = 0.0
    for v in p.values():
        if v > 1e-12:
            h -= v * math.log(v)
    return h


# ---------------------------------------------------------------------------
# 双方 Bot
# ---------------------------------------------------------------------------
class HunterBot:
    """带**显式信念状态**的猎手。

    belief 就是它的世界模型。猎手对逃犯的移动规律有一个假设
    （stay_assume），这个假设未必正确 —— 而真实逃犯的行为恰好教会我们：
    这个世界模型是可以自发校准的，还是始终保持着错误。
    """

    def __init__(self, gene):
        self.gene = list(gene)
        (self.stay_assume, self.obs_decay, self.prior_noise,
         self.predict_step, self.chase, self.recency, self.hub,
         self.rand, self.intercept, self.persist,
         self.obs_trust, self.model_target) = self.gene
        self.belief = None
        self.visit = None
        self.locked = None
        self.rng = random.Random()

    # --- 信念维护 ---
    def reset_belief(self, n, exclude=None):
        self.belief = {v: 1.0 / n for v in range(n)}
        self.visit = {v: 0 for v in range(n)}
        self.locked = None
        if exclude is not None and exclude in self.belief:
            self.belief[exclude] = 0.0
            tot = sum(self.belief.values()) or 1.0
            for v in self.belief:
                self.belief[v] /= tot

    def predict(self, adj, steps=None, supply_map=None):
        """按「对手模型」把信念沿图向前推演若干步。

        supply_map: {node: 到最近补给点的步数}。补给点是公开信息，逃犯的任务
        就是去踩它们 —— 所以一个**合格**的世界模型必须把这件事算进去。
        早期版本只有「随机游走」这一个假设，加了补给点之后猎手的对手模型
        就跟实际差得越来越远，信念质量也就一直不动（实测相关系数 −0.046）。
        model_target 就是「把对手的目标取向加入先验」的权重，可进化。
        """
        steps = steps if steps is not None else int(round(self.predict_step))
        belief = dict(self.belief)
        stay = min(max(self.stay_assume, 0.0), 0.95)
        for _ in range(max(1, steps)):
            nb = {v: 0.0 for v in belief}
            for u, p in belief.items():
                if p <= 0:
                    continue
                nbrs = adj[u]
                if not nbrs:
                    nb[u] += p
                    continue
                nb[u] += p * stay
                share = p * (1.0 - stay) / len(nbrs)
                for v in nbrs:
                    nb[v] += share
            belief = nb
            self._renorm(belief)
        if supply_map and self.model_target > 0:
            att = {v: 1.0 / (1.0 + supply_map.get(v, 99)) for v in belief}
            self._renorm(att)
            w = min(max(self.model_target, 0.0), 1.0)
            for v in belief:
                belief[v] = (1.0 - w) * belief[v] + w * att[v]
            self._renorm(belief)
        return belief

    def observe(self, tip, adj):
        """拿到一次线索，按**贝叶斯乘法**更新信念。

        早期版本是 `belief[pos] = max(belief[pos], 0.55)` —— 等价于假设
        "观测到的位置就是逃犯本尊"。可环境里 obs_noise 高达 0.8（八成的线索
        落在邻居身上），于是那一版在**系统性地强化错误信念**，越进化越差。
        修法是把似然做成 kernel：信任度高就集中在 tip，信任度低就摊到
        tip 的邻居 —— 这才有资格叫信念更新（HMM 前向步）。
        """
        n = len(self.belief)
        lik = {v: 0.0 for v in self.belief}
        trust = min(max(self.obs_trust, 0.0), 1.0)
        nb = list(adj.get(tip, []))
        if trust >= 0.999 or not nb:
            lik[tip] = 1.0
        else:
            lik[tip] = trust
            for v in nb:
                lik[v] = lik.get(v, 0.0) + (1.0 - trust) / len(nb)
        # obs_decay 复用为「似然软化」：越大越不自信，越向均匀摊平
        soft = min(max(self.obs_decay, 0.0), 1.0)
        for v in lik:
            lik[v] = (lik[v] + soft / n) / (1.0 + soft)

        eps = self.prior_noise
        for v in self.belief:
            self.belief[v] = (1 - eps) * self.belief.get(v, 0.0) * lik.get(v, 0.0) \
                             + eps / n
        self._renorm(self.belief)

    def _renorm(self, d):
        tot = sum(d.values())
        if tot <= 1e-12:
            n = len(d) or 1
            for v in d:
                d[v] = 1.0 / n
            return
        for v in d:
            d[v] /= tot

    # --- 行动 ---
    def decide(self, adj, me, visible=None, supply_map=None):
        """返回要去的节点。visible 非 None 时 = 完全信息（对照条件）。"""
        cand = [me] + list(adj[me])
        if visible is not None:
            # 完全信息对照条件：直接咬住真实位置，绕过整套推断 machinery
            target = visible
        else:
            fut = self.predict(adj, supply_map=supply_map)
            cur = dict(self.belief)
            mix = {}
            for v in cur:
                mix[v] = (1 - self.intercept) * cur[v] + self.intercept * fut.get(v, 0.0)
            self._renorm(mix)
            peak = max(mix, key=lambda v: mix[v])
            if (self.locked is not None and self.locked in mix
                    and mix[self.locked] >= self.persist * mix[peak]):
                target = self.locked
            else:
                target = peak
                self.locked = peak

        best, best_s = cand[0], None
        for m in cand:
            dm = bfs_dist(adj, m)
            s = 0.0
            s += self.chase * (1.0 / (1.0 + dm.get(target, 99)))
            s -= self.recency * self.visit.get(m, 0) * 0.1
            s += self.hub * len(adj[m]) * 0.05
            s += self.rand * self.rng.random()
            if best_s is None or s > best_s:
                best, best_s = m, s
        self.visit[me] = self.visit.get(me, 0) + 1
        return best


class EvaderBot:
    """逃犯：不对称地，它连猎手在哪都不知道（只在极近距离时察觉）。"""

    def __init__(self, gene):
        self.gene = list(gene)
        (self.stay, self.flee, self.hub, self.rand, self.loop_home,
         self.anticipate, self.backtrack, self.panic, self.crowd,
         self.edge_run) = self.gene
        self.rng = random.Random()
        self.recent = []

    def decide(self, adj, me, home, last_obs=None, hunter_sensed=None,
               panic=0.0, supplies=None):
        """supplies: **还没踩到的**补给点列表。逃犯的任务是逐个踩完它们，
        所以这里给一个固定的趋向项 —— 这是任务定义，不是可进化的偏好。
        """
        cand = list(adj[me]) + [me]
        best, best_s = me, None
        for m in cand:
            dm = bfs_dist(adj, m)
            s = 0.0
            if supplies:
                nearest = min(dm.get(x, 99) for x in supplies)
                s += 0.6 / (1.0 + nearest)
            if m == me:
                s += self.stay * 2.0
            if last_obs is not None:
                s += self.flee * min(dm.get(last_obs, 0), 6) * 0.25
                # 预判：猎手大概率朝旧脚印来，那就反向走
                s += self.anticipate * min(dm.get(last_obs, 0), 6) * 0.20
            if hunter_sensed is not None:
                # 离察觉到的猎手越近，扣分越多 —— 往远处闪
                s -= max(0.0, 6 - dm.get(hunter_sensed, 99)) * 0.35
            s += self.hub * len(adj[m]) * 0.06
            s += self.crowd * len(adj[m]) * 0.06
            if home is not None:
                s -= self.loop_home * (1.0 - 1.0 / (1.0 + dm.get(home, 99))) * 0.8
            if self.recent and self.backtrack > 0:
                hit = sum(1 for x in self.recent[-3:] if x == m)
                s -= self.backtrack * hit * 0.3
            s += self.rand * self.rng.random()
            s += self.panic * panic * self.rng.random()
            s += self.edge_run * (1.0 if len(adj[m]) <= 2 else 0.0) * 0.2
            if best_s is None or s > best_s:
                best, best_s = m, s
        self.recent.append(me)
        if len(self.recent) > 8:
            self.recent.pop(0)
        return best


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
class HuntEnv:
    # 远距离 Byela - 补给点：逃犯必须逐个访问，**集齐才算逃脱**。
    # 这一维是被逼出来的：没有它，双方同速 + 逃犯有「距离<=1 就察觉」的近距
    # 感知 + 环形图没有死胡同，于是逃犯贴着跑就永远追不上 —— 实测到后期连
    # **完全信息**（看得到）的猎手抓捕率都掉到 0.092。那时候瓶颈已经完全不在
    # 推断，这个域也就失去了意义。
    # 给了补给点，逃犯必须冒险走固定点位，猎手可以预测并拦截 ——
    # 「推断」这才重新变成胜负手。
    def __init__(self, seed=0, nodes=16, extra=6, obs_every=4,
                 max_turns=24, obs_noise=0.5, visible=None, n_supplies=3):
        self.seed = int(seed)
        self.nodes = int(nodes)
        self.extra = int(extra)
        self.obs_every = int(obs_every)
        self.max_turns = int(max_turns)
        self.n_supplies = int(n_supplies)
        # 线索噪声：obx_noise 的概率里，线索落在逃犯的**某个邻居**而不是它本人。
        # 没有这一维，hunter 等价于每 k 回合拿到一次精确坐标，直接走过去几乎必撞，
        # 信念退化成点估计，「推断」也就无从谈起了。
        self.obs_noise = float(obs_noise)
        # visible 不是 None → 完全信息对照条件（把 belief 换成真值）
        self.visible = visible

    def _setup(self):
        rng = random.Random(self.seed)
        self.adj = make_graph(rng, self.nodes, self.extra)
        all_nodes = list(range(self.nodes))
        self.start_h = rng.randrange(self.nodes)
        pool = [v for v in all_nodes
                if bfs_dist(self.adj, self.start_h).get(v, 99) >= 3]
        self.start_e = rng.choice(pool) if pool else rng.randrange(self.nodes)
        self.home = self.start_e
        # 补给点：公开信息，双方都知道。逃犯要逐个踩完才算逃脱，
        # 所以它没法只靠兜圈子活下来 —— 这也是这个域里「推断」重新变得值钱的原因。
        cands = [v for v in all_nodes if v != self.start_e]
        rng.shuffle(cands)
        self.supplies = cands[:self.n_supplies]
        # 猎手侧的对手模型要用：补给点是公开的，这里预先算好各点到最近补给的步数
        self.supply_map = supply_dist(self.adj, self.supplies)

    def run(self, hunter, evader):
        """跑一局，返回 dict。零随机：给定输入，结果确定。"""
        self._setup()
        n = self.nodes
        hpos, epos = self.start_h, self.start_e
        hunter.reset_belief(n, exclude=hpos)
        hunter.rng = random.Random(self.seed * 7919 + 13)
        evader.rng = random.Random(self.seed * 104729 + 29)
        evader.recent = []

        last_obs = None
        turns = 0
        caught = False
        escaped = False
        # 还没踩到的补给点。它在循环外初始化，逐回合减少 ——
        # 逃犯的目标就是把它们跑完。
        todo = list(self.supplies)

        ent_sum = 0.0
        kl_sum = 0.0
        top1_sum = 0.0
        samples = 0

        for turn in range(1, self.max_turns + 1):
            turns = turn
            # 逃犯察觉：只有极近距离才知道猎手方位
            dh = bfs_dist(self.adj, epos)
            sensed = hpos if dh.get(hpos, 99) <= 1 else None
            # 线索阶段：定期泄露逃犯位置
            leak = (self.obs_every > 0 and turn % self.obs_every == 0)
            if leak:
                tip = epos
                if self.obs_noise > 0 and \
                        random.Random(self.seed * 7717 + turn).random() < self.obs_noise:
                    nb = list(self.adj[epos])
                    if nb:
                        tip = nb[random.Random(self.seed * 9371 + turn).randrange(len(nb))]
                last_obs = tip
                hunter.observe(tip, self.adj)
            if self.visible is not None:
                # 完全信息对照：信念被真值顶掉
                hunter.belief = {v: 0.0 for v in range(n)}
                hunter.belief[epos] = 1.0

            # --- 逃犯先动 ---
            new_epos = evader.decide(self.adj, epos, self.home,
                                     last_obs=last_obs,
                                     hunter_sensed=sensed,
                                     panic=1.0 if leak else 0.0,
                                     supplies=todo)
            if new_epos in ([epos] + list(self.adj[epos])):
                epos = new_epos
            if epos in todo:
                todo.remove(epos)
            if epos == hpos:
                caught = True
                break
            if not todo:
                # 补给全部集齐 —— 逃脱成功
                escaped = True
                break

            # --- 猎手后动 ---
            aim = epos if self.visible is not None else None
            new_hpos = hunter.decide(self.adj, hpos, visible=aim,
                                     supply_map=self.supply_map)
            if new_hpos in ([hpos] + list(self.adj[hpos])):
                hpos = new_hpos
            if hpos == epos:
                caught = True
                break

            # --- 信念推演一步（世界模型在跑） ---
            if self.visible is None:
                # 没有新线索时也照样推演一步 —— 预测会让信念自然扩散，
                # 这正是「不知道的时间越久越不确定」应有的表现。
                hunter.belief = hunter.predict(self.adj, steps=1,
                                               supply_map=self.supply_map)

            # --- 记录信念质量 ---
            if self.visible is None:
                ent_sum += entropy_of(hunter.belief)
                p = max(1e-12, hunter.belief.get(epos, 1e-12))
                kl_sum += -math.log(p)          # 真值处的负对数似然
                peak = max(hunter.belief, key=lambda v: hunter.belief[v])
                top1_sum += 1.0 if peak == epos else 0.0
                samples += 1

        return {
            "caught": caught,
            "escaped": escaped,
            # 集齐补给算逃犯赢；超时没集齐，即使没被抓到也算猎手赢
            "winner": "hunter" if caught else ("evader" if escaped else "hunter"),
            "turns": turns,
            "supplies_left": len(todo),
            "belief_entropy": round(ent_sum / max(1, samples), 4),
            "belief_nll": round(kl_sum / max(1, samples), 4),
            "belief_top1": round(top1_sum / max(1, samples), 4),
            "nodes": n,
        }
