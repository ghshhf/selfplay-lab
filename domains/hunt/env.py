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
    "model_target",  # 对手模型里，多大程度认为逃犯在朝目标点（机器/门）走
    "patrol",        # 守机：往「有进度的解码机」靠的权重（vs 纯追信念峰值）
    "camp",          # 守门：通电后往大门靠的权重
]

BOUNDS_H = [
    (0.0, 0.9), (0.02, 1.0), (0.0, 0.30), (1.0, 4.0),
    (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 0.40),
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    (0.0, 1.5), (0.0, 1.5),
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
    "greed",         # 贪进度：猎手在附近时还赖在机器上修的胆量（沉没成本）
    "machine_far",   # 挑机器偏好：远机安全 vs 近机省时
]

BOUNDS_E = [
    (0.0, 0.8), (0.0, 1.0), (-1.0, 1.0), (0.0, 0.40),
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
    (-1.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
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
         self.obs_trust, self.model_target,
         self.patrol, self.camp) = self.gene
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

    def predict(self, adj, steps=None, objective_map=None):
        """按「对手模型」把信念沿图向前推演若干步。

        objective_map: {node: 到最近目标点（解码机/大门）的步数}。目标点是
        公开信息，逃犯的任务就是去那里 —— 所以一个**合格**的世界模型必须
        把这件事算进去。早期版本只有「随机游走」这一个假设，猎手的对手模型
        就跟实际差得越来越远，信念质量一直不动（实测相关系数 −0.046）。
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
        if objective_map and self.model_target > 0:
            att = {v: 1.0 / (1.0 + objective_map.get(v, 99)) for v in belief}
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
    def decide(self, adj, me, visible=None, objective_map=None,
               machines=None, powered=False, gates=None):
        """返回要去的节点。

        visible 非 None 时 = 完全信息（对照条件）。

        machines: [(节点, 进度)] —— 解码机是**公开信息**（第五人格里猎手
        也能看到哪台在闪），所以守机不需要推断。这让猎手第一次有了
        两个互相冲突的动作：追信念峰值（可能被遛）vs 守有进度的机器
        （可能扑空）。patrol/camp 就是这两个倾向的权重，交给进化去调。
        """
        cand = [me] + list(adj[me])
        if visible is not None:
            # 完全信息对照条件：直接咬住真实位置，绕过整套推断 machinery
            target = visible
        else:
            fut = self.predict(adj, objective_map=objective_map)
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
            # 守机：往「有进度且没修完」的机器靠。进度越高越值得守 ——
            # 一台 90% 的机器被砸掉，比一台 10% 的机器疼得多。
            if self.patrol > 0 and machines:
                hunt_m = 0.0
                for mv, prog in machines:
                    if prog <= 0.0 or prog >= 1.0:
                        continue
                    hunt_m = max(hunt_m, prog / (1.0 + dm.get(mv, 99)))
                s += self.patrol * hunt_m
            # 守门：通电之后，门才是真正的终局
            if self.camp > 0 and powered and gates:
                s += self.camp * max(1.0 / (1.0 + dm.get(g, 99)) for g in gates)
            s -= self.recency * self.visit.get(m, 0) * 0.1
            s += self.hub * len(adj[m]) * 0.05
            s += self.rand * self.rng.random()
            if best_s is None or s > best_s:
                best, best_s = m, s
        self.visit[me] = self.visit.get(me, 0) + 1
        return best


class EvaderBot:
    """逃犯：不对称地，它连猎手在哪都不知道（只在极近距离时察觉）。

    它的胜利路径和第五人格同构，必须走完三步：
      1. 站在解码机上**静止**若干回合把进度修满
      2. 修满 3 台 → 大门通电
      3. 站到大门上**静止**若干回合开锁
    关键不是"修机"这个动作，而是**修机必须静止**：它把"移动自由"当筹码押了出去。
    之前那版逃犯可以一直绕圈，于是猎手永远算不准；现在它必须反复钉在
    固定点位上，推断才第一次有真东西可抓。
    """

    def __init__(self, gene):
        self.gene = list(gene)
        (self.stay, self.flee, self.hub, self.rand, self.loop_home,
         self.anticipate, self.backtrack, self.panic, self.crowd,
         self.edge_run, self.greed, self.machine_far) = self.gene
        self.rng = random.Random()
        self.recent = []

    def pick_goal(self, adj, me, machines, gates, powered):
        """挑这一趟要去哪个目标点。返回 (node, 当前进度) 或 None。

        machine_far 是这一版新加的选择轴：远机更安全（猎手跑过来要时间）
        但路上风险更大、总耗时更长。0.5 = 完全不在乎距离。
        """
        if powered:
            cand = [(g, 1.0) for g in gates]
        else:
            cand = [(mv, pr) for mv, pr in machines if pr < 1.0]
        if not cand:
            return None
        d0 = bfs_dist(adj, me)
        best, best_s = None, None
        for node, prog in cand:
            d = d0.get(node, 99)
            # 进度高的机器有沉没成本，值得回去补
            s = prog * 1.5
            # machine_far > 0.5 → 偏好远机；< 0.5 → 偏好近机
            s += (self.machine_far * 2.0 - 1.0) * d * 0.35
            if best_s is None or s > best_s:
                best, best_s = (node, prog), s
        return best

    def decide(self, adj, me, machines, gates, powered,
               last_obs=None, hunter_sensed=None, panic=0.0):
        """返回要去的节点（返回 me 本身就是「留在这里」，也就是在修/在开门）。"""
        goal = self.pick_goal(adj, me, machines, gates, powered)
        goal_node, here_prog = (goal if goal else (None, 0.0))

        cand = list(adj[me]) + [me]
        best, best_s = me, None
        for m in cand:
            dm = bfs_dist(adj, m)
            s = 0.0
            # 趋向目标 —— 这是任务定义，不是可进化的偏好
            if goal_node is not None:
                s += 0.6 / (1.0 + dm.get(goal_node, 99))
            if m == me:
                s += self.stay * 2.0
            if last_obs is not None:
                s += self.flee * min(dm.get(last_obs, 0), 6) * 0.25
                # 预判：猎手大概率朝旧脚印来，那就反向走
                s += self.anticipate * min(dm.get(last_obs, 0), 6) * 0.20
            if hunter_sensed is not None:
                dh = dm.get(hunter_sensed, 99)
                danger = max(0.0, 1.0 - dh / 4.0)     # 越近越危险
                s -= danger * 2.5
                # 贪：进度越高越舍不得走，greed 就是「愿意拿命换进度」的那部分。
                # 这一项和上面的惩罚方向相反，所以 greed 高的个体真的会赌。
                if m == me and here_prog > 0.0:
                    s += self.greed * here_prog * danger * 2.0
            s += self.hub * len(adj[m]) * 0.06
            s += self.crowd * len(adj[m]) * 0.06
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
    # 第五人格式的不对称对抗：逃脱者必须**静止**才能推进度。
    #
    # 三轮迭代的教训在这里各留一条：
    # 1. 逃犯能一直绕圈时（双方同速 + 近距感知 + 环形图没死胡同），连
    #    **完全信息**的猎手抓捕率都掉到 0.092 —— 瓶颈彻底不在推断。
    # 2. 只加「踩点位」不够：逃犯踩完就走，位置照样飘，推断的代价只有 +0.065。
    # 3. 真正的解法是让**静止成为进度的唯一来源**（修机/开门）。静止 = 位置被钉死，
    #    猎手第一次有稳定的东西可以推断；而逃犯要在「贪进度（暴露）」和
    #    「保命（没进度）」之间反复取舍 —— 这个取舍才是这个域的核心张力。
    def __init__(self, seed=0, nodes=16, extra=6, obs_every=4,
                 max_turns=40, obs_noise=0.5, visible=None,
                 n_machines=5, need=3, machine_ticks=4, gate_ticks=3,
                 sabotage=0.34, hunter_slow=2):
        self.seed = int(seed)
        self.nodes = int(nodes)
        self.extra = int(extra)
        self.obs_every = int(obs_every)
        self.max_turns = int(max_turns)
        self.n_machines = int(n_machines)
        self.need = int(need)
        self.tick = 1.0 / max(1, int(machine_ticks))
        self.gate_tick = 1.0 / max(1, int(gate_ticks))
        # 猎手路过未修完的机器时，砸掉的进度比例。这一条是「巡逻」存在的理由：
        # 没有它，猎手只有「追」一个动作，守机就永远是次优解。
        self.sabotage = float(sabotage)
        # 猎手每 hunter_slow 回合才走一步。这是非对称对抗的标准补偿：
        # 猎手更强但更慢，逃犯靠机动拉开距离 —— 于是「停下修机」才真正有代价，
        # 因为它把唯一的速度优势也押了出去。没有这一条，20 节点图上猎手
        # 几乎是必撞（实测随机种群 0.86~0.95 胜率），域就退化成"谁先看见谁赢"。
        self.hunter_slow = max(1, int(hunter_slow))
        # 线索噪声：obs_noise 的概率里，线索落在逃犯的**某个邻居**而不是它本人。
        # 没有这一维，猎手等价于每 k 回合拿到一次精确坐标，直接走过去几乎必撞，
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

        # 解码机：尽量铺开。挤在一团的话逃犯修一台走两步就到下一台，
        # 「带路的时间 vs 静止的风险」这个取舍就消失了。
        cands = [v for v in all_nodes if v != self.start_e]
        rng.shuffle(cands)
        self.machines = []
        for v in cands:
            if len(self.machines) >= self.n_machines:
                break
            if not self.machines:
                self.machines.append(v)
                continue
            if min(bfs_dist(self.adj, v).get(u, 99) for u in self.machines) >= 2:
                self.machines.append(v)
        for v in cands:                      # 铺不开就凑满
            if len(self.machines) >= self.n_machines:
                break
            if v not in self.machines:
                self.machines.append(v)
        self.machine_prog = {v: 0.0 for v in self.machines}

        # 大门：取离所有机器总和最远的两个点。通电之后逃犯还得再跑一段，
        # 于是开门成了第二个"必须静止"的暴露窗口。
        dsum = {}
        for v in all_nodes:
            dv = bfs_dist(self.adj, v)
            dsum[v] = sum(dv.get(mv, 99) for mv in self.machines)
        gpool = [v for v in all_nodes if v not in self.machines] or all_nodes
        gpool.sort(key=lambda v: -dsum.get(v, 0))
        self.gates = gpool[:2]

    def _objective_map(self, powered):
        """到最近「该去的地方」的步数，给猎手的对手模型用。"""
        src = self.gates if powered else \
            [v for v, pr in self.machine_prog.items() if pr < 1.0]
        return supply_dist(self.adj, src) if src else {}

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
        powered = False
        gate_prog = 0.0
        still_turns = 0        # 花了多少回合静止在机器/门上（暴露时长）

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

            # --- 逃犯先动（返回原格 = 留在这里，也就是在修机/开门） ---
            mlist = list(self.machine_prog.items())
            new_epos = evader.decide(self.adj, epos, mlist, self.gates, powered,
                                     last_obs=last_obs,
                                     hunter_sensed=sensed,
                                     panic=1.0 if leak else 0.0)
            if new_epos in ([epos] + list(self.adj[epos])):
                epos = new_epos
            if epos == hpos:
                caught = True
                break

            # --- 静止即工作 ---
            repair_tip = None
            if not powered:
                pr = self.machine_prog.get(epos, 0.0)
                if pr < 1.0:
                    self.machine_prog[epos] = min(1.0, pr + self.tick)
                    still_turns += 1
                    # 修机有动静：密码机在闪，猎手看得见哪台被修。
                    # 这是「静止」的真正代价 —— 暴露频率从「每 obs_every 回合
                    # 一次」变成「每回合一次」，代价高了一个量级。
                    # 没有这一条，逃犯可以无风险地修机，整个域就没有张力：
                    # 实测信念熵 2.94 ≈ 均匀 ln(20)=2.996，等于什么也没推断出来。
                    repair_tip = epos
                    if sum(1 for x in self.machine_prog.values() if x >= 1.0) >= self.need:
                        powered = True
            else:
                if epos in self.gates:
                    gate_prog = min(1.0, gate_prog + self.gate_tick)
                    still_turns += 1
                    repair_tip = epos          # 开门同样有动静
                    if gate_prog >= 1.0:
                        escaped = True
                        break
            if repair_tip is not None and self.visible is None:
                tip = repair_tip
                if self.obs_noise > 0 and \
                        random.Random(self.seed * 6151 + turn).random() < self.obs_noise * 0.6:
                    nb = list(self.adj[repair_tip])
                    if nb:
                        tip = nb[random.Random(self.seed * 8317 + turn).randrange(len(nb))]
                hunter.observe(tip, self.adj)

            # --- 猎手后动（每 hunter_slow 回合才走一步：更快会变成"看见就赢"） ---
            if turn % self.hunter_slow == 0:
                aim = epos if self.visible is not None else None
                new_hpos = hunter.decide(self.adj, hpos, visible=aim,
                                         objective_map=self._objective_map(powered),
                                         machines=list(self.machine_prog.items()),
                                         powered=powered, gates=self.gates)
                if new_hpos in ([hpos] + list(self.adj[hpos])):
                    hpos = new_hpos
                if hpos == epos:
                    caught = True
                    break
                # --- 砸机：猎手路过有进度的机器就砸掉一截 ---
                hp_ = self.machine_prog.get(hpos, 0.0)
                if 0.0 < hp_ < 1.0:
                    self.machine_prog[hpos] = max(0.0, hp_ - self.sabotage)

            # --- 信念推演一步（世界模型在跑） ---
            if self.visible is None:
                # 没有新线索时也照样推演一步 —— 预测会让信念自然扩散，
                # 这正是「不知道的时间越久越不确定」应有的表现。
                hunter.belief = hunter.predict(self.adj, steps=1,
                                               objective_map=self._objective_map(powered))

            # --- 记录信念质量 ---
            if self.visible is None:
                ent_sum += entropy_of(hunter.belief)
                p = max(1e-12, hunter.belief.get(epos, 1e-12))
                kl_sum += -math.log(p)          # 真值处的负对数似然
                peak = max(hunter.belief, key=lambda v: hunter.belief[v])
                top1_sum += 1.0 if peak == epos else 0.0
                samples += 1

        machines_done = sum(1 for x in self.machine_prog.values() if x >= 1.0)
        return {
            "caught": caught,
            "escaped": escaped,
            # 门开满才算逃出去；超时没出去，即使没被抓到也算猎手赢
            "winner": "hunter" if caught else ("evader" if escaped else "hunter"),
            "turns": turns,
            "machines_done": machines_done,
            "powered": powered,
            "gate_prog": round(gate_prog, 3),
            "still_turns": still_turns,
            "belief_entropy": round(ent_sum / max(1, samples), 4),
            "belief_nll": round(kl_sum / max(1, samples), 4),
            "belief_top1": round(top1_sum / max(1, samples), 4),
            "nodes": n,
        }
