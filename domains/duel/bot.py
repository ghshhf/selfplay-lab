"""
bot.py —— 对打用的脚本 AI（参数随机化，用来产生有变化的数据）

不做神经网络，就一个带参数的状态机。四个参数拉开分布，就能采出
"敢打的人"和"只会挨打的人"，让后续模型既有正面样本也有失败样本。
"""

import numpy as np
import duel as D


class Bot:
    def __init__(self, rng=None,
                 weapon="none",       # 武器名（离散）：none/polearm/shield/twin/javelin
                 aggression=0.55,     # 距离合适时出招概率
                 guard_bias=0.45,     # 看到对手起手时选择格挡的概率（否则后撤）
                 kick_bias=0.35,      # 出招时用踢的比例（仅当武器有踢招时）
                 prefer_dist=1.20,    # 偏好的交战距离（会随武器射程自然漂移）
                 react=4,             # 每几帧决策一次（反应延迟）
                 idle_bias=0.12,      # 没事干时发呆/走位的比例
                 loot_bias=0.30,      # 主动去捡道具的倾向（0~1）
                 avoid_bias=0.30):    # 主动绕开陷阱的倾向（0~1）
        self.rng = rng or np.random.default_rng()
        self.weapon = str(weapon)
        self.aggression = float(aggression)
        self.guard_bias = float(guard_bias)
        self.kick_bias = float(kick_bias)
        self.prefer_dist = float(prefer_dist)
        self.react = max(1, int(react))
        self.idle_bias = float(idle_bias)
        self.loot_bias = float(loot_bias)
        self.avoid_bias = float(avoid_bias)
        self.kind = "param"
        self._act = D.IDLE
        self._wait = 0

    @classmethod
    def random(cls, rng, wide=False):
        """wide=True 时参数域拉开到边界，专门去采稀有但真实的打法。
        weapon 在 wide/non-wide 都随机抽 —— 这里产的是"随机样本数据"，
        不是进化个体；进化个体的武器由基因决定（见 evolve.make_bot）。"""
        wp = rng.choice(D.WEAPON_NAMES)
        if wide:
            return cls(
                rng=rng, weapon=wp,
                aggression=float(rng.uniform(0.05, 0.98)),
                guard_bias=float(rng.uniform(0.02, 0.95)),
                kick_bias=float(rng.uniform(0.02, 0.90)),
                prefer_dist=float(rng.uniform(0.55, 4.20)),
                react=int(rng.integers(1, 21)),
                idle_bias=float(rng.uniform(0.02, 0.60)),
                loot_bias=float(rng.uniform(0.0, 0.95)),
                avoid_bias=float(rng.uniform(0.0, 0.95)),
            )
        return cls(
            rng=rng, weapon=wp,
            aggression=float(rng.uniform(0.15, 0.92)),
            guard_bias=float(rng.uniform(0.05, 0.85)),
            kick_bias=float(rng.uniform(0.05, 0.70)),
            prefer_dist=float(rng.uniform(0.75, 3.20)),
            react=int(rng.integers(2, 11)),
            idle_bias=float(rng.uniform(0.03, 0.30)),
            loot_bias=float(rng.uniform(0.0, 0.60)),
            avoid_bias=float(rng.uniform(0.0, 0.60)),
        )


    def reset(self):
        self._act = D.IDLE
        self._wait = 0

    def __call__(self, me: D.Fighter, opp: D.Fighter) -> int:
        # 出招/硬直中不决策，把动作交还给环境（动作会被忽略）
        if not me.can_act():
            return self._act

        if self._wait > 0:
            self._wait -= 1
            return self._act
        self._wait = self.react - 1

        dist = abs(opp.x - me.x)
        threat = (opp.state == D.S_ATTACK and opp.phase in ("windup", "active"))
        reach = me.max_reach()        # 武器射程自适应（长矛 2.8 / 投枪 4.0 / 徒手 1.5）

        # 0) 陷阱规避（最高优先）：最近陷阱很近且 avoid 高 → 远离陷阱。
        #    踩到硬直/掉血不划算，进化出的 avoid_bias 决定多积极绕。
        if me.trap_dist < 0.9 and self.rng.random() < self.avoid_bias:
            self._act = D.FWD if me.trap_dx < 0 else D.BACK
            return self._act

        # 1) 对手在出招活帧：必防（最关键的防守决策，用武器射程作基准）
        if threat:
            me_attacking = (me.state == D.S_ATTACK)
            if opp.phase == "active" and not me_attacking and dist <= reach + 0.4:
                self._act = D.BLOCK
                return self._act
            if dist <= reach + 0.4 and self.rng.random() < self.guard_bias:
                self._act = D.BLOCK
            elif dist <= reach + 0.4:
                self._act = D.BACK
            else:
                self._act = D.FWD
            return self._act

        # 2) 道具拾取：最近道具很近且 loot 高 → 优先去捡（覆盖贴近对手）
        if me.loot_dist < 1.3 and self.rng.random() < self.loot_bias:
            self._act = D.FWD if me.loot_dx > 0 else D.BACK
            return self._act

        # 3) 太远：贴近到自己的偏好距离（用武器射程作基准，长矛/投枪自然远战）
        if dist > self.prefer_dist + 0.12:
            self._act = D.FWD
            return self._act

        # 4) 太近，且刚打完：拉开
        if dist < 0.72 and me.cooldown > 0:
            self._act = D.BACK
            return self._act

        # 5) 在射程里：按攻击性出招。先筛出"射程 >= 当前距离"能命中的招，
        #    再按 kick_bias 折半投放 —— 否则距离落在拳/踢射程之间时会选拳空挥
        #    （v1 是强制用踢，v2 初版漏了这层，命中率从 88% 掉到 27%）。
        if dist <= reach and self.rng.random() < self.aggression:
            usable = [m for m in me.attacks().keys()
                      if me.reach_of(m) >= dist - 1e-6]
            if not usable:
                usable = list(me.attacks().keys())
            if D.KICK in usable and self.rng.random() < self.kick_bias * 0.5:
                self._act = D.KICK
            else:
                self._act = D.PUNCH if D.PUNCH in usable else usable[0]
            return self._act

        # 6) 什么都不想干
        r = self.rng.random()
        if r < self.idle_bias:
            self._act = D.IDLE
        elif r < self.idle_bias + 0.55:
            self._act = D.FWD if dist > self.prefer_dist else D.BACK
        else:
            self._act = D.BLOCK
        return self._act


class RandomBot:
    """均匀随机动作。人不会这么打，但它是"无策略"的状态分布基线，
    能覆盖到参数化状态机永远走不出来的局面（比如贴脸乱按）。"""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self.kind = "flail"

    def reset(self):
        pass

    def __call__(self, me: D.Fighter, opp: D.Fighter) -> int:
        return int(self.rng.integers(0, 6))


# 原型权重：一半是常规参数机器人（最像人），其余是刻意偏科的极端打法。
# 目的不是"打得好看"，是让数据把状态空间撑开。
KIND_WEIGHTS = {
    "param": 0.45,
    "param_wide": 0.20,
    "rush": 0.10,      # 只会往前压、几乎不格挡
    "turtle": 0.09,    # 疯狂格挡、几乎不主动
    "kite": 0.07,      # 拉开距离打后手
    "coward": 0.05,    # 只逃不打 —— 专门喂"远距离接近"这种样本
    "flail": 0.04,     # 乱按
}


def make_bot(rng, kind):
    """按原型造一个 bot。偏科原型也带抖动，避免每次一模一样。
    这里的武器/loot/avoid 是给"随机样本数据"用的（stream/collect 路径）；
    进化个体的武器由基因决定（见 evolve.make_bot）。"""
    j = lambda a, b: float(rng.uniform(a, b))
    if kind == "param":
        b = Bot.random(rng)
    elif kind == "param_wide":
        b = Bot.random(rng, wide=True)
        b.kind = "param_wide"
    elif kind == "rush":
        b = Bot(rng=rng, weapon="twin", aggression=j(0.88, 0.99), guard_bias=j(0.0, 0.08),
                kick_bias=j(0.25, 0.75), prefer_dist=j(0.70, 0.95),
                react=int(rng.integers(1, 5)), idle_bias=j(0.0, 0.05),
                loot_bias=j(0.2, 0.4), avoid_bias=j(0.0, 0.2))
        b.kind = "rush"
    elif kind == "turtle":
        b = Bot(rng=rng, weapon="shield", aggression=j(0.02, 0.15), guard_bias=j(0.78, 0.98),
                kick_bias=j(0.1, 0.6), prefer_dist=j(1.10, 1.60),
                react=int(rng.integers(1, 6)), idle_bias=j(0.05, 0.25),
                loot_bias=j(0.1, 0.3), avoid_bias=j(0.2, 0.4))
        b.kind = "turtle"
    elif kind == "kite":
        b = Bot(rng=rng, weapon="polearm", aggression=j(0.30, 0.60), guard_bias=j(0.25, 0.60),
                kick_bias=j(0.5, 0.9), prefer_dist=j(1.55, 2.30),
                react=int(rng.integers(1, 5)), idle_bias=j(0.02, 0.12),
                loot_bias=j(0.3, 0.5), avoid_bias=j(0.3, 0.5))
        b.kind = "kite"
    elif kind == "coward":
        # 实测发现 56% 的帧都挤在 1~1.5m，2m 以外只有 4%：bot 都会收敛到交战距离，
        # 光把开局距离随机开没用。要造远距离动态，只能造一个根本不想打的人。
        b = Bot(rng=rng, weapon="none", aggression=j(0.0, 0.04), guard_bias=j(0.05, 0.35),
                kick_bias=j(0.2, 0.8), prefer_dist=j(2.60, 3.60),
                react=int(rng.integers(1, 6)), idle_bias=j(0.0, 0.08),
                loot_bias=j(0.0, 0.15), avoid_bias=j(0.4, 0.6))
        b.kind = "coward"
    elif kind == "flail":
        b = RandomBot(rng)
    else:
        raise ValueError("unknown kind: %r" % (kind,))
    return b


def sample_kind(rng, weights=None):
    w = weights or KIND_WEIGHTS
    keys = list(w)
    p = np.array([w[k] for k in keys], dtype=float)
    p /= p.sum()
    return keys[int(rng.choice(len(keys), p=p))]
