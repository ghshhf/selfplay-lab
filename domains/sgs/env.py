"""sgs/env.py —— 简化 1v1 三国杀环境（不完全信息 self-play 骨架）。

规则（最小可玩版）：
  · 基本牌：杀(shā) / 闪(shǎn) / 桃(táo)
  · 回合制：摸牌 → 出牌阶段(可多次出杀/桃，直到 pass) → 弃牌(手牌>血量则弃到血量) → 换手
  · 杀：指定对手；对手进入响应阶段，出闪抵消，否则掉 1 血
  · 桃：自己回 1 血（未满时）
  · 不完全信息：决策时只能看到「自己手牌 + 双方血量 + 牌堆剩余」，看不到对手手牌
  · 胜负：一方血量 ≤ 0 即负；回合用尽判平局

武将技能层（hero）待 clone 无名杀后接入 —— 届时 gene 增一维 hero（离散），
World 开 hero 难度维，_apply 里按 hero 改规则（如张飞咆哮=无限出杀、华佗青囊=桃多回血…）。
现在的骨架先跑通「信息差 + 策略基因」这条主线。

基因 gene = [aggr, guard, heal, dist] ∈ [0,1]^4
  aggr  出杀倾向（手里有杀就出的概率/阈值）
  guard 留闪倾向（被杀时出闪的概率）
  heal  用桃时机（越低血越用：阈值 = base_hp*(1-heal*0.8)）
  dist  距离掌控（简化：dist 高则杀必中、且自己少被闪抵消的概率低 —— 占位，待武将层细化）
"""
import random

SHA, SHAN, TAO = "sha", "shan", "tao"


class SgsEnv:
    def __init__(self, seed=0, hp=4, draw=2, start_hand=4,
                 deck_comp=(12, 12, 6), max_turns=300):
        self.base_seed = int(seed)
        self.base_hp = hp
        self.draw = draw
        self.start_hand = start_hand
        self.max_turns = max_turns
        self.deck_comp = deck_comp            # (sha, shan, tao) 数量
        self.reset()

    # ---------------------------------------------------------------- 状态
    def reset(self):
        self.rng = random.Random(self.base_seed)
        self.hp = [self.base_hp, self.base_hp]
        self.deck = ([SHA] * self.deck_comp[0] +
                     [SHAN] * self.deck_comp[1] +
                     [TAO] * self.deck_comp[2])
        self.rng.shuffle(self.deck)
        self.hands = [[], []]
        for p in (0, 1):
            self.hands[p] = [self.deck.pop() for _ in range(self.start_hand)]
        self.turn = 0
        self.cur = 0
        self.winner = None
        self.last_event = ""
        self.discard = []                     # 弃牌堆（出过的牌进这里，空了回洗）
        return self

    def view(self, player):
        """player 的可见信息。注意：没有对手手牌 —— 这是不完全信息。"""
        return {
            "me": player,
            "hp": list(self.hp),
            "my_hand": list(self.hands[player]),
            "deck_left": len(self.deck),
            "turn": self.turn,
            "cur": self.cur,
        }

    # ---------------------------------------------------------------- 对局
    def run(self, bot0, bot1):
        """跑完整一局，返回 {'winner': 0/1/-1, 'turns': int}。返回前已 reset。"""
        self.reset()
        bots = (bot0, bot1)
        while self.turn < self.max_turns and self.winner is None:
            p = self.cur
            # 摸牌（牌堆空则弃牌堆回洗，保证牌无限供应 → 对局能持续产生伤害）
            for _ in range(self.draw):
                if not self.deck:
                    self._reshuffle()
                if self.deck:
                    self.hands[p].append(self.deck.pop())
            # 出牌阶段（可多次出牌，直到 pass 或非法）
            while self.winner is None:
                legal = self._legal(p)
                if not legal:
                    break
                act = bots[p].decide(self, p, opp_sha=False)
                if act[0] == "pass" or act not in legal:
                    break
                self._apply(p, act, bots)
            # 弃牌阶段：手牌 > 血量则弃到血量（弃的牌进弃牌堆）
            while len(self.hands[p]) > self.hp[p] and self.hands[p]:
                c = self.hands[p].pop(self.rng.randrange(len(self.hands[p])))
                self.discard.append(c)
            # 换手
            self.cur = 1 - p
            self.turn += 1
        if self.winner is None:
            self.winner = -1                      # 平局
        return {"winner": self.winner, "turns": self.turn}

    def _reshuffle(self):
        """弃牌堆洗回牌堆。牌堆空时调用，保证对局可持续。"""
        if not self.discard:
            return
        self.deck = self.discard
        self.discard = []
        self.rng.shuffle(self.deck)

    def _legal(self, p):
        hand = self.hands[p]
        acts = []
        if SHA in hand:
            acts.append(("sha", 1 - p))
        if TAO in hand and self.hp[p] < self.base_hp:
            acts.append(("tao", p))
        acts.append(("pass",))
        return acts

    def _apply(self, p, act, bots):
        if act[0] == "sha":
            tgt = 1 - p
            resp = bots[tgt].decide(self, tgt, opp_sha=True)
            if resp[0] == "shan" and SHAN in self.hands[tgt]:
                idx = self.hands[tgt].index(SHAN)
                self.discard.append(self.hands[tgt].pop(idx))
                self.last_event = "sha blocked"
            else:
                self.hp[tgt] -= 1
                self.last_event = "sha hit"
                if self.hp[tgt] <= 0:
                    self.winner = p
            idx = self.hands[p].index(SHA)
            self.discard.append(self.hands[p].pop(idx))
        elif act[0] == "tao":
            self.hp[p] = min(self.base_hp, self.hp[p] + 1)
            idx = self.hands[p].index(TAO)
            self.discard.append(self.hands[p].pop(idx))


class Bot:
    """基因→策略。确定性由基因阈值 + 环境 rng 共同决定（避免过拟合单一套路）。"""

    def __init__(self, gene):
        import numpy as np
        self.g = np.asarray(gene, dtype=float)
        self.aggr, self.guard, self.heal, self.dist = self.g

    def decide(self, env, player, opp_sha=False):
        if opp_sha:
            # 被杀：出闪（guard 高必闪，低则按概率；dist 高表示站位好，更敢闪）
            if SHAN in env.hands[player]:
                if self.guard > 0.5 or env.rng.random() < self.guard:
                    return ("shan",)
            return ("hit",)                      # 吃伤害
        # 出牌阶段
        hand = env.hands[player]
        # 桃：血量低于阈值才用
        if TAO in hand:
            thr = max(1.0, env.base_hp * (1.0 - self.heal * 0.8))
            if env.hp[player] <= thr:
                return ("tao", player)
        # 杀：aggr 高倾向出；dist 高表示距离掌控好，更敢出
        if SHA in hand:
            p_sha = min(1.0, 0.5 * self.aggr + 0.5 * self.dist)
            if self.aggr > 0.5 or env.rng.random() < p_sha:
                return ("sha", 1 - player)
        return ("pass",)


if __name__ == "__main__":
    # 冒烟测试：两个随机策略打 200 局，看胜率是否平衡
    import numpy as np
    rng = np.random.default_rng(1)
    wins = [0, 0, 0]
    for i in range(200):
        g0 = rng.random(4)
        g1 = rng.random(4)
        env = SgsEnv(seed=i)
        r = env.run(Bot(g0), Bot(g1))
        wins[r["winner"] + 1] += 1
    print("200 局随机 vs 随机  胜/平/负 =", wins[2], wins[1], wins[0])
    print("说明：接近 1/3 1/3 1/3 说明环境无先天偏袒，骨架可用。")
