"""sgs/env.py —— 3 人身份局三国杀环境（不完全信息 self-play 骨架）。

身份（每局固定 3 个座位，随机分配）：
  主公 ×1  反贼 ×1  内奸 ×1
座位 0 固定为主公并先手；座位 1、2 随机分给反贼/内奸。
身份只对本人可见（view() 不泄露他人身份），这是不完全信息的核心。

血量：主公 4，反贼 3，内奸 3（经典设定，主公血多但身份暴露）。

胜负（每回合末 _check_winner() 判定）：
  · 主公胜：所有反贼、内奸死亡
  · 反贼胜：主公死亡（反贼即便同归于尽也算胜）
  · 内奸胜：主公是最后一个死亡的人（内奸需在反贼死后才杀主公）

基本牌：杀 / 闪 / 桃
回合：摸牌 → 出牌阶段（可对任意活着对手出杀/桃，直到 pass）→ 弃牌 → 换手

基因 gene = [aggr, guard, heal, dist] ∈ [0,1]^4
  aggr  出杀倾向
  guard 留闪倾向
  heal  用桃时机（阈值 = base_hp*(1-heal*0.8)）
  dist  距离掌控（占位，待武将层细化）
身份策略由 Bot 决定自己身份后硬编码倾向（见 Bot.decide 的 target 选择）。
"""
import random

SHA, SHAN, TAO = "sha", "shan", "tao"
ZHU, FAN, NEI = "zhu", "fan", "nei"   # 主公、反贼、内奸
IDENTITIES = [ZHU, FAN, NEI]
BASE_HP = {ZHU: 4, FAN: 3, NEI: 3}


class SgsEnv:
    def __init__(self, seed=0, draw=2, start_hand=4,
                 deck_comp=(12, 12, 6), max_turns=300, hp=None):
        self.base_seed = int(seed)
        self.draw = draw
        self.start_hand = start_hand
        self.deck_comp = deck_comp          # (sha, shan, tao) 数量
        self.max_turns = max_turns
        self.reset(hp=hp)

    # ---------------------------------------------------------------- 状态
    def reset(self, hp=None):
        self.rng = random.Random(self.base_seed)
        # 身份：座位 0 主公，座位 1/2 随机反贼/内奸
        others = [FAN, NEI]
        self.rng.shuffle(others)
        self.identities = [ZHU] + others
        self.base_hp = {p: BASE_HP[self.identities[p]] for p in range(3)}
        self.hp = {p: self.base_hp[p] for p in range(3)}
        if hp is not None:
            # 难度维“hp”：统一覆盖所有座位起始血量，base_hp/hp 同步，
            # 避免桃的回血上限与 heal 阈值跟身份血量不一致。
            hp = int(hp)
            for p in range(3):
                self.base_hp[p] = hp
                self.hp[p] = hp
        self.deck = ([SHA] * self.deck_comp[0] +
                     [SHAN] * self.deck_comp[1] +
                     [TAO] * self.deck_comp[2])
        self.rng.shuffle(self.deck)
        self.hands = {p: [self.deck.pop() for _ in range(self.start_hand)]
                      for p in range(3)}
        self.turn = 0
        self.cur = 0                      # 当前行动座位，主公先手
        self.winner = None                # None / seat_id / "draw"
        self.winning_identity = None      # ZHU/FAN/NEI / None
        self.last_event = ""
        self.discard = []
        return self

    def _alive(self):
        return [p for p in range(3) if self.hp[p] > 0]

    def view(self, player):
        """player 的可见信息：只看到自己的身份，不泄露他人身份。"""
        return {
            "me": player,
            "my_identity": self.identities[player],
            "hp": {p: self.hp[p] for p in range(3)},
            "alive": self._alive(),
            "my_hand": list(self.hands[player]),
            "deck_left": len(self.deck),
            "turn": self.turn,
            "cur": self.cur,
        }

    # ---------------------------------------------------------------- 胜负
    def _check_winner(self):
        alive = self._alive()
        zhu_seat = 0
        zhu_alive = self.hp[zhu_seat] > 0

        fan_alive = any(self.identities[p] == FAN and self.hp[p] > 0 for p in alive)
        nei_alive = any(self.identities[p] == NEI and self.hp[p] > 0 for p in alive)

        if zhu_alive and not fan_alive and not nei_alive:
            self.winner = zhu_seat
            self.winning_identity = ZHU
            return True
        if not zhu_alive:
            if fan_alive:
                # 反贼杀主公（哪怕反贼自己也死了，经典规则反贼胜）
                fan_seats = [p for p in range(3)
                             if self.identities[p] == FAN]
                self.winner = fan_seats[0] if fan_seats else alive[0] if alive else -1
                self.winning_identity = FAN
                return True
            # 没反贼了，内奸杀主公 → 内奸胜
            nei_seats = [p for p in range(3)
                         if self.identities[p] == NEI]
            self.winner = nei_seats[0] if nei_seats else alive[0] if alive else -1
            self.winning_identity = NEI
            return True
        return False

    # ---------------------------------------------------------------- 对局
    def run(self, bots):
        """跑完整一局。bots: {seat: Bot}。
        返回 {'winner': seat_id|-1, 'winning_identity': str, 'turns': int}。
        返回前已 reset。"""
        self.reset()
        while self.turn < self.max_turns and self.winner is None:
            p = self.cur
            # 摸牌
            for _ in range(self.draw):
                if not self.deck:
                    self._reshuffle()
                if self.deck:
                    self.hands[p].append(self.deck.pop())
            # 出牌阶段
            while self.winner is None:
                legal = self._legal(p)
                if not legal:
                    break
                act = bots[p].decide(self, p)
                if act[0] == "pass" or act not in legal:
                    break
                self._apply(p, act, bots)
                if self._check_winner():
                    break
            # 弃牌
            while len(self.hands[p]) > self.hp[p] and self.hands[p]:
                c = self.hands[p].pop(self.rng.randrange(len(self.hands[p])))
                self.discard.append(c)
            # 换手（下一个活着的座位）
            nxt = (p + 1) % 3
            while nxt != p and self.hp[nxt] <= 0:
                nxt = (nxt + 1) % 3
            self.cur = nxt
            self.turn += 1

        if self.winner is None:
            self.winner = -1
            self.winning_identity = None
        return {"winner": self.winner,
                "winning_identity": self.winning_identity,
                "turns": self.turn}

    def _reshuffle(self):
        if not self.discard:
            return
        self.deck = self.discard
        self.discard = []
        self.rng.shuffle(self.deck)

    def _legal(self, p):
        hand = self.hands[p]
        acts = []
        alive_others = [q for q in range(3) if q != p and self.hp[q] > 0]
        if SHA in hand and alive_others:
            for q in alive_others:
                acts.append(("sha", q))
        if TAO in hand and self.hp[p] < self.base_hp[p]:
            acts.append(("tao", p))
        acts.append(("pass",))
        return acts

    def _apply(self, p, act, bots):
        if act[0] == "sha":
            tgt = act[1]
            resp = bots[tgt].decide_shan(self, tgt)
            if resp[0] == "shan" and SHAN in self.hands[tgt]:
                idx = self.hands[tgt].index(SHAN)
                self.discard.append(self.hands[tgt].pop(idx))
                self.last_event = "sha blocked by %d" % tgt
            else:
                self.hp[tgt] -= 1
                self.last_event = "sha hit %d" % tgt
            idx = self.hands[p].index(SHA)
            self.discard.append(self.hands[p].pop(idx))
        elif act[0] == "tao":
            self.hp[p] = min(self.base_hp[p], self.hp[p] + 1)
            idx = self.hands[p].index(TAO)
            self.discard.append(self.hands[p].pop(idx))


class Bot:
    """基因→策略。确定性由基因阈值 + 环境 rng 共同决定。"""

    def __init__(self, gene):
        import numpy as np
        self.g = np.asarray(gene, dtype=float)
        self.aggr, self.guard, self.heal, self.dist = self.g

    def decide(self, env, player):
        view = env.view(player)
        my_id = view["my_identity"]
        hand = view["my_hand"]
        alive_others = [q for q in view["alive"] if q != player]

        # ---- 被杀：是否出闪 ----
        # decide 会被 _apply 在被杀时调用，此时无法从 act 直接知晓。
        # 约定：若 env.last_event 为 "sha hit <player>"，说明刚被杀且未闪；
        # 但 run 里 _apply 先调 decide 判是否闪，所以这里需由 _apply 传信号。
        # 为保持接口简单，_apply 里被杀时会把 opp_sha=True 标记（见下方 decide 的专用分支）。

        # 桃：血量低于阈值才用
        if TAO in hand:
            thr = max(1.0, env.base_hp[player] * (1.0 - self.heal * 0.8))
            if view["hp"][player] <= thr:
                return ("tao", player)

        # 杀：选目标
        if SHA in hand and alive_others:
            p_sha = min(1.0, 0.5 * self.aggr + 0.5 * self.dist)
            want_sha = (self.aggr > 0.5) or (env.rng.random() < p_sha)
            if want_sha:
                tgt = self._pick_target(my_id, alive_others, view)
                return ("sha", tgt)

        return ("pass",)

    def decide_shan(self, env, player):
        """被杀时是否出闪（供 _apply 调用）。"""
        if SHAN not in env.hands[player]:
            return ("hit",)
        if self.guard > 0.5 or env.rng.random() < self.guard:
            return ("shan",)
        return ("hit",)

    # ---------------------------------------------------------------- 目标选择（身份相关）
    def _pick_target(self, my_id, candidates, view):
        """根据自己身份选攻击目标。"""
        hp = view["hp"]
        if my_id == FAN:
            # 反贼的唯一胜利条件 = 主公死亡，所以只要主公活着就锁定座位 0。
            #
            # 踩过的坑：原来写的是「杀血量最高的」（想当然以为血最多的就是主公）。
            # 但主公残血后血量会低于反贼/内奸，max() 就把目标漂到别人身上，
            # 主公反而缓过来 —— 实测 60 局反贼 0 胜，根因就是这条。
            # 胜利条件单一的身份，攻击目标不该随血量漂移。
            if 0 in candidates:
                return 0
            return candidates[0]
        if my_id == NEI:
            # 内奸：先杀血量最低的（假装杀非主公），只剩主公时杀主公
            non_zhu = [q for q in candidates if q != 0]
            if non_zhu:
                return min(non_zhu, key=lambda q: hp[q])
            return 0
        # 主公：随机攻击（不知道谁是反贼/内奸）
        # 注意：此处不持有 env.rng，用 Python 全局 random；
        # 如需完全可复现，可在 Bot 级别自备 rng。
        import random
        return random.choice(candidates)

    # ---- 被杀时是否出闪（供 _apply 调用）----
