"""
MiniDuel —— 两个小人对打的最小环境（纯 numpy，无外部依赖）

设计原则（从火柴人那次失败里学来的）：
  1. 位置就是唯一状态，动作直接产生速度，不做约束求解 —— 没有约束就不会崩。
  2. 边界只做位置钳制 + 清零速度，不做"穿过去再推回来"的投影。
  3. 所有时间量用帧计，不用秒，避免 dt 混用。
  4. 伤害只减不增，每局结束能对账（总伤害 == 初始血量 - 剩余血量）。

一帧对应 1/60 秒，一局最长 30 秒。

v2（随机到底）：环境不再写死。场地宽度、障碍墙、道具、陷阱、双方武器都是
可配置的。默认配置（无武器、无墙、无道具、8m 场地）完全等价于 v1，保证
旧采集脚本（stream_duel / collect_duel）不坏；v2 的新能力只由 evolve 路径启用。
"""

import numpy as np

FPS = 60
ARENA_HALF = 4.0      # 默认场地 x ∈ [-4, 4]（可被 env_cfg 覆盖）
BODY_GAP = 0.62       # 两人最小间距（身位），不允许穿模
MAX_HP = 100.0

# ---------------- 动作 ----------------
FWD, BACK, PUNCH, KICK, BLOCK, IDLE = range(6)
ACTIONS = (FWD, BACK, PUNCH, KICK, BLOCK, IDLE)
ACTION_NAMES = ("FWD", "BACK", "PUNCH", "KICK", "BLOCK", "IDLE")

# 招式表: (前摇, 活跃, 后摇, 攻击距离, 伤害) —— 这是 none 武器的默认表，
# 同时作为外部（bot/evolve）向后兼容的引用。具体 Fighter 用 WEAPONS[weapon]。
MOVES = {
    PUNCH: (6, 3, 9, 1.15, 7.0),
    KICK: (11, 3, 16, 1.50, 11.0),
}

BLOCK_DAMAGE_MUL = 0.28      # 格挡成功只吃 28% 伤害（none 武器；shield 会更硬）
HURT_STUN = 10               # 受击硬直帧数
WALK_SPEED = 2.6             # 走位速度 m/s
KNOCKBACK = 2.2              # 受击击退初速度 m/s
FRICTION = 6.0               # 击退衰减（1/s）

# ---------------- 武器 ----------------
# 每个武器改的是 Fighter 的"物理属性"，不动 Bot 的决策结构 —— 这样进化出的
# 策略是"无论拿到什么武器都能打"的鲁棒策略，而不是"我拿了长矛所以远战"的写死逻辑。
#   attacks : 该武器可用的招式（替代全局 MOVES）
#   block_mul: 格挡减伤系数（越小越硬）
#   speed_mul: 移动速度系数
#   hurt_add : 被命中时额外硬直帧
#   label   : 人类可读
WEAPONS = {
    "none":    {"attacks": {PUNCH: (6, 3, 9, 1.15, 7.0),
                            KICK: (11, 3, 16, 1.50, 11.0)},
                "block_mul": 0.28, "speed_mul": 1.0, "hurt_add": 0, "label": "徒手"},
    "polearm": {"attacks": {PUNCH: (8, 3, 10, 2.80, 9.0)},   # 长矛：只有长戳，射程 2.8
                "block_mul": 0.28, "speed_mul": 1.0, "hurt_add": 0, "label": "长矛"},
    "shield":  {"attacks": {PUNCH: (6, 3, 9, 1.15, 7.0),
                            KICK: (11, 3, 16, 1.50, 11.0)},
                "block_mul": 0.10, "speed_mul": 0.82, "hurt_add": 0, "label": "盾"},  # 更硬但慢
    "twin":    {"attacks": {PUNCH: (5, 3, 8, 1.10, 9.8),
                            KICK: (10, 3, 15, 1.45, 15.4)},
                "block_mul": 0.28, "speed_mul": 1.0, "hurt_add": 3, "label": "双刃"},  # 伤害×1.4 但挨打硬直+3
    "javelin": {"attacks": {PUNCH: (10, 3, 14, 4.00, 12.0)},  # 投枪：远程 4.0 但装填长
                "block_mul": 0.28, "speed_mul": 1.0, "hurt_add": 0, "label": "投枪"},
}
WEAPON_NAMES = tuple(WEAPONS.keys())   # 演化基因里的离散武器字段按这个顺序索引

# ---------------- 道具 / 陷阱 ----------------
# 道具：可拾取，碰到即获得整局 buff。Bot 通过 me 的感知字段能看到"最近道具方向/距离"，
# 进化出的 loot_bias 决定多积极去捡。
LOOT_TYPES = ("speed", "power", "armor", "heal")
LOOT_LABEL = {"speed": "疾行", "power": "狂暴", "armor": "护甲", "heal": "治疗"}
LOOT_SPEED_MUL = 1.30      # 拾取后移动速度 ×1.3
LOOT_POWER_MUL = 1.40      # 拾取后伤害 ×1.4
LOOT_ARMOR_MUL = 0.60      # 拾取后受击 ×0.6
LOOT_HEAL = 25.0           # 拾取后回血

# 陷阱：隐形踩到才触发（Bot 能通过 me 感知字段看到陷阱位置，进化出 avoid_bias 主动绕开）。
TRAP_TYPES = ("slow", "stun", "spike")
TRAP_LABEL = {"slow": "泥沼", "stun": "电网", "spike": "尖刺"}
TRAP_SLOW_FRAMES = 30      # slow：速度 ×0.5 持续帧
TRAP_STUN_FRAMES = 20      # stun：硬直帧
TRAP_SPIKE_DMG = 15.0      # spike：直接掉血
TRAP_REARM = 60            # 触发后重新布防的冷却帧（防同一个人反复踩）

# 状态编号（写进数据集，别用字符串撑表）
S_IDLE, S_WALK, S_ATTACK, S_BLOCK, S_HURT, S_KO = range(6)
STATE_NAMES = ("IDLE", "WALK", "ATTACK", "BLOCK", "HURT", "KO")


class Fighter:
    def __init__(self, x, facing, hp=MAX_HP, weapon="none"):
        self.x = float(x)
        self.vx = 0.0
        self.facing = facing          # +1 / -1
        self.hp = float(hp)
        self.weapon = weapon
        self.state = S_IDLE
        self.timer = 0                # 当前动作还剩多少帧
        self.move = None              # 正在出的招（PUNCH/KICK）
        self.phase = ""               # windup / active / recover
        self.cooldown = 0
        self.landed = False           # 本次出招是否已命中（防一拳多段）
        # v2 buff / debuff 标记
        self.speed_buff = 1.0         # 道具：疾行
        self.power_buff = 1.0         # 道具：狂暴
        self.armor_buff = 1.0         # 道具：护甲
        self.slow_timer = 0           # 陷阱：泥沼剩余帧
        # v2 感知字段（每帧由 Duel._perceive 写入，Bot 只读不写）
        self.loot_dx = 0.0            # 最近可拾取道具的相对 x（无则 0）
        self.loot_dist = 99.0         # 最近可拾取道具距离（无则大数）
        self.trap_dx = 0.0            # 最近陷阱相对 x（无则 0）
        self.trap_dist = 99.0         # 最近陷阱距离（无则大数）
        self.wall_to_opp = 0          # 与对手之间是否被墙隔开（0/1）
        self.wall_dx = 0.0            # 最近墙边缘相对 x（无则 0）

    # ---- 只读派生量 ----
    @property
    def alive(self):
        return self.hp > 0.0

    def can_act(self):
        return self.state not in (S_ATTACK, S_HURT, S_KO)

    def attacks(self):
        """该武器可用的招式表（替代全局 MOVES）。"""
        return WEAPONS[self.weapon]["attacks"]

    def reach_of(self, move):
        windup, active, recover, reach, dmg = self.attacks()[move]
        return reach

    def max_reach(self):
        return max(r[3] for r in self.attacks().values())

    def block_mul(self):
        return WEAPONS[self.weapon]["block_mul"]

    def speed_mul(self):
        return WEAPONS[self.weapon]["speed_mul"]

    def hurt_add(self):
        return WEAPONS[self.weapon]["hurt_add"]

    def obs(self, other):
        """归一化观测：自己 6 维 + 对手 6 维 + 相对量 3 维 = 15 维（v1 兼容）。
        v2 没改 obs 维度——dataset 那条采集链暂时不动；武器/道具信息走 step_raw 路径。"""
        return np.array([
            self.x / ARENA_HALF,
            self.vx / 4.0,
            self.hp / MAX_HP,
            self.state / 5.0,
            self.timer / 30.0,
            float(self.facing),
            other.x / ARENA_HALF,
            other.vx / 4.0,
            other.hp / MAX_HP,
            other.state / 5.0,
            other.timer / 30.0,
            float(other.facing),
            (other.x - self.x) / ARENA_HALF,
            abs(other.x - self.x) / ARENA_HALF,
            float(self.cooldown) / 20.0,
        ], dtype=np.float32)


class Duel:
    """两人对打环境。step 接收双方动作，返回 (obs0, obs1, r0, r1, done, info)。

    环境变量（env_cfg）默认值保证与 v1 完全等价：
      half     : 场地半宽（默认 4.0 → 8m 场地）
      walls    : 障碍墙列表 [(中心x, 半宽)]，默认空
      loot     : 道具列表 [{"x","type","taken"}]，默认空
      traps    : 陷阱列表 [{"x","type","cd"}]，默认空
      friction : 击退衰减（默认 6.0）
      knockback: 击退初速（默认 2.2）
    weapons   : (weapon0, weapon1) 双方武器名，默认 ("none","none")
    """

    def __init__(self, seed=None, hp=MAX_HP, max_frames=1800, start_gap=2.2,
                 env_cfg=None, weapons=("none", "none")):
        self.rng = np.random.default_rng(seed)
        self.hp0 = float(hp)
        self.max_frames = int(max_frames)
        self.start_gap = float(start_gap)
        self.weapons = tuple(weapons)
        # 环境配置（深拷贝，避免跨局串改）
        ec = dict(env_cfg) if env_cfg else {}
        self.half = float(ec.get("half", ARENA_HALF))
        self.walls = [tuple(w) for w in ec.get("walls", [])]
        self.friction = float(ec.get("friction", FRICTION))
        self.knockback = float(ec.get("knockback", KNOCKBACK))
        # 道具/陷阱每局重新随机生成（见 _spawn_hazards），这里只存模板
        self.loot_spec = list(ec.get("loot", []))     # [(x, type)] 模板
        self.trap_spec = list(ec.get("traps", []))    # [(x, type)] 模板
        self.reset()

    # ---------------- 重置 ----------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        gap = self.start_gap * float(self.rng.uniform(0.85, 1.15))
        # 出生点：在 [-half*0.6, half*0.6] 随机偏移，制造不同开局
        off = self.half * 0.6
        x0 = float(self.rng.uniform(-off, off)) - gap * 0.5
        x1 = float(self.rng.uniform(-off, off)) + gap * 0.5
        self.p0 = Fighter(x0, +1, self.hp0, self.weapons[0])
        self.p1 = Fighter(x1, -1, self.hp0, self.weapons[1])
        self.frames = 0
        self.events = []            # (frame, move, damage, blocked, attacker_idx, hit_dist)
        self.damage_dealt = [0.0, 0.0]
        self.heal_total = [0.0, 0.0]  # 各自回血总量（heal 道具），用于对账
        self.n_swings = [0, 0]      # 各自起手次数（只计真正进入出招的，空挥也算）
        # 道具/陷阱实例化（每局重置 taken/cd）
        self.loot = [{"x": float(x), "type": t, "taken": False}
                     for (x, t) in self.loot_spec]
        self.traps = [{"x": float(x), "type": t, "cd": 0}
                      for (x, t) in self.trap_spec]
        return self._obs()

    def _obs(self):
        return self.p0.obs(self.p1), self.p1.obs(self.p0)

    # ---------------- 主步进 ----------------
    def step(self, a0, a1):
        rewards, done, info = self._advance(a0, a1)
        o0, o1 = self._obs()
        return o0, o1, rewards[0], rewards[1], done, info

    def step_raw(self, a0, a1):
        """不构造观测的 step，返回 (r0, r1, done, info)。
        Bot 直接读 Fighter 属性，一个观测都没用上；_obs() 每帧建两个 15 维数组，
        cProfile 实测占 37%。进化循环跑几十万局，这 37% 不能白花。
        但 obs 维度对账/数据集那一侧仍走 step()。"""
        rewards, done, info = self._advance(a0, a1)
        return rewards[0], rewards[1], done, info

    # ---------------- 感知：把最近道具/陷阱写进 Fighter ----------------
    def _perceive(self):
        for f, loot_list, trap_list in ((self.p0, self.loot, self.traps),
                                        (self.p1, self.loot, self.traps)):
            best_l, best_ld = 0.0, 99.0
            for it in loot_list:
                if it["taken"]:
                    continue
                d = it["x"] - f.x
                ad = abs(d)
                if ad < best_ld:
                    best_ld, best_l = ad, d
            f.loot_dx, f.loot_dist = best_l, best_ld
            best_t, best_td = 0.0, 99.0
            for it in trap_list:
                d = it["x"] - f.x
                ad = abs(d)
                if ad < best_td:
                    best_td, best_t = ad, d
            f.trap_dx, f.trap_dist = best_t, best_td
            # 墙：最近墙边缘方向 + 与对手是否被墙隔开
            best_w = 0.0
            best_wd = 99.0
            for (wx, ww) in self.walls:
                for edge in (wx - ww, wx + ww):
                    d = edge - f.x
                    ad = abs(d)
                    if ad < best_wd:
                        best_wd, best_w = ad, d
            f.wall_dx, f.wall_dist = best_w, best_wd
            f.wall_to_opp = 1 if self._wall_blocks(f.x, (self.p1.x if f is self.p0 else self.p0.x)) else 0

    # ---------------- 危险物：拾取道具 / 触发陷阱 ----------------
    def _hazards(self):
        for f, idx in ((self.p0, 0), (self.p1, 1)):
            # 道具：踩到即拾取（触发半径 0.6*BODY_GAP，覆盖身位分离后两人离道具的最小距离）
            for it in self.loot:
                if it["taken"]:
                    continue
                if abs(it["x"] - f.x) < BODY_GAP * 0.6:
                    it["taken"] = True
                    if it["type"] == "speed":
                        f.speed_buff = LOOT_SPEED_MUL
                    elif it["type"] == "power":
                        f.power_buff = LOOT_POWER_MUL
                    elif it["type"] == "armor":
                        f.armor_buff = LOOT_ARMOR_MUL
                    elif it["type"] == "heal":
                        before = f.hp
                        f.hp = min(MAX_HP, f.hp + LOOT_HEAL)
                        real = f.hp - before
                        if real > 0.0:
                            self.heal_total[idx] += real   # 只有真实回血才记，满血时不记
            # 陷阱：踩到且已布防则触发
            for it in self.traps:
                if it["cd"] > 0:
                    it["cd"] -= 1
                    continue
                if abs(it["x"] - f.x) < BODY_GAP * 0.6:
                    it["cd"] = TRAP_REARM
                    if it["type"] == "slow":
                        f.slow_timer = TRAP_SLOW_FRAMES
                    elif it["type"] == "stun":
                        if f.state not in (S_ATTACK, S_KO):
                            f.state, f.timer = S_HURT, TRAP_STUN_FRAMES
                            f.vx, f.move, f.phase = 0.0, None, ""
                    elif it["type"] == "spike":
                        before = f.hp
                        f.hp = max(0.0, f.hp - TRAP_SPIKE_DMG)
                        real = before - f.hp
                        if real > 0.0:
                            self.damage_dealt[idx] += real   # 只有真实掉血才记（已死/被夹断时不记）
            if f.slow_timer > 0:
                f.slow_timer -= 1

    # ---------------- 主推进 ----------------
    def _advance(self, a0, a1):
        rewards = [0.0, 0.0]

        # 0) 感知（道具/陷阱方向写进 Fighter，供 Bot 决策）
        self._perceive()

        # 1) 双方各自解析动作
        for me, opp, act, idx in ((self.p0, self.p1, a0, 0), (self.p1, self.p0, a1, 1)):
            self._apply_action(me, opp, int(act), idx, rewards)

        # 2) 攻击活跃帧的命中判定
        for me, opp, idx in ((self.p0, self.p1, 0), (self.p1, self.p0, 1)):
            if me.state == S_ATTACK and me.phase == "active" and not me.landed:
                self._try_hit(me, opp, 1 - idx, rewards)

        # 3) 物理积分（位置 + 击退衰减 + 边界 + 墙 + 身位分离）
        self._integrate()

        # 4) 危险物（道具拾取 / 陷阱触发）
        self._hazards()

        # 5) 计时器推进
        for f in (self.p0, self.p1):
            self._tick(f)

        self.frames += 1

        # 6) 结束判定
        done = False
        winner = None
        if not self.p0.alive or not self.p1.alive:
            done = True
            if self.p0.alive and not self.p1.alive:
                winner = 0
            elif self.p1.alive and not self.p0.alive:
                winner = 1
            else:
                winner = -1                       # 同归于尽
        elif self.frames >= self.max_frames:
            done = True
            winner = 0 if self.p0.hp > self.p1.hp else (1 if self.p1.hp > self.p0.hp else -1)

        if done:
            for i, f in enumerate((self.p0, self.p1)):
                if not f.alive:
                    f.state = S_KO
            if winner == 0:
                rewards[0] += 100.0; rewards[1] -= 100.0
            elif winner == 1:
                rewards[1] += 100.0; rewards[0] -= 100.0

        info = {
            "frames": self.frames,
            "hp": (self.p0.hp, self.p1.hp),
            "winner": winner,
            "x": (self.p0.x, self.p1.x),
            "damage": tuple(self.damage_dealt),
            "hits": len(self.events),
            "swings": tuple(self.n_swings),
        }
        return rewards, done, info

    # ---------------- 内部 ----------------
    def _wall_blocks(self, x0, x1):
        """x0 与 x1 之间是否被某堵墙隔开（挡移动也挡攻击）。

        正确语义：墙区间 [lo,hi] 落在两人连线内部 —— 两人分处墙两侧、墙在中间。
        旧逻辑用「墙完全覆盖两人区间」会漏掉窄墙：墙在 (-0.5,0.5)、两人站 -2 和 2，
        墙没"包住"他们，但物理上窄墙正好挡在中间，攻击该打不到。"""
        for (wx, ww) in self.walls:
            lo, hi = wx - ww, wx + ww
            if min(x0, x1) <= lo and max(x0, x1) >= hi:
                return True
        return False

    def _apply_action(self, me, opp, act, idx, rewards):
        if not me.alive:
            return
        # 朝向永远看向对手（格斗游戏惯例）
        me.facing = 1 if opp.x >= me.x else -1

        if me.state in (S_ATTACK, S_HURT):
            return                      # 出招中/硬直中，动作无效

        spd = WALK_SPEED * me.speed_mul() * me.speed_buff
        if me.slow_timer > 0:
            spd *= 0.5

        if act == FWD:
            me.vx = me.facing * spd
            me.state, me.timer = S_WALK, 2
        elif act == BACK:
            me.vx = -me.facing * spd * 0.85
            me.state, me.timer = S_WALK, 2
        elif act in me.attacks() and me.cooldown <= 0:
            windup, active, recover, reach, dmg = me.attacks()[act]
            self.n_swings[idx] += 1         # 真正起手才算，冷却中按键不算
            me.move = act
            me.state = S_ATTACK
            me.phase = "windup"
            me.timer = windup
            me.landed = False
            me.vx = 0.0
        elif act == BLOCK:
            me.state, me.timer = S_BLOCK, 2
            me.vx = 0.0
        else:
            me.state, me.timer = S_IDLE, 1
            me.vx *= 0.5

    def _try_hit(self, me, opp, opp_idx, rewards):
        _, _, _, reach, dmg = me.attacks()[me.move]
        dist = abs(opp.x - me.x)
        toward = (opp.x - me.x) * me.facing > 0
        if dist > reach or not toward:
            return
        # 墙挡攻击：两人之间被墙隔开则打不到（矮墙变掩体）
        if self._wall_blocks(me.x, opp.x):
            return
        if opp.state == S_KO or not opp.alive:
            return
        me.landed = True
        blocked = (opp.state == S_BLOCK)
        mul = me.block_mul() if blocked else 1.0
        want = dmg * me.power_buff * mul * opp.armor_buff
        hp_before = opp.hp
        opp.hp = max(0.0, opp.hp - want)
        real = hp_before - opp.hp           # 只记真实掉血，最后一击的溢出不算
        self.damage_dealt[1 - opp_idx] += real
        self.events.append((self.frames, me.move, real, blocked, 1 - opp_idx, dist))
        # 奖励：命中得分，被打扣分（格挡按比例）
        rewards[1 - opp_idx] -= real * 0.5
        rewards[opp_idx] += real * 0.8
        if not blocked:
            opp.state = S_HURT
            opp.timer = HURT_STUN + me.hurt_add()
            opp.vx = me.facing * self.knockback
            opp.phase = ""
            opp.move = None

    def _integrate(self):
        for f in (self.p0, self.p1):
            if f.state == S_HURT:
                # 击退速度衰减
                f.vx *= max(0.0, 1.0 - self.friction / FPS)
            elif f.state == S_IDLE:
                f.vx *= 0.85
            elif f.state in (S_ATTACK, S_BLOCK, S_KO):
                f.vx *= 0.6
            f.x += f.vx / FPS
            # 边界：钳位置 + 清速度（不做投影回推）
            if f.x < -self.half:
                f.x, f.vx = -self.half, 0.0
            elif f.x > self.half:
                f.x, f.vx = self.half, 0.0
            # 墙：单体不能进入墙区间（钳到最近墙边 + 清速度）
            for (wx, ww) in self.walls:
                lo, hi = wx - ww, wx + ww
                if lo < f.x < hi:
                    f.x = lo if (f.x - lo) < (hi - f.x) else hi
                    f.vx = 0.0
                f.x = float(np.clip(f.x, -self.half, self.half))

        # 身位分离：只挪位置，不动速度，避免注入能量。
        # 但两人之间被墙隔开时不做分离（墙挡着，不能把人推过墙）。
        d = self.p1.x - self.p0.x
        if abs(d) < BODY_GAP and not self._wall_blocks(self.p0.x, self.p1.x):
            sign = 1.0 if d >= 0 else -1.0
            push = (BODY_GAP - abs(d)) * 0.5
            self.p0.x -= sign * push
            self.p1.x += sign * push
            self.p0.x = float(np.clip(self.p0.x, -self.half, self.half))
            self.p1.x = float(np.clip(self.p1.x, -self.half, self.half))

    def _tick(self, f):
        if f.cooldown > 0:
            f.cooldown -= 1
        if f.state == S_ATTACK:
            f.timer -= 1
            if f.timer <= 0:
                windup, active, recover, _, _ = f.attacks()[f.move]
                if f.phase == "windup":
                    f.phase, f.timer = "active", active
                elif f.phase == "active":
                    f.phase, f.timer = "recover", recover
                else:
                    f.state, f.phase, f.move = S_IDLE, "", None
                    f.cooldown = 4
        elif f.state in (S_HURT, S_WALK, S_BLOCK, S_IDLE):
            f.timer -= 1
            if f.timer <= 0:
                f.state, f.timer = S_IDLE, 1

    # ---------------- 环境描述（写日志用） ----------------
    def describe_env(self):
        """返回当前环境的可打印摘要。"""
        w = []
        if abs(self.half - ARENA_HALF) > 0.01:
            w.append("场地%.1fm" % (self.half * 2))
        if self.walls:
            w.append("墙%d" % len(self.walls))
        if self.loot:
            w.append("道具%d[%s]" % (len(self.loot),
                                    "/".join(LOOT_LABEL[t["type"]] for t in self.loot)))
        if self.traps:
            w.append("陷阱%d[%s]" % (len(self.traps),
                                     "/".join(TRAP_LABEL[t["type"]] for t in self.traps)))
        if abs(self.friction - FRICTION) > 0.01 or abs(self.knockback - KNOCKBACK) > 0.01:
            w.append("物理")
        arms = []
        if self.weapons[0] != "none" or self.weapons[1] != "none":
            arms = ["%svs%s" % (WEAPONS[self.weapons[0]]["label"],
                                WEAPONS[self.weapons[1]]["label"])]
        return " ".join(arms + w) if (arms + w) else "基础"
