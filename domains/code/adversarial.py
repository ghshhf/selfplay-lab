"""adversarial.py —— 对抗性出题：世界盯着冠军基因，挑克制它的题。

为什么需要它
------------
code 域跑了 2500+ 代，冠军 fit 长期 =1.000（99.6% 的记录满分），选择压力归零。
根因之一是出题是**纯随机**的：

    fam = rng.choice(GOLDEN)
    bp  = rng.choice(fam["bug_patterns"])

随机出题对「暴力撞」型冠军几乎无压力——题简单，撞也能撞对。要恢复梯度，
世界必须**针对冠军的行为倾向**出题：冠军不爱读码，就出必须通读才改得对的题；
冠军爱大范围乱改，就出必须精准改、大改会破坏不变量的题。

做法
----
每个 bug 模式带一个 counters 标签，声明它克制冠军的哪个基因维度，以及克制
的是「偏高」还是「偏低」：

    {"read_first": -0.9}   冠军 read_first 越低，这题越该出（必须读码才能改对）
    {"refactor":   +0.6}   冠军 refactor 越高，这题越该出（乱重构会破坏不变量）

    weight = 1 + ALPHA * Σ 强度 × 偏离度

偏离度由基因在 BOUNDS 内的归一化位置算出。于是「克制冠军」的题被选中概率更高，
世界从静态难度旋钮升级为对抗性出题者。

标签分四类，依据是 bug 的**技术机制**（不是拍脑袋）：
  A · 契约/隐含需求 —— 修复所需信息不在代码里，只在测试或需求描述里
  B · 不变量/状态   —— 修复点在一处，但周围有必须保持的结构
  C · 边界/基准     —— 差一个字符就错，考验精确
  D · 算法本身难    —— 必须真正写对，撞不出来（dijkstra 已被探针验证）

诚实边界（重要）
----------------
counters 是**人工标注的假设**，不是机器验证过的事实。上面每一条"这题能克制
某类策略"都只是推理。它到底有没有效，由 probe_adversarial.py 的对照实验裁定：
随机出题 vs 对抗出题，各跑 N 局，比冠军成功率。若实验证明无效，本文档会被改写，
不会硬撑。
"""
import numpy as np

# 出题加权强度。0 = 退化成随机出题（对照）。
ALPHA = 2.0

# ---------------------------------------------------------------------------
# 反制标签表：(族 id, bug 模式名) -> {基因名: 带符号强度}
#   负值 = 冠军该维度越低，越被这题克制
#   正值 = 冠军该维度越高，越被这题克制
# ---------------------------------------------------------------------------
COUNTERS = {
    # ---- A 类 · 契约/隐含需求：代码看不出来，只有测试/需求能告诉你 ----
    # 专治 read_first 低（不爱读码、不读测试）的冠军
    ("calc", "divide_none"):        {"test_cover": -0.9, "read_first": -0.6},
    ("bisect", "wrong_return"):     {"test_cover": -0.9, "read_first": -0.6},
    ("dedupe", "unordered"):        {"read_first": -0.9, "test_cover": -0.5},
    ("palindrome", "no_lower"):     {"read_first": -0.9, "test_cover": -0.4},
    ("palindrome", "keep_symbol"):  {"read_first": -0.9, "test_cover": -0.4},

    # ---- B 类 · 不变量/状态：改错地方会破坏结构 ----
    # 专治 refactor/edit_scope 高（爱大改、爱重构）的冠军
    ("lru", "no_reorder"):          {"refactor": 0.8, "edit_scope": 0.5},
    ("lru", "wrong_capacity"):      {"edit_scope": 0.8},
    ("stack", "no_close_check"):    {"refactor": 0.6, "read_first": -0.5},
    ("stack", "wrong_pair"):        {"edit_scope": 0.6, "read_first": -0.5},

    # ---- C 类 · 边界/基准：差一个字符就错 ----
    # 专治 edit_scope 高（大改容易越界）的冠军
    ("bisect", "off_by_one"):       {"edit_scope": 0.9},
    ("dedupe", "off_by_one"):       {"edit_scope": 0.9},
    ("fib", "base_wrong"):          {"edit_scope": 0.7, "read_first": -0.5},
    ("fib", "off_by_one"):          {"edit_scope": 0.8},

    # ---- D 类 · 算法本身难：必须真正写对，撞不出来 ----
    # 专治 retry_cap 高（靠重试撞运气）的冠军
    ("topo", "dup_indeg"):          {"read_first": -0.9, "retry_cap": 0.5},
    ("topo", "no_cycle_check"):     {"read_first": -0.8, "retry_cap": 0.5},
    ("dijkstra", "dist_off_by_one"): {"read_first": -1.0, "retry_cap": 0.6},
    ("merge", "no_merge"):          {"read_first": -0.7},
    ("merge", "boundary_off"):      {"edit_scope": 0.7, "read_first": -0.5},
    ("quicksort", "wrong_bound"):   {"edit_scope": 0.8, "read_first": -0.6},
    ("bfs", "missing_plus_one"):    {"read_first": -0.8},
    ("kadane", "init_zero"):        {"read_first": -0.8, "edit_scope": 0.5},
    ("kadane", "no_reset"):         {"read_first": -0.8, "edit_scope": 0.5},
}

# 人类可读解释：给日志用，说明"这题为什么被选中"
COUNTER_TEXT = {
    ("read_first", -1): "必须通读代码/测试才能定位",
    ("test_cover", -1): "只有跑测试才知道契约",
    ("refactor", 1): "乱重构会破坏不变量",
    ("edit_scope", 1): "大范围改会误伤边界",
    ("retry_cap", 1): "重试撞不出来，必须一次想对",
    ("doc_first", 1): "文档会误导，得看真实行为",
}


def _text_of(gname, spec):
    """把一个 counter 标注翻译成人话。"""
    if isinstance(spec, (tuple, list)):
        direction, strength = spec
    else:
        strength = float(spec)
        direction = -1.0 if strength >= 0 else 1.0
        strength = abs(strength)
    d = -1 if direction < 0 else 1
    return COUNTER_TEXT.get((gname, d), "%s %s" % (gname, "偏低" if d < 0 else "偏高"))


def normalize(champ, bounds):
    """冠军基因 {基因名: 原始值} → 按 BOUNDS 归一化到 [0,1]。

    bounds: {基因名: (lo, hi)}
    """
    out = {}
    for gname, val in champ.items():
        lo, hi = bounds.get(gname, (0.0, 1.0))
        try:
            hi = float(hi)
            lo = float(lo)
            v = float(val)
        except (TypeError, ValueError):
            out[gname] = 0.5
            continue
        n = 0.5 if hi <= lo else (v - lo) / (hi - lo)
        out[gname] = min(1.0, max(0.0, n))
    return out


def deviation(norm_val, direction):
    """偏离度：direction<0 表示「值越低越被克制」，>0 表示「值越高越被克制」。

    返回 0~1，越大说明冠军在这个维度上越"偏科"，越该被这题克制。
    """
    n = min(1.0, max(0.0, float(norm_val)))
    return (1.0 - n) if direction < 0 else n


def weights_for_family(champ, fam, bounds, alpha=ALPHA):
    """算某族下每个 bug 模式的出题权重。克制当前冠军的权重高。

    champ: {基因名: 原始值}；为空 dict 时全部返回 1.0（= 随机出题）。
    返回 np.ndarray，长度 = len(fam["bug_patterns"])。
    """
    nz = normalize(champ, bounds) if champ else {}
    ws = []
    for bp in fam["bug_patterns"]:
        counters = bp.get("counters") or COUNTERS.get((fam["id"], bp["name"]))
        s = 0.0
        if counters and nz:
            for gname, spec in counters.items():
                if gname not in nz:
                    continue
                if isinstance(spec, (tuple, list)):
                    direction, strength = spec
                else:
                    strength = float(spec)
                    direction = -1.0 if strength >= 0 else 1.0
                    strength = abs(strength)
                s += strength * deviation(nz[gname], direction)
        ws.append(1.0 + alpha * s)
    return np.asarray(ws, dtype=float)


_default_bounds_cache = {}


def default_bounds():
    """懒加载 env 的 BOUNDS，返回 {基因名: (lo, hi)}。

    为什么是懒加载：env → tasks → adversarial 是单向的，但 tasks 需要在**出题时**
    知道基因上下界。若在模块级 import env 会形成 env→tasks→env 循环，
    因此放到运行时再取（此时 env 早已加载完毕）。
    """
    if not _default_bounds_cache:
        import env  # 延迟到调用时，避免 import 环
        for i, name in enumerate(env.GENE_NAMES):
            _default_bounds_cache[name] = tuple(env.BOUNDS[i])
    return _default_bounds_cache


def pick_bug_pattern(champ, fam, rng, bounds=None, alpha=ALPHA):
    """按权重挑一个 bug 模式。

    champ 为空 → 退化为均匀随机（与改造前行为一致，保证可对照）。
    bounds 为空 → 用 env 的 BOUNDS（懒加载）。
    rng 可以是 random.Random 或 numpy Generator。
    """
    if bounds is None:
        bounds = default_bounds()
    bps = fam.get("bug_patterns") or []
    if not bps:
        return None
    if not champ:
        return rng.choice(bps) if hasattr(rng, "choice") else bps[0]

    w = weights_for_family(champ, fam, bounds, alpha=alpha)
    tot = float(w.sum())
    if tot <= 0:
        return rng.choice(bps) if hasattr(rng, "choice") else bps[0]

    p = (w / tot).tolist()
    if hasattr(rng, "choices"):
        return rng.choices(bps, weights=p, k=1)[0]
    # numpy Generator 没有 choices，用 choice + p
    idx = int(rng.choice(len(bps), p=np.asarray(p, dtype=float)))
    return bps[idx]


def explain_pick(champ, fam, bp, bounds=None):
    """解释这次出题在克制冠军的什么倾向（给日志用）。"""
    counters = bp.get("counters") or COUNTERS.get((fam["id"], bp["name"]))
    if not counters or not champ:
        return ""
    if bounds is None:
        bounds = default_bounds()
    nz = normalize(champ, bounds)
    parts = []
    for gname, spec in counters.items():
        if gname in nz:
            parts.append("%s(%.2f)→%s" % (gname, nz[gname], _text_of(gname, spec)))
    return " ".join(parts)
