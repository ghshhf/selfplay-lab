"""VCF（连续冲四）与威胁点搜索

为什么必须有它：
    原来的 net_pick 只查「一手成五」，等于完全不会算杀。
    Rapfi 在 300ms 里杀我们，靠的就是这个 —— 它算得出连续冲四，我们算不出。

为什么成本极低：
    冲四是「强制手」——对手只有 1~2 个必堵点，不能不理。
    所以哪怕搜 10 层，有效分支也只有几个。
    相比之下常规搜索每层 40 个分支，2 层就 1600 个，10 层根本不敢想。

为什么永远不会错：
    它只看规则（能不能成五），不吃任何评估函数。
    不猜、不学、不会因为训练不够而判错。
"""

from __future__ import annotations

import numpy as np

from gomoku import (CELL_PTR, CELL_WID, CELLS, FULL_ME, FULL_OPP, K, POW3,
                    WIDX, Board)

# ---------------------------------------------------------------- 预建表
# 位权 → 窗口内第几位
POW_POS = {int(p): i for i, p in enumerate(POW3)}

# 「4 我方 + 1 空」的窗口 → 那个空位的位权；不是这种形状则为 0
_FOUR_B = np.zeros(3 ** K, dtype=np.int32)
_FOUR_W = np.zeros(3 ** K, dtype=np.int32)

for _code in range(3 ** K):
    _v = []
    _c = _code
    for _i in range(K):
        _v.append(_c // int(POW3[_i]))
        _c %= int(POW3[_i])
    if _v.count(0) == 1:
        _k = _v.index(0)
        if _v.count(1) == K - 1:
            _FOUR_B[_code] = POW3[_k]
        elif _v.count(2) == K - 1:
            _FOUR_W[_code] = POW3[_k]

# 纯 Python 版索引缓存 —— numpy 单元素访问太慢，VCF 一局要查几万次
_CELL_WIDS = [CELL_WID[CELL_PTR[c]:CELL_PTR[c + 1]].tolist() for c in range(CELLS)]


# ---------------------------------------------------------------- 单点威胁
def threats_at(b: Board, cell: int, player: int) -> tuple[bool, set[int]]:
    """cell 是刚落下的一手（属于 player）。

    返回 (是否成五, 这手新造出的「四」的填空点集合)。
    判定只看 cell 参与的窗口 —— 只有这些窗口的编码变了。
    """
    wids = _CELL_WIDS[cell]
    codes = b.codes[wids]
    full = FULL_ME if player == 1 else FULL_OPP
    if full in codes:
        return True, set()
    tbl = _FOUR_B if player == 1 else _FOUR_W
    pw = tbl[codes]
    blocks = set()
    for i in np.nonzero(pw)[0]:
        blocks.add(int(WIDX[wids[i], POW_POS[int(pw[i])]]))
    return False, blocks


# ---------------------------------------------------------------- VCF
def find_vcf(b: Board, me: int, depth: int = 8, budget: int = 30000) -> list[int] | None:
    """找一条必胜的连续冲四线路。返回落子序列（第一手是我方要下的），无解返回 None。"""
    st = [budget]
    return _vcf(b, me, depth, st)


def _vcf(b: Board, me: int, depth: int, st: list[int]) -> list[int] | None:
    if st[0] <= 0 or depth <= 0:
        return None
    opp = 3 - me
    cands = b.candidates()
    if len(cands) == 0:
        return None

    # 1) 能一手成五，赢了
    for c in cands:
        c = int(c)
        if b.put(c, me):
            b.undo()
            return [c]
        b.undo()
        st[0] -= 1

    # 1.5) 对手已经能一手成五 —— 我方任何冲四都会被直接无视（他成五就赢了），
    #      所以这条线根本不存在。原来漏了这一步，会算出一堆走不通的「杀」。
    for c in cands:
        c = int(c)
        if b.put(c, opp):
            b.undo()
            return None
        b.undo()
        st[0] -= 1

    # 2) 收集所有能「成四」的点，双四优先
    fours = []
    for c in cands:
        c = int(c)
        b.put(c, me)
        st[0] -= 1
        _, blocks = threats_at(b, c, me)
        b.undo()
        if blocks:
            fours.append((len(blocks), c, blocks))
        if st[0] <= 0:
            return None
    if not fours:
        return None
    fours.sort(key=lambda x: (-x[0], x[1]))

    for nblk, c, blocks in fours:
        if st[0] <= 0:
            break
        if nblk >= 2:
            return [c]                     # 双四 / 活四：对手堵不完

        blk = min(blocks)
        b.put(c, me)
        b.put(blk, opp)
        st[0] -= 2
        ofive, ofour = threats_at(b, blk, opp)
        if ofive or ofour:
            b.undo()
            b.undo()
            continue                       # 对手反杀或反四，这条线断了

        rest = _vcf(b, me, depth - 1, st)
        b.undo()
        b.undo()
        if rest is not None:
            return [c] + rest
    return None


# ---------------------------------------------------------------- 防守
def danger_points(b: Board, player: int, topk: int = 3, min_level: int = 0) -> list[int]:
    """player 一子就能造出的「必杀点」——对手最该被提前堵的地方。

    判据（按危险度排序）：
      成五（99）> 双四/活四（空位数 ≥2，记 10+n）> 单四（空位数 1，记 1）
    min_level 用来筛：只要最危险那档就传 11，单四也算就传 0。
    """
    out = []
    for c in b.candidates():
        c = int(c)
        b.put(c, player)
        five, blocks = threats_at(b, c, player)
        b.undo()
        if five:
            out.append((99, c))
        elif len(blocks) >= 2:
            out.append((10 + len(blocks), c))
        elif len(blocks) == 1:
            out.append((1, c))
    out = [x for x in out if x[0] >= min_level]
    out.sort(key=lambda x: -x[0])
    return [c for _, c in out[:topk]]


def five_points(b: Board, player: int) -> list[int]:
    """player 一手成五的落点（如果非空，说明已经拦不住了或必须立刻拦）"""
    out = []
    for c in b.candidates():
        c = int(c)
        if b.put(c, player):
            b.undo()
            out.append(c)
        else:
            b.undo()
    return out


# ---------------------------------------------------------------- 自测
if __name__ == "__main__":
    import time

    def show(b: Board):
        s = []
        for r in range(15):
            row = ""
            for c in range(15):
                v = b.b[r * 15 + c]
                row += ".XO"[v]
            s.append(row)
        return "\n".join(s)

    # 用例 1：空盘不该找到任何杀，且要快
    b = Board()
    t = time.time()
    r = find_vcf(b, 1, depth=8, budget=30000)
    print(f"空盘: {r}  {time.time()-t:.3f}s")
    assert r is None

    # 用例 2：黑方活三 OOO + 落子成双四
    #  行7：黑 (7,5)(7,6)(7,7)，落 (7,8) → 需要再看另一条线
    b = Board()
    for c in [(7, 4), (7, 5), (7, 6), (8, 4), (8, 5), (8, 6)]:
        pass
    # 造一个 VCF：黑在 (7,5)(7,6)(7,7)，白随便下，黑落(7,4)或(7,8)成四
    b = Board()
    for cell, p in [(( 7, 5), 1), ((0, 0), 2), ((7, 6), 1), ((0, 1), 2), ((7, 7), 1)]:
        b.put(cell[0] * 15 + cell[1], p)
    t = time.time()
    plan = find_vcf(b, 1, depth=8, budget=30000)
    print(f"活三局面: {plan}  {time.time()-t:.3f}s")
    print(show(b))

    # 用例 3：交叉双四 —— 黑 横向(7,4)(7,5)(7,6) + 纵向(4,7)(5,7)(6,7)
    b = Board()
    for cell, p in [((7, 4), 1), ((4, 7), 2), ((7, 5), 1), ((5, 7), 2), ((7, 6), 1), ((6, 7), 2)]:
        b.put(cell[0] * 15 + cell[1], p)
    t = time.time()
    plan = find_vcf(b, 1, depth=8, budget=30000)
    print(f"交叉双四: {plan}  {time.time()-t:.3f}s")
    assert plan, "这里必须找到杀"

    # 用例 4：把 plan 完整重放一遍，确认最后真的成五（序列必须可执行）
    def replay(b0: Board, me: int, plan: list[int]) -> bool:
        opp = 3 - me
        for mv in plan:
            if b0.b[mv] != 0:
                print(f"  ✗ 序列里的 {mv} 已经被占了")
                return False
            if b0.put(mv, me):
                b0.undo()
                return True                      # 这一手就成五，赢
            _, blocks = threats_at(b0, mv, me)
            if not blocks:
                print(f"  ✗ {mv} 既不成五也不成四，不是 VCF 手")
                return False
            if len(blocks) >= 2:
                # 活四：对手堵一头，我方填另一头
                blk = min(blocks)
                b0.put(blk, opp)
                other = [x for x in blocks if x != blk][0]
                assert b0.put(other, me), "活四竟然没成五"
                b0.undo()
                b0.undo()
                return True
            b0.put(min(blocks), opp)
        return False

    ok = replay(b, 1, plan)
    print(f"序列重放: {'✓ 确实能赢' if ok else '✗ 序列无法执行'}")
    assert ok

    # 用例 5：真双四（两个不同方向的四，各一个堵点）——用交叉点 (7,7)
    b = Board()
    for cell, p in [((7, 4), 1), ((7, 5), 1), ((7, 6), 1),
                    ((4, 7), 1), ((5, 7), 1), ((6, 7), 1)]:
        b.put(cell[0] * 15 + cell[1], p)
    five, blocks = (b.put(7 * 15 + 7, 1), threats_at(b, 7 * 15 + 7, 1))[1]
    b.undo()
    print(f"交叉点(7,7) 棋型: 成五={five} 堵点={sorted(blocks)}")
    assert not five and len(blocks) >= 2, "应为一子多四，堵不完才算杀"

    print("自测全部通过")
