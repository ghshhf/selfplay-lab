"""五子棋：规则引擎 + 可进化评估函数（增量版，纯 numpy / 纯 CPU）

低配版的"低"在三个地方：
  1. 规则用代码写死（裁判绝对公正，赖不掉）
  2. 没有世界模型（盘面本身就是真理，不需要"猜"）
  3. 纯 CPU、纯 numpy，一局几十毫秒 —— 只有这么快，进化才谈得上"大规模"

核心技巧（这决定了速度）：
  每个 5 长窗口编码成三进制数 code = Σ v_k·3^(4-k)，v∈{0空,1黑,2白}。
  落在格子 c 上时，包含 c 的窗口新编码 = 旧编码 + POW3[c 在窗口内的位置] × 落子方。
  —— 不用重算，直接加。于是落子只碰 ~35 个窗口，评估退化成 6 维直方图的点积。
"""

from __future__ import annotations

import numpy as np

N = 15
K = 5
CELLS = N * N
NWIN = 572

POW3 = (3 ** np.arange(K - 1, -1, -1)).astype(np.int32)  # [81,27,9,3,1]
FULL_ME = int(POW3.sum())          # 121 = 窗口全黑
FULL_OPP = int((POW3 * 2).sum())   # 242 = 窗口全白

# ---------------------------------------------------------------- 窗口索引
def _windows() -> tuple[np.ndarray, np.ndarray]:
    """所有 5 长窗口（横/竖/两斜），返回 (窗口格子索引, 每格在窗口内的位权)。"""
    idx, pw = [], []
    def add(cells, poses):
        idx.append(cells)
        pw.append(POW3[list(poses)])

    for r in range(N):
        for c in range(N - K + 1):
            add([r * N + c + k for k in range(K)], range(K))
    for c in range(N):
        for r in range(N - K + 1):
            add([(r + k) * N + c for k in range(K)], range(K))
    for r in range(N - K + 1):
        for c in range(N - K + 1):
            add([(r + k) * N + c + k for k in range(K)], range(K))
    for r in range(N - K + 1):
        for c in range(K - 1, N):
            add([(r + k) * N + c - k for k in range(K)], range(K))
    return np.asarray(idx, dtype=np.int32), np.asarray(pw, dtype=np.int32)


WIDX, WPOS = _windows()          # WIDX: (572,5) 格子; WPOS: (572,5) 位权
NWIN = WIDX.shape[0]


def _csr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把「格子 → 参与哪些窗口」压成 CSR，供批量向量化取用。"""
    pairs = []                    # (cell, win, pos)
    for wi in range(NWIN):
        for k in range(K):
            pairs.append((int(WIDX[wi, k]), wi, k))
    pairs.sort()
    ptr = np.zeros(CELLS + 1, dtype=np.int32)
    wid = np.zeros(len(pairs), dtype=np.int32)
    pos = np.zeros(len(pairs), dtype=np.int32)
    for i, (c, wi, k) in enumerate(pairs):
        ptr[c + 1] = i + 1
        wid[i] = wi
        pos[i] = k
    for c in range(CELLS):
        if ptr[c + 1] < ptr[c]:
            ptr[c + 1] = ptr[c]
    return ptr, wid, POW3[pos]


CELL_PTR, CELL_WID, CELL_POW = _csr()
NEI_OFFSETS = None


def _build_nei() -> np.ndarray:
    """每格半径 2 的邻居（用于增量维护候选点）。"""
    out = np.zeros((CELLS, 24), dtype=np.int32)
    for r in range(N):
        for c in range(N):
            n = 0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < N and 0 <= cc < N:
                        out[r * N + c, n] = rr * N + cc
                        n += 1
            # 补齐用格子自身：near 会多计，但该格已有子，不会污染候选点
            out[r * N + c, n:] = r * N + c
    return out


NEI = _build_nei()
NEI_COUNT = np.array([len({int(x) for x in row if int(x) != row[0] or True}) for row in NEI])


def _build_tables():
    """code → 该窗口里「黑」的等级 / 「白」的等级（0 无 .. 5 连五）"""
    t1 = np.zeros(3 ** K, dtype=np.int8)
    t2 = np.zeros(3 ** K, dtype=np.int8)
    for code in range(3 ** K):
        c, cnt = code, [0, 0, 0]
        for _ in range(K):
            cnt[c % 3] += 1
            c //= 3
        # cnt[0]=空 cnt[1]=黑 cnt[2]=白
        if cnt[2] == 0:
            t1[code] = cnt[1]
        if cnt[1] == 0:
            t2[code] = cnt[2]
    return t1, t2


TBL1, TBL2 = _build_tables()

# ---------------------------------------------------------------- 权重
W_DIM = 7   # 等级 0..5 的分值 + 防守倍率
W_INIT = np.array([0.0, 1.0, 12.0, 120.0, 1200.0, 200000.0, 1.15])


def clip_w(w) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).copy()
    w[0] = 0.0
    w[1] = np.clip(w[1], 0.0, 20.0)
    w[2] = np.clip(w[2], 0.0, 400.0)
    w[3] = np.clip(w[3], 0.0, 6000.0)
    w[4] = np.clip(w[4], 0.0, 2e5)
    w[5] = np.clip(w[5], 1e4, 1e7)
    w[6] = np.clip(w[6], 0.5, 3.0)
    return w


# ---------------------------------------------------------------- 棋盘
class Board:
    __slots__ = ("b", "moves", "last", "codes", "near", "_stack")

    def __init__(self):
        self.b = np.zeros(CELLS, dtype=np.int8)
        self.codes = np.zeros(NWIN, dtype=np.int32)
        self.near = np.zeros(CELLS, dtype=np.int8)
        self.moves = 0
        self.last = -1
        self._stack = []

    # -- 落子：返回是否形成五连 -------------------------------------
    def put(self, cell: int, player: int) -> bool:
        cell = int(cell)
        lo, hi = CELL_PTR[cell], CELL_PTR[cell + 1]
        wid = CELL_WID[lo:hi]
        old_c = self.codes[wid]
        new_c = old_c + CELL_POW[lo:hi] * player
        self.codes[wid] = new_c
        self.b[cell] = player
        nb = NEI[cell]
        self.near[nb] += 1
        self._stack.append((cell, wid, old_c, nb))
        self.moves += 1
        self.last = cell
        return bool((new_c == (FULL_ME if player == 1 else FULL_OPP)).any())

    def undo(self):
        cell, wid, old_c, nb = self._stack.pop()
        self.codes[wid] = old_c
        self.b[cell] = 0
        self.near[nb] -= 1
        self.moves -= 1
        self.last = -1

    # -- 评估 --------------------------------------------------------
    def evaluate(self, player: int, w: np.ndarray) -> float:
        c = self.codes
        w6 = w[6]
        if player == 1:
            return float(w[TBL1[c]].sum() - w6 * w[TBL2[c]].sum())
        return float(w[TBL2[c]].sum() - w6 * w[TBL1[c]].sum())

    # -- 候选点 ------------------------------------------------------
    def candidates(self) -> np.ndarray:
        if self.moves == 0:
            return np.array([(N // 2) * N + N // 2], dtype=np.int32)
        mask = (self.near > 0) & (self.b == 0)
        return np.flatnonzero(mask).astype(np.int32)

    # -- 批量排序：一次性算所有候选的增量收益 ------------------------
    def rank(self, cands: np.ndarray, player: int, w: np.ndarray) -> np.ndarray:
        starts = CELL_PTR[cands]
        counts = CELL_PTR[cands + 1] - starts
        keep = counts > 0                      # 极端边缘点可能不参与任何窗口
        if not keep.all():
            cands, starts, counts = cands[keep], starts[keep], counts[keep]
        total = int(counts.sum())
        if total == 0:
            return cands
        seg = np.cumsum(counts) - counts        # 每段在扁平数组里的起点
        flat = np.repeat(starts, counts) + (
            np.arange(total, dtype=np.int32) - np.repeat(seg, counts)
        )
        wid = CELL_WID[flat]
        old_c = self.codes[wid]
        new_c = old_c + CELL_POW[flat] * player
        if player == 1:
            d_my = (w[TBL1[new_c]] - w[TBL1[old_c]])
            d_op = (w[TBL2[new_c]] - w[TBL2[old_c]])
        else:
            d_my = (w[TBL2[new_c]] - w[TBL2[old_c]])
            d_op = (w[TBL1[new_c]] - w[TBL1[old_c]])
        sc = np.add.reduceat(d_my, seg) - w[6] * np.add.reduceat(d_op, seg)
        return cands[np.argsort(-sc)]
