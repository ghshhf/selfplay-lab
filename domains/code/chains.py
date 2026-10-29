# -*- coding: utf-8 -*-
"""跨层隐含契约题 chain/bug family。

与 GOLDEN 的根本区别
--------------------
GOLDEN 是**单文件 + 单点字符串替换**：bug 注入在某个文件里，那个文件的
测试就直接报错。强笔只要认出它是「哪一类经典错误」（off-by-one / 漏了 None
检查 / 容量 > 写成 >=）就能改对 —— 那是背答案，不是读码。这也解释了为什么
冠军 read_first=0.08（几乎不读代码）仍能拿 92%。

CHAIN 改成本质不同的题型：

    core.py     底层实现，bug 在这里
    service.py  上层，依赖 core 的一个**没有写在文档里的不变量**
    tests.py    只通过 service 断言

关键约束：**注入 bug 之后，core 层的单元性质断言仍然全部通过**
（集合相同 / 长度正确 / 每行的初始值正确 / 分组内容正确）。
core 单独看是「正确」的，只有理解了 service 隐含依赖了什么，才知道它错了。

这条约束是有量化验收的，`selftest()` 会逐条检查，不是嘴上说说。

它还顺带带来一个新的可观测信号：editable 同时开放 core 与 service，
于是可以看出笔是去**修根因**还是**在上层打补丁**——这两种行为在 GOLDEN
里被单文件结构抹平了，在这里第一次能被看见。
"""

# ---------------------------------------------------------------------------
# 族 1：idxsort —— 「返回值有序」契约
# ---------------------------------------------------------------------------
def _idxsort_golden():
    return {
        "store.py": (
            "def normalize(values):\n"
            "    \"\"\"返回 values 的规范形式：去重后升序的列表。\n"
            "\n"
            "    契约（上层依赖）：返回值严格升序，可直接用于二分查找。\n"
            "    \"\"\"\n"
            "    return sorted(set(values))\n"
        ),
        "unit_core.py": (
            "# core 层单元测试 —— 只检验 store.normalize 自身的性质。\n"
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from store import normalize\n"
            "\n"
            "def test_keeps_members():\n"
            "    xs = normalize([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])\n"
            "    assert len(xs) == 10\n"
            "    assert set(xs) == set(range(10))\n"
            "\n"
            "def test_is_idempotent():\n"
            "    xs = normalize([3, 1, 2, 1, 3])\n"
            "    assert normalize(xs) == xs\n"
            "\n"
            "def test_handles_duplicates():\n"
            "    xs = normalize([2, 2, 2])\n"
            "    assert xs == [2]\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_keeps_members()\n"
            "    test_is_idempotent()\n"
            "    test_handles_duplicates()\n"
            "    print('All tests passed')\n"
        ),
        "lookup.py": (
            "from bisect import bisect_left\n"
            "\n"
            "from store import normalize\n"
            "\n"
            "class Lookup:\n"
            "    \"\"\"在一份规范化的值集合上做存在性/区间查询。\"\"\"\n"
            "\n"
            "    def __init__(self, values):\n"
            "        self.items = normalize(values)\n"
            "\n"
            "    def contains(self, v):\n"
            "        i = bisect_left(self.items, v)\n"
            "        return i < len(self.items) and self.items[i] == v\n"
            "\n"
            "    def between(self, lo, hi):\n"
            "        i = bisect_left(self.items, lo)\n"
            "        j = bisect_left(self.items, hi)\n"
            "        return [x for x in self.items[i:j] if lo <= x < hi]\n"
            "\n"
            "    def first_at_or_after(self, v):\n"
            "        i = bisect_left(self.items, v)\n"
            "        return self.items[i] if i < len(self.items) else None\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from store import normalize\n"
            "from lookup import Lookup\n"
            "\n"
            "RAW = [5, 1, 9, 3, 7, 2, 8, 4, 6, 0]\n"
            "\n"
            "# --- core 层单元性质：无论有没有 bug 都应成立 ---\n"
            "def test_core_unit_properties():\n"
            "    xs = normalize(RAW)\n"
            "    assert len(xs) == len(set(RAW))\n"
            "    assert set(xs) == set(RAW)\n"
            "    assert all(isinstance(x, int) for x in xs)\n"
            "\n"
            "# --- 跨层契约：只有 %s% 不该被破坏的性质 ---\n"
            "def test_lookup_finds_every_member():\n"
            "    lk = Lookup(RAW)\n"
            "    missing = [v for v in RAW if not lk.contains(v)]\n"
            "    assert not missing\n"
            "\n"
            "def test_between_is_complete():\n"
            "    lk = Lookup(RAW)\n"
            "    got = lk.between(2, 8)\n"
            "    want = sorted(v for v in set(RAW) if 2 <= v < 8)\n"
            "    assert got == want\n"
            "\n"
            "def test_first_at_or_after():\n"
            "    lk = Lookup(RAW)\n"
            "    assert lk.first_at_or_after(3) == 3\n"
            "    assert lk.first_at_or_after(6) == 6\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_core_unit_properties()\n"
            "    test_lookup_finds_every_member()\n"
            "    test_between_is_complete()\n"
            "    test_first_at_or_after()\n"
            "    print('All tests passed')\n"
        ),
    }


def _idxsort_tail_unsorted(files, rng):
    """最后一个不重复元素不参与排序：集合/长度/去重全对，但升序契约崩了。

    早期版本写成 `sorted(set(values[:-1])) + [values[-1]]`，会让重复输入直接暴露
    （len != len(set)），core 单测自己就能抓到 —— 那就退化成普通单文件题了。
    改成先按首次出现顺序去重、再只把最后一个元素置于末尾，集合语义因此保持不变。
    """
    files = dict(files)
    files["store.py"] = files["store.py"].replace(
        "    return sorted(set(values))\n",
        "    seen = []\n"
        "    for v in values:\n"
        "        if v not in seen:\n"
        "            seen.append(v)\n"
        "    if not seen:\n"
        "        return []\n"
        "    return sorted(seen[:-1]) + [seen[-1]]\n",
    )
    return files


# ---------------------------------------------------------------------------
# 族 2：matrow —— 「每次调用返回独立对象」契约
# ---------------------------------------------------------------------------
def _matrow_golden():
    return {
        "matrix.py": (
            "def make_row(width, fill=0):\n"
            "    \"\"\"返回长度为 width、全部填 fill 的新行。\n"
            "\n"
            "    契约（上层依赖）：每次调用都返回一份全新的列表，\n"
            "    调用方之间不会互相看见对方的修改。\n"
            "    \"\"\"\n"
            "    return [fill for _ in range(width)]\n"
            "\n"
            "\n"
            "def make_matrix(height, width, fill=0):\n"
            "    return [make_row(width, fill) for _ in range(height)]\n"
        ),
        "unit_core.py": (
            "# core 层单元测试 —— 只检验 matrix 自身的性质。\n"
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from matrix import make_row, make_matrix\n"
            "\n"
            "def test_row_shape_and_fill():\n"
            "    assert make_row(4, 7) == [7, 7, 7, 7]\n"
            "\n"
            "def test_matrix_shape():\n"
            "    m = make_matrix(3, 5)\n"
            "    assert len(m) == 3\n"
            "    assert all(len(r) == 5 for r in m)\n"
            "\n"
            "def test_matrix_initial_content():\n"
            "    m = make_matrix(3, 5, 2)\n"
            "    assert all(x == 2 for r in m for x in r)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_row_shape_and_fill()\n"
            "    test_matrix_shape()\n"
            "    test_matrix_initial_content()\n"
            "    print('All tests passed')\n"
        ),
        "board.py": (
            "from matrix import make_matrix\n"
            "\n"
            "class Board:\n"
            "    \"\"\"一块可写的格子板。\"\"\"\n"
            "\n"
            "    def __init__(self, h, w):\n"
            "        self.cells = make_matrix(h, w)\n"
            "\n"
            "    def set(self, r, c, v):\n"
            "        self.cells[r][c] = v\n"
            "\n"
            "    def get(self, r, c):\n"
            "        return self.cells[r][c]\n"
            "\n"
            "    def count_equal(self, v):\n"
            "        return sum(1 for row in self.cells for x in row if x == v)\n"
            "\n"
            "    def row_sum(self, r):\n"
            "        return sum(self.cells[r])\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from matrix import make_row, make_matrix\n"
            "from board import Board\n"
            "\n"
            "# --- core 层单元性质 ---\n"
            "def test_core_shape():\n"
            "    m = make_matrix(3, 4)\n"
            "    assert len(m) == 3\n"
            "    assert all(len(r) == 4 for r in m)\n"
            "    assert all(x == 0 for r in m for x in r)\n"
            "\n"
            "def test_row_is_fresh():\n"
            "    r1 = make_row(3, 0)\n"
            "    r2 = make_row(3, 0)\n"
            "    r1[0] = 9\n"
            "    assert r2[0] == 0\n"
            "\n"
            "# --- 跨层契约 ---\n"
            "def test_set_is_local():\n"
            "    b = Board(4, 4)\n"
            "    b.set(0, 2, 7)\n"
            "    assert b.get(0, 2) == 7\n"
            "    assert b.get(1, 2) == 0\n"
            "    assert b.get(2, 2) == 0\n"
            "\n"
            "def test_count_reflects_single_write():\n"
            "    b = Board(3, 3)\n"
            "    b.set(1, 1, 5)\n"
            "    assert b.count_equal(5) == 1\n"
            "\n"
            "def test_row_sum_is_isolated():\n"
            "    b = Board(3, 3)\n"
            "    b.set(0, 0, 2)\n"
            "    assert b.row_sum(0) == 2\n"
            "    assert b.row_sum(1) == 0\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_core_shape()\n"
            "    test_row_is_fresh()\n"
            "    test_set_is_local()\n"
            "    test_count_reflects_single_write()\n"
            "    test_row_sum_is_isolated()\n"
            "    print('All tests passed')\n"
        ),
    }


def _matrow_shared_row(files, rng):
    """整列复用同一个行对象：形状、初始值全对，但行之间不再独立。"""
    files = dict(files)
    files["matrix.py"] = files["matrix.py"].replace(
        "    return [make_row(width, fill) for _ in range(height)]\n",
        "    row = make_row(width, fill)\n"
        "    return [row for _ in range(height)]\n",
    )
    return files


# ---------------------------------------------------------------------------
# 族 3：groupord —— 「保持 key 首次出现顺序」契约
# ---------------------------------------------------------------------------
def _groupord_golden():
    return {
        "collect.py": (
            "def group_by(items, keyfn):\n"
            "    \"\"\"按 keyfn 分组。\n"
            "\n"
            "    契约（上层依赖）：结果的 key 顺序 == key 首次出现的顺序。\n"
            "    上层用它来决定报表里各分节的排版次序，不能重排。\n"
            "    \"\"\"\n"
            "    out = {}\n"
            "    for it in items:\n"
            "        out.setdefault(keyfn(it), []).append(it)\n"
            "    return out\n"
        ),
        "unit_core.py": (
            "# core 层单元测试 —— 只检验 collect.group_by 自身的性质。\n"
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from collect import group_by\n"
            "\n"
            "ROWS = [\n"
            "    {'cat': 'zeta', 'id': 1},\n"
            "    {'cat': 'alpha', 'id': 2},\n"
            "    {'cat': 'zeta', 'id': 3},\n"
            "    {'cat': 'mid', 'id': 4},\n"
            "    {'cat': 'alpha', 'id': 5},\n"
            "]\n"
            "\n"
            "def KEY(r):\n"
            "    return r['cat']\n"
            "\n"
            "def test_group_members():\n"
            "    g = group_by(ROWS, KEY)\n"
            "    assert sorted(g.keys()) == ['alpha', 'mid', 'zeta']\n"
            "    assert [r['id'] for r in g['zeta']] == [1, 3]\n"
            "    assert [r['id'] for r in g['alpha']] == [2, 5]\n"
            "\n"
            "def test_group_sizes():\n"
            "    g = group_by(ROWS, KEY)\n"
            "    # 顺序无关：这里是 core 层，不该泄漏任何关于顺序的信息\n"
            "    assert sorted(len(v) for v in g.values()) == [1, 2, 2]\n"
            "\n"
            "def test_within_group_order():\n"
            "    g = group_by(ROWS, KEY)\n"
            "    for rows in g.values():\n"
            "        ids = [r['id'] for r in rows]\n"
            "        assert ids == sorted(ids)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_group_members()\n"
            "    test_group_sizes()\n"
            "    test_within_group_order()\n"
            "    print('All tests passed')\n"
        ),
        "report.py": (
            "from collect import group_by\n"
            "\n"
            "def section_order(rows, keyfn):\n"
            "    \"\"\"各分节的排版次序：按分节首次出现的先后。\"\"\"\n"
            "    return list(group_by(rows, keyfn).keys())\n"
            "\n"
            "\n"
            "def first_of_each(rows, keyfn):\n"
            "    \"\"\"每个分节的代表行，同样按分节首次出现顺序。\"\"\"\n"
            "    g = group_by(rows, keyfn)\n"
            "    return [v[0] for v in g.values()]\n"
            "\n"
            "\n"
            "def summary(rows, keyfn):\n"
            "    \"\"\"(分节名, 行数) 列表，按分节首次出现顺序。\"\"\"\n"
            "    g = group_by(rows, keyfn)\n"
            "    return [(k, len(v)) for k, v in g.items()]\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from collect import group_by\n"
            "from report import section_order, first_of_each, summary\n"
            "\n"
            "ROWS = [\n"
            "    {'cat': 'zeta', 'id': 1},\n"
            "    {'cat': 'alpha', 'id': 2},\n"
            "    {'cat': 'zeta', 'id': 3},\n"
            "    {'cat': 'mid', 'id': 4},\n"
            "    {'cat': 'alpha', 'id': 5},\n"
            "]\n"
            "\n"
            "def KEY(r):\n"
            "    return r['cat']\n"
            "\n"
            "# --- core 层单元性质 ---\n"
            "def test_core_groups_correctly():\n"
            "    g = group_by(ROWS, KEY)\n"
            "    assert sorted(g.keys()) == ['alpha', 'mid', 'zeta']\n"
            "    assert [r['id'] for r in g['zeta']] == [1, 3]\n"
            "    assert [r['id'] for r in g['alpha']] == [2, 5]\n"
            "    assert [r['id'] for r in g['mid']] == [4]\n"
            "\n"
            "def test_core_keeps_within_group_order():\n"
            "    g = group_by(ROWS, KEY)\n"
            "    for rows in g.values():\n"
            "        ids = [r['id'] for r in rows]\n"
            "        assert ids == sorted(ids)\n"
            "\n"
            "# --- 跨层契约 ---\n"
            "def test_section_follows_first_seen():\n"
            "    assert section_order(ROWS, KEY) == ['zeta', 'alpha', 'mid']\n"
            "\n"
            "def test_first_of_each_follows_first_seen():\n"
            "    got = [r['id'] for r in first_of_each(ROWS, KEY)]\n"
            "    assert got == [1, 2, 4]\n"
            "\n"
            "def test_summary_follows_first_seen():\n"
            "    assert summary(ROWS, KEY) == [('zeta', 2), ('alpha', 2), ('mid', 1)]\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_core_groups_correctly()\n"
            "    test_core_keeps_within_group_order()\n"
            "    test_section_follows_first_seen()\n"
            "    test_first_of_each_follows_first_seen()\n"
            "    test_summary_follows_first_seen()\n"
            "    print('All tests passed')\n"
        ),
    }


def _groupord_sorted_keys(files, rng):
    """把 key 按字典序重排：分组内容一个字没变，但顺序契约崩了。"""
    files = dict(files)
    files["collect.py"] = files["collect.py"].replace(
        "        out.setdefault(keyfn(it), []).append(it)\n"
        "    return out\n",
        "        out.setdefault(keyfn(it), []).append(it)\n"
        "    return {k: out[k] for k in sorted(out, key=str)}\n",
    )
    return files


# ---------------------------------------------------------------------------
# 族 4：snapdiff —— 「快照与活对象不再共享」契约
# ---------------------------------------------------------------------------
def _snapdiff_golden():
    return {
        "history.py": (
            "def snapshot(state):\n"
            "    \"\"\"返回 state 的独立快照。\n"
            "\n"
            "    契约（上层依赖）：返回之后，再改 state 不会反过来改到快照。\n"
            "    上层靠「快照 vs 现状」的差来判断哪里被改动过。\n"
            "    \"\"\"\n"
            "    out = {}\n"
            "    for k, v in state.items():\n"
            "        out[k] = list(v) if isinstance(v, (list, tuple)) else v\n"
            "    return out\n"
            "\n"
            "\n"
            "def merge(base, incoming):\n"
            "    \"\"\"把 incoming 合进 base 的一份副本，不改动调用方的对象。\"\"\"\n"
            "    merged = {}\n"
            "    for src in (base, incoming):\n"
            "        for k, v in src.items():\n"
            "            merged[k] = list(v) if isinstance(v, (list, tuple)) else v\n"
            "    return merged\n"
        ),
        "unit_core.py": (
            "# core 层单元测试 —— 只检验 history 自身的性质。\n"
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from history import snapshot, merge\n"
            "\n"
            "def test_snapshot_keys_and_values():\n"
            "    s = {'alice': [1, 2, 3], 'bob': [4, 5]}\n"
            "    snap = snapshot(s)\n"
            "    assert sorted(snap.keys()) == ['alice', 'bob']\n"
            "    assert snap['alice'] == [1, 2, 3]\n"
            "    assert list(snap['bob']) == [4, 5]\n"
            "\n"
            "def test_merge_union():\n"
            "    m = merge({'alice': [1, 2, 3], 'bob': [4, 5]}, {'carol': [6]})\n"
            "    assert sorted(m.keys()) == ['alice', 'bob', 'carol']\n"
            "    assert m['carol'] == [6]\n"
            "\n"
            "def test_merge_overwrites():\n"
            "    m = merge({'a': [1]}, {'a': [2]})\n"
            "    assert m['a'] == [2]\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_snapshot_keys_and_values()\n"
            "    test_merge_union()\n"
            "    test_merge_overwrites()\n"
            "    print('All tests passed')\n"
        ),
        "diffview.py": (
            "from history import snapshot, merge\n"
            "\n"
            "class Tracker:\n"
            "    \"\"\"跟踪一份状态自建立以来被改过哪些键。\"\"\"\n"
            "\n"
            "    def __init__(self, state):\n"
            "        self.live = state\n"
            "        self.base = snapshot(state)\n"
            "\n"
            "    def changed_keys(self):\n"
            "        return sorted(k for k in self.live\n"
            "                      if list(self.live[k]) != self.base[k])\n"
            "\n"
            "    def reset_to_base(self):\n"
            "        for k in self.base:\n"
            "            self.live[k] = list(self.base[k])\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from history import snapshot, merge\n"
            "from diffview import Tracker\n"
            "\n"
            "def _state():\n"
            "    return {'alice': [1, 2, 3], 'bob': [4, 5]}\n"
            "\n"
            "# --- core 层单元性质 ---\n"
            "def test_core_snapshot_shape():\n"
            "    snap = snapshot(_state())\n"
            "    assert sorted(snap.keys()) == ['alice', 'bob']\n"
            "    assert snap['alice'] == [1, 2, 3]\n"
            "    assert list(snap['bob']) == [4, 5]\n"
            "\n"
            "def test_core_merge_shape():\n"
            "    m = merge(_state(), {'carol': [6]})\n"
            "    assert sorted(m.keys()) == ['alice', 'bob', 'carol']\n"
            "    assert m['carol'] == [6]\n"
            "\n"
            "# --- 跨层契约 ---\n"
            "def test_tracker_sees_nothing_initially():\n"
            "    t = Tracker(_state())\n"
            "    assert t.changed_keys() == []\n"
            "\n"
            "def test_tracker_detects_mutation():\n"
            "    t = Tracker(_state())\n"
            "    t.live['alice'].append(99)\n"
            "    assert t.changed_keys() == ['alice']\n"
            "\n"
            "def test_reset_restores_base():\n"
            "    t = Tracker(_state())\n"
            "    t.live['alice'].append(99)\n"
            "    t.reset_to_base()\n"
            "    assert t.live['alice'] == [1, 2, 3]\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_core_snapshot_shape()\n"
            "    test_core_merge_shape()\n"
            "    test_tracker_sees_nothing_initially()\n"
            "    test_tracker_detects_mutation()\n"
            "    test_reset_restores_base()\n"
            "    print('All tests passed')\n"
        ),
    }


def _snapdiff_shallow(files, rng):
    """快照退化成浅拷贝：键、值、长度全对，但嵌套列表与活对象共享。"""
    files = dict(files)
    files["history.py"] = files["history.py"].replace(
        "    out = {}\n"
        "    for k, v in state.items():\n"
        "        out[k] = list(v) if isinstance(v, (list, tuple)) else v\n"
        "    return out\n",
        "    return dict(state)\n",
    )
    return files


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------
CHAIN = [
    {"id": "idxsort", "bug_file": "store.py", "service_file": "lookup.py",
     "entry": "tests.py", "files": _idxsort_golden(),
     "bug_patterns": [
         {"name": "tail_unsorted",
          "desc": "normalize 把去重后的最后一个元素追加在末尾而不参与排序，返回值不再严格升序",
          "mutate": _idxsort_tail_unsorted},
     ]},
    {"id": "matrow", "bug_file": "matrix.py", "service_file": "board.py",
     "entry": "tests.py", "files": _matrow_golden(),
     "bug_patterns": [
         {"name": "shared_row",
          "desc": "make_matrix 复用同一个行对象，各行不再是彼此独立的列表",
          "mutate": _matrow_shared_row},
     ]},
    {"id": "groupord", "bug_file": "collect.py", "service_file": "report.py",
     "entry": "tests.py", "files": _groupord_golden(),
     "bug_patterns": [
         {"name": "sorted_keys",
          "desc": "group_by 把 key 按字典序重排，不再是首次出现的顺序",
          "mutate": _groupord_sorted_keys},
     ]},
    {"id": "snapdiff", "bug_file": "history.py", "service_file": "diffview.py",
     "entry": "tests.py", "files": _snapdiff_golden(),
     "bug_patterns": [
         {"name": "shallow_snapshot",
          "desc": "snapshot 退化成浅拷贝，嵌套列表仍与原始状态共享",
          "mutate": _snapdiff_shallow},
     ]},
]


# ---------------------------------------------------------------------------
# 自检：把「这组题到底是不是跨层题」变成可执行断言
# ---------------------------------------------------------------------------
#
# 注意：本机评估路径是 `python <entry>` + 匹配 "All tests passed"（见
# CodeEnv.run_tests），没有 pytest，且只看全通过与否。所以「core 层单独看仍然
# 正确」这条约束在正式评估里是**看不见**的 —— 必须额外配一个 unit_core.py
# 单独跑，否则本题型的设计意图只能是嘴上说说。
#
# ---------------------------------------------------------------------------


def _harness(files, workdir):
    """把一组文件落到 workdir（清空重建）。"""
    import os
    import shutil

    if os.path.isdir(workdir):
        shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    for name, src in files.items():
        path = os.path.join(workdir, name)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)


def run_entry(files, entry, workdir, timeout=60.0):
    """按正式评估的方式跑一个入口，返回 (通过?, stdout, stderr)。"""
    import subprocess
    import sys

    _harness(files, workdir)
    try:
        p = subprocess.run([sys.executable, entry], cwd=workdir,
                           capture_output=True, text=True, timeout=timeout)
        return ("All tests passed" in (p.stdout or ""),
                p.stdout or "", p.stderr or "")
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", ""


def run_percase(files, entry, workdir):
    """逐条用例跑，返回 [(用例名, 通过?, 错因)]。观测用，不是评估路径。"""
    import importlib.util
    import os
    import sys
    import traceback

    _harness(files, workdir)
    sys.path.insert(0, workdir)
    modname = "_probe_" + os.path.basename(os.path.normpath(workdir))
    try:
        spec = importlib.util.spec_from_file_location(
            modname, os.path.join(workdir, entry))
        m = importlib.util.module_from_spec(spec)
        sys.modules[modname] = m
        spec.loader.exec_module(m)
    except Exception:
        return [("<import>", False,
                 traceback.format_exc(limit=1).strip().splitlines()[-1][:160])]
    finally:
        if workdir in sys.path:
            sys.path.remove(workdir)
        sys.modules.pop(modname, None)

    out = []
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        fn = getattr(m, name)
        if not callable(fn):
            continue
        try:
            fn()
            out.append((name, True, ""))
        except Exception as e:
            msg = str(e).strip().splitlines()
            out.append((name, False, "%s: %s" % (type(e).__name__,
                                                 msg[0][:110] if msg else "")))
    return out


def selftest(workdir=None):
    """验收每条跨层题都满足设计约束：

    1. golden 版 tests.py 全通过，core 单测也全通过；
    2. 注入 bug 后 tests.py 失败（题是有效的）；
    3. **注入 bug 后 unit_core.py 仍然全绿** —— bug 在 core 单独看来是对的。
       这条是 CHAIN 与 GOLDEN 的分水岭，必须量化，不能靠嘴。
    """
    import os
    import random
    import tempfile

    rng = random.Random(0)
    base = workdir or tempfile.mkdtemp(prefix="chain_selftest_")
    rows = []

    for fam in CHAIN:
        for bp in fam["bug_patterns"]:
            tag = "%s_%s" % (fam["id"], bp["name"])
            gdir = os.path.join(base, tag + "_golden")
            bdir = os.path.join(base, tag + "_bug")

            gold_ok, _, gold_err = run_entry(fam["files"], fam["entry"], gdir)
            gold_core_ok, _, _ = run_entry(fam["files"], "unit_core.py", gdir)

            bad = bp["mutate"](fam["files"], rng)
            bug_ok, _, bug_err = run_entry(bad, fam["entry"], bdir)
            bug_core_ok, _, _ = run_entry(bad, "unit_core.py", bdir)

            cases = run_percase(bad, fam["entry"], bdir)
            failed = [(n, e) for n, ok, e in cases if not ok]

            ok = gold_ok and gold_core_ok and (not bug_ok) and bug_core_ok
            rows.append({
                "id": "%s::%s" % (fam["id"], bp["name"]),
                "ok": ok,
                "golden": gold_ok,
                "golden_core": gold_core_ok,
                "bug_detected": not bug_ok,
                "bug_core_still_ok": bug_core_ok,
                "n_cases": len(cases),
                "failed_cases": failed,
                "err_hint": (bug_err or gold_err or "").strip()[:200],
            })

    return rows


if __name__ == "__main__":
    print("%-30s %7s %7s %7s %7s" % ("题目", "golden", "core绿", "捉到bug", "bug后core仍绿"))
    print("-" * 72)
    bad_cnt = 0
    for r in selftest():
        flag = "OK " if r["ok"] else "BAD"
        if not r["ok"]:
            bad_cnt += 1
        print("%s %-26s %7s %7s %7s %7s"
              % (flag, r["id"], r["golden"], r["golden_core"],
                 r["bug_detected"], r["bug_core_still_ok"]))
        for n, e in r["failed_cases"]:
            print("        × %-32s %s" % (n, e[:70]))
    print("-" * 72)
    print("不合格 %d 条" % bad_cnt)
