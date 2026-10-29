# -*- coding: utf-8 -*-
"""任务自举：golden 正确实现池 + 程序化 bug 注入器。

设计定位（主人 2026-10-29 定）：
- 自进化的本质是「世界模型不断迭代最优解」。编程域里，golden 实现 = 已知最优解，
  注入 bug 后让「生成器（LLM，脑+手）」去重构最优解；
- 世界模型（难度维度 size/depth/hidden/scaffold/resource）决定：
    * 选哪个算法族；
    * 注入哪种 bug 模式（off-by-one / 边界 / 异常 / 逻辑反转 / 缺校验…）；
    * hidden>=2 => 盲改（不把测试报错喂给生成器，逼它盲修）；
    * size>=3 => 加干扰代码增大体积；
    * scaffold 维度决定可改文件范围（多文件族生效）。
- 任务空间 = 族数 × bug 模式数 × 难度组合，指数级，且每个都有确定性 golden 测试，
  注入 bug 后必败、修复后必胜。这就是「题不再是单个死题」，而是世界持续生成。

mutate 用确定性字符串替换（与 golden 源码里的确切文本对应），保证注入可靠、
修复可过。新增族只需往 GOLDEN 里加一项即可。
"""

import random


# ---------------------------------------------------------------------------
# golden 族：每族 = 正确实现 files + 测试入口 entry + 若干 bug 模式
# 每个 bug 模式 = (name, desc, mutate(files, rng)->files) 把正确版改坏
# ---------------------------------------------------------------------------

def _calc_golden():
    return {
        "calculator.py": (
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "def subtract(a, b):\n"
            "    return a - b\n"
            "\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "\n"
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ZeroDivisionError('division by zero')\n"
            "    return a / b\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from calculator import add, subtract, multiply, divide\n"
            "\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n"
            "    assert add(-1, 1) == 0\n"
            "\n"
            "def test_subtract():\n"
            "    assert subtract(10, 4) == 6\n"
            "\n"
            "def test_multiply():\n"
            "    assert multiply(3, 4) == 12\n"
            "\n"
            "def test_divide():\n"
            "    assert divide(10, 2) == 5.0\n"
            "    try:\n"
            "        divide(1, 0)\n"
            "        raise AssertionError('should have raised')\n"
            "    except ZeroDivisionError:\n"
            "        pass\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_add(); test_subtract(); test_multiply(); test_divide()\n"
            "    print('All tests passed')\n"
        ),
    }


def _calc_mutate_none(files, rng):
    files = dict(files)
    files["calculator.py"] = files["calculator.py"].replace(
        "    if b == 0:\n        raise ZeroDivisionError('division by zero')\n",
        "    if b == 0:\n        return None\n",
    )
    return files


def _calc_mutate_add(files, rng):
    files = dict(files)
    files["calculator.py"] = files["calculator.py"].replace(
        "    return a + b\n",
        "    return a + b + 1\n",
    )
    return files


def _calc_mutate_mul(files, rng):
    files = dict(files)
    files["calculator.py"] = files["calculator.py"].replace(
        "    return a * b\n",
        "    return a * b - 1\n",
    )
    return files


def _lru_golden():
    return {
        "lru.py": (
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache = {}\n"
            "        self.order = []\n"
            "\n"
            "    def get(self, key):\n"
            "        if key in self.cache:\n"
            "            self.order.remove(key)\n"
            "            self.order.append(key)\n"
            "            return self.cache[key]\n"
            "        return -1\n"
            "\n"
            "    def put(self, key, value):\n"
            "        if key in self.cache:\n"
            "            self.cache[key] = value\n"
            "        else:\n"
            "            if len(self.cache) >= self.capacity:\n"
            "                oldest = self.order.pop(0)\n"
            "                del self.cache[oldest]\n"
            "            self.cache[key] = value\n"
            "            self.order.append(key)\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from lru import LRUCache\n"
            "\n"
            "def test_lru_evict():\n"
            "    c = LRUCache(2)\n"
            "    c.put(1, 1); c.put(2, 2)\n"
            "    assert c.get(1) == 1\n"
            "    c.put(3, 3)\n"
            "    assert c.get(2) == -1, 'least recent should be evicted'\n"
            "    c.put(4, 4)\n"
            "    assert c.get(1) == -1\n"
            "    assert c.get(3) == 3 and c.get(4) == 4\n"
            "\n"
            "def test_lru_update():\n"
            "    c = LRUCache(1)\n"
            "    c.put(1, 1); c.put(1, 2)\n"
            "    assert c.get(1) == 2\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_lru_evict(); test_lru_update()\n"
            "    print('All tests passed')\n"
        ),
    }


def _lru_no_reorder(files, rng):
    files = dict(files)
    files["lru.py"] = files["lru.py"].replace(
        "        if key in self.cache:\n            self.order.remove(key)\n            self.order.append(key)\n            return self.cache[key]\n",
        "        if key in self.cache:\n            return self.cache[key]\n",
    )
    return files


def _lru_wrong_cap(files, rng):
    files = dict(files)
    files["lru.py"] = files["lru.py"].replace(
        "            if len(self.cache) >= self.capacity:\n",
        "            if len(self.cache) > self.capacity:\n",
    )
    return files


def _bs_golden():
    return {
        "binary_search.py": (
            "def bisect(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from binary_search import bisect\n"
            "\n"
            "def test_found():\n"
            "    a = [1, 3, 5, 7, 9]\n"
            "    assert bisect(a, 5) == 2\n"
            "    assert bisect(a, 1) == 0\n"
            "    assert bisect(a, 9) == 4\n"
            "\n"
            "def test_not_found():\n"
            "    a = [1, 3, 5, 7, 9]\n"
            "    assert bisect(a, 4) == -1\n"
            "    assert bisect(a, 0) == -1\n"
            "    assert bisect(a, 100) == -1\n"
            "    assert bisect([], 1) == -1\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_found(); test_not_found()\n"
            "    print('All tests passed')\n"
        ),
    }


def _bs_off_by_one(files, rng):
    files = dict(files)
    files["binary_search.py"] = files["binary_search.py"].replace(
        "    while lo <= hi:\n",
        "    while lo < hi:\n",
    )
    return files


def _bs_wrong_return(files, rng):
    files = dict(files)
    files["binary_search.py"] = files["binary_search.py"].replace(
        "    return -1\n",
        "    return hi\n",
    )
    return files


def _dedupe_golden():
    return {
        "dedupe.py": (
            "def dedupe(arr):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for x in arr:\n"
            "        if x not in seen:\n"
            "            seen.add(x)\n"
            "            out.append(x)\n"
            "    return out\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from dedupe import dedupe\n"
            "\n"
            "def test_keep_order():\n"
            "    assert dedupe([1, 2, 2, 3, 1]) == [1, 2, 3]\n"
            "    assert dedupe([5, 5, 5]) == [5]\n"
            "    assert dedupe([]) == []\n"
            "    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]\n"
            "    assert dedupe([1, 2, 3]) == [1, 2, 3]\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_keep_order()\n"
            "    print('All tests passed')\n"
        ),
    }


def _dedupe_unordered(files, rng):
    files = dict(files)
    files["dedupe.py"] = files["dedupe.py"].replace(
        "    seen = set()\n    out = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
        "    return list(set(arr))\n",
    )
    return files


def _dedupe_off_by_one(files, rng):
    files = dict(files)
    files["dedupe.py"] = files["dedupe.py"].replace(
        "    for x in arr:\n",
        "    for i in range(len(arr) - 1):\n        x = arr[i]\n",
    )
    return files


def _palin_golden():
    return {
        "palindrome.py": (
            "def is_palindrome(s):\n"
            "    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
            "    return t == t[::-1]\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from palindrome import is_palindrome\n"
            "\n"
            "def test_basic():\n"
            "    assert is_palindrome('aba') is True\n"
            "    assert is_palindrome('abc') is False\n"
            "    assert is_palindrome('A man a plan a canal Panama') is True\n"
            "    assert is_palindrome('RaceCar') is True\n"
            "    assert is_palindrome('') is True\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_basic()\n"
            "    print('All tests passed')\n"
        ),
    }


def _palin_no_lower(files, rng):
    files = dict(files)
    files["palindrome.py"] = files["palindrome.py"].replace(
        "    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n",
        "    t = ''.join(ch for ch in s if ch.isalnum())\n",
    )
    return files


def _palin_keep_symbol(files, rng):
    files = dict(files)
    files["palindrome.py"] = files["palindrome.py"].replace(
        "    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n",
        "    t = ''.join(ch.lower() for ch in s)\n",
    )
    return files


def _fib_golden():
    return {
        "fib.py": (
            "def fib(n, memo=None):\n"
            "    if memo is None:\n"
            "        memo = {}\n"
            "    if n in memo:\n"
            "        return memo[n]\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n"
            "    return memo[n]\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from fib import fib\n"
            "\n"
            "def test_fib():\n"
            "    assert fib(0) == 0\n"
            "    assert fib(1) == 1\n"
            "    assert fib(10) == 55\n"
            "    assert fib(20) == 6765\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_fib()\n"
            "    print('All tests passed')\n"
        ),
    }


def _fib_base(files, rng):
    files = dict(files)
    files["fib.py"] = files["fib.py"].replace(
        "    if n <= 1:\n        return n\n",
        "    if n <= 1:\n        return 1\n",
    )
    return files


def _fib_off_by_one(files, rng):
    files = dict(files)
    files["fib.py"] = files["fib.py"].replace(
        "    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n",
        "    memo[n] = fib(n - 2, memo) + fib(n - 3, memo)\n",
    )
    return files


def _stack_golden():
    return {
        "stack.py": (
            "def is_balanced(s):\n"
            "    stack = []\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    for ch in s:\n"
            "        if ch in '([{':\n"
            "            stack.append(ch)\n"
            "        elif ch in ')]}':\n"
            "            if not stack or stack.pop() != pairs[ch]:\n"
            "                return False\n"
            "    return len(stack) == 0\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from stack import is_balanced\n"
            "\n"
            "def test_balanced():\n"
            "    assert is_balanced('()') is True\n"
            "    assert is_balanced('([]){}') is True\n"
            "    assert is_balanced('(]') is False\n"
            "    assert is_balanced('(') is False\n"
            "    assert is_balanced(')(') is False\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_balanced()\n"
            "    print('All tests passed')\n"
        ),
    }


def _stack_no_close(files, rng):
    files = dict(files)
    files["stack.py"] = files["stack.py"].replace(
        "            if not stack or stack.pop() != pairs[ch]:\n                return False\n",
        "            if not stack:\n                return False\n",
    )
    return files


def _stack_wrong_pair(files, rng):
    files = dict(files)
    files["stack.py"] = files["stack.py"].replace(
        "    pairs = {')': '(', ']': '[', '}': '{'}\n",
        "    pairs = {')': '[', ']': '(', '}': '{'}\n",
    )
    return files


# ---------------------------------------------------------------------------
# 注册 golden 族
# ---------------------------------------------------------------------------
GOLDEN = [
    {"id": "calc", "bug_file": "calculator.py", "entry": "tests.py",
     "files": _calc_golden(),
     "bug_patterns": [
         {"name": "divide_none", "desc": "divide 在 b==0 时返回 None 而不是抛 ZeroDivisionError",
          "mutate": _calc_mutate_none},
         {"name": "add_off_by_one", "desc": "add 多算了 1（off-by-one）",
          "mutate": _calc_mutate_add},
         {"name": "mul_minus_one", "desc": "multiply 少算了 1（逻辑错误）",
          "mutate": _calc_mutate_mul},
     ]},
    {"id": "lru", "bug_file": "lru.py", "entry": "tests.py",
     "files": _lru_golden(),
     "bug_patterns": [
         {"name": "no_reorder", "desc": "get 命中时不把 key 移到最近位置，破坏 LRU 顺序",
          "mutate": _lru_no_reorder},
         {"name": "wrong_capacity", "desc": "容量判断用 > 而非 >=，多放一个元素",
          "mutate": _lru_wrong_cap},
     ]},
    {"id": "bisect", "bug_file": "binary_search.py", "entry": "tests.py",
     "files": _bs_golden(),
     "bug_patterns": [
         {"name": "off_by_one", "desc": "二分查找循环条件写成 lo < hi，漏判 mid==hi",
          "mutate": _bs_off_by_one},
         {"name": "wrong_return", "desc": "没找到时返回 hi 而非 -1",
          "mutate": _bs_wrong_return},
     ]},
    {"id": "dedupe", "bug_file": "dedupe.py", "entry": "tests.py",
     "files": _dedupe_golden(),
     "bug_patterns": [
         {"name": "unordered", "desc": "用 set 去重后转 list，丢失原始顺序",
          "mutate": _dedupe_unordered},
         {"name": "off_by_one", "desc": "循环少处理最后一个元素",
          "mutate": _dedupe_off_by_one},
     ]},
    {"id": "palindrome", "bug_file": "palindrome.py", "entry": "tests.py",
     "files": _palin_golden(),
     "bug_patterns": [
         {"name": "no_lower", "desc": "比较前不转小写，大小写敏感导致误判",
          "mutate": _palin_no_lower},
         {"name": "keep_symbol", "desc": "不过滤非字母数字字符，标点破坏回文判断",
          "mutate": _palin_keep_symbol},
     ]},
    {"id": "fib", "bug_file": "fib.py", "entry": "tests.py",
     "files": _fib_golden(),
     "bug_patterns": [
         {"name": "base_wrong", "desc": "fib(0) 和 fib(1) 都返回 1（基准错误）",
          "mutate": _fib_base},
         {"name": "off_by_one", "desc": "递归式写成 fib(n-2)+fib(n-3)，偏移一位",
          "mutate": _fib_off_by_one},
     ]},
    {"id": "stack", "bug_file": "stack.py", "entry": "tests.py",
     "files": _stack_golden(),
     "bug_patterns": [
         {"name": "no_close_check", "desc": "遇到右括号时不检查是否匹配，只看栈空",
          "mutate": _stack_no_close},
         {"name": "wrong_pair", "desc": "括号配对表写错（) 对应 [ 而非 (）",
          "mutate": _stack_wrong_pair},
     ]},
]


def make_task(rng, dims):
    """程序化生成一项编程任务。

    dims: dict，含 size/depth/hidden/scaffold/resource（来自 World 难度维度）。
    返回 {id, files, entry, bug_file, bug_desc, hidden, editable}：
      - hidden: bool，True 表示盲改（不把测试报错喂给生成器）
      - editable: 生成器可修改的文件名列表（scaffold 维度预留）
    """
    fam = rng.choice(GOLDEN)
    bp = rng.choice(fam["bug_patterns"])
    files = dict(fam["files"])
    files = bp["mutate"](files, rng)
    bug_file = fam["bug_file"]

    size = dims.get("size", 1)
    hidden = dims.get("hidden", 1)
    scaffold = dims.get("scaffold", 1)

    # size>=3：加真实（但无关）的干扰函数，逼迫生成器在噪声里定位 bug
    if size >= 3:
        noise = (
            "\n\n"
            "def _noise_helper(x):\n"
            "    # 无关辅助函数，纯干扰\n"
            "    acc = 0\n"
            "    for i in range(len(x) if isinstance(x, (list, str)) else 0):\n"
            "        acc += i\n"
            "    return acc\n"
            "\n"
            "def _noise_another(n):\n"
            "    return [i * 2 for i in range(n)]\n"
        )
        files[bug_file] = files[bug_file] + noise

    # hidden>=2：盲改，不把测试报错喂给生成器；hidden>=3：连 bug 描述都模糊化
    blind = hidden >= 2
    if hidden >= 3:
        bug_desc = "请根据测试失败信息自行定位并修复该实现中的 bug（不提供具体描述）"

    # scaffold>=3：多文件可改（当前族都是单文件，预留接口）
    editable = [bug_file]

    return {
        "id": "%s::%s" % (fam["id"], bp["name"]),
        "files": files,
        "entry": fam["entry"],
        "bug_file": bug_file,
        "bug_desc": bp["desc"],
        "hidden": blind,
        "editable": editable,
    }


def list_families():
    return [(f["id"], [b["name"] for b in f["bug_patterns"]]) for f in GOLDEN]


# ---------------------------------------------------------------------------
# 硬核算法族（2026-10-29 新增）：算法正确性 + 数值/逻辑微妙错，盲改真会错
# ---------------------------------------------------------------------------
def _topo_golden():
    return {
        "topo.py": (
            "from collections import deque\n"
            "\n"
            "def topo_sort(graph):\n"
            "    indeg = {u: 0 for u in graph}\n"
            "    for u in graph:\n"
            "        for v in graph[u]:\n"
            "            indeg[v] = indeg.get(v, 0) + 1\n"
            "    q = deque([u for u in graph if indeg[u] == 0])\n"
            "    order = []\n"
            "    while q:\n"
            "        u = q.popleft()\n"
            "        order.append(u)\n"
            "        for v in graph[u]:\n"
            "            indeg[v] -= 1\n"
            "            if indeg[v] == 0:\n"
            "                q.append(v)\n"
            "    if len(order) != len(graph):\n"
            "        raise ValueError('cycle detected')\n"
            "    return order\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from topo import topo_sort\n"
            "\n"
            "def test_chain():\n"
            "    g = {'A': ['B'], 'B': ['C'], 'C': []}\n"
            "    assert topo_sort(g) == ['A', 'B', 'C']\n"
            "\n"
            "def test_repeated_edge():\n"
            "    g = {'A': ['B', 'B'], 'B': []}\n"
            "    assert topo_sort(g) == ['A', 'B']\n"
            "\n"
            "def test_cycle():\n"
            "    g = {'A': ['B'], 'B': ['A']}\n"
            "    try:\n"
            "        topo_sort(g)\n"
            "        raise AssertionError('should detect cycle')\n"
            "    except ValueError:\n"
            "        pass\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_chain(); test_repeated_edge(); test_cycle()\n"
            "    print('All tests passed')\n"
        ),
    }


def _topo_dup_indeg(files, rng):
    files = dict(files)
    files["topo.py"] = files["topo.py"].replace(
        "            indeg[v] = indeg.get(v, 0) + 1\n",
        "            indeg[v] = indeg.get(v, 0) + 2\n",
    )
    return files


def _topo_no_check(files, rng):
    files = dict(files)
    files["topo.py"] = files["topo.py"].replace(
        "    if len(order) != len(graph):\n        raise ValueError('cycle detected')\n",
        "",
    )
    return files


def _dij_golden():
    return {
        "dijkstra.py": (
            "import heapq\n"
            "\n"
            "def dijkstra(graph, start):\n"
            "    dist = {n: float('inf') for n in graph}\n"
            "    dist[start] = 0\n"
            "    pq = [(0, start)]\n"
            "    while pq:\n"
            "        d, u = heapq.heappop(pq)\n"
            "        if d > dist[u]:\n"
            "            continue\n"
            "        for v, w in graph[u]:\n"
            "            if dist[u] + w < dist[v]:\n"
            "                dist[v] = dist[u] + w\n"
            "                heapq.heappush(pq, (dist[v], v))\n"
            "    return dist\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from dijkstra import dijkstra\n"
            "\n"
            "def test_basic():\n"
            "    g = {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)], 'C': []}\n"
            "    d = dijkstra(g, 'A')\n"
            "    assert d['B'] == 1\n"
            "    assert d['C'] == 3\n"
            "    assert d['A'] == 0\n"
            "\n"
            "def test_unreachable():\n"
            "    g = {'A': [('B', 1)], 'C': []}\n"
            "    d = dijkstra(g, 'A')\n"
            "    assert d['C'] == float('inf')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_basic(); test_unreachable()\n"
            "    print('All tests passed')\n"
        ),
    }


def _dij_dist_off(files, rng):
    files = dict(files)
    files["dijkstra.py"] = files["dijkstra.py"].replace(
        "            if dist[u] + w < dist[v]:\n",
        "            if dist[u] + w + 1 < dist[v]:\n",
    )
    return files


def _merge_golden():
    return {
        "merge.py": (
            "def merge_intervals(intervals):\n"
            "    if not intervals:\n"
            "        return []\n"
            "    ivs = sorted(intervals, key=lambda x: x[0])\n"
            "    merged = [list(ivs[0])]\n"
            "    for s, e in ivs[1:]:\n"
            "        if s <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], e)\n"
            "        else:\n"
            "            merged.append([s, e])\n"
            "    return merged\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from merge import merge_intervals\n"
            "\n"
            "def test_overlap():\n"
            "    assert merge_intervals([[1, 3], [2, 5]]) == [[1, 5]]\n"
            "    assert merge_intervals([[1, 3], [4, 6]]) == [[1, 3], [4, 6]]\n"
            "    assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]\n"
            "\n"
            "def test_adjacent():\n"
            "    assert merge_intervals([[1, 3], [3, 6]]) == [[1, 6]]\n"
            "\n"
            "def test_empty():\n"
            "    assert merge_intervals([]) == []\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_overlap(); test_adjacent(); test_empty()\n"
            "    print('All tests passed')\n"
        ),
    }


def _merge_no_merge(files, rng):
    files = dict(files)
    files["merge.py"] = files["merge.py"].replace(
        "        if s <= merged[-1][1]:\n"
        "            merged[-1][1] = max(merged[-1][1], e)\n"
        "        else:\n"
        "            merged.append([s, e])\n",
        "        merged.append([s, e])\n",
    )
    return files


def _merge_boundary(files, rng):
    files = dict(files)
    files["merge.py"] = files["merge.py"].replace(
        "        if s <= merged[-1][1]:\n",
        "        if s < merged[-1][1]:\n",
    )
    return files


def _qs_golden():
    return {
        "quicksort.py": (
            "def quicksort(arr, lo=0, hi=None):\n"
            "    if hi is None:\n"
            "        hi = len(arr) - 1\n"
            "    if lo < hi:\n"
            "        p = partition(arr, lo, hi)\n"
            "        quicksort(arr, lo, p - 1)\n"
            "        quicksort(arr, p + 1, hi)\n"
            "    return arr\n"
            "\n"
            "def partition(arr, lo, hi):\n"
            "    pivot = arr[hi]\n"
            "    i = lo\n"
            "    for j in range(lo, hi):\n"
            "        if arr[j] < pivot:\n"
            "            arr[i], arr[j] = arr[j], arr[i]\n"
            "            i += 1\n"
            "    arr[i], arr[hi] = arr[hi], arr[i]\n"
            "    return i\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from quicksort import quicksort\n"
            "\n"
            "def test_sort():\n"
            "    assert quicksort([3, 1, 2, 5, 4]) == [1, 2, 3, 4, 5]\n"
            "    assert quicksort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]\n"
            "    assert quicksort([2, 2, 1]) == [1, 2, 2]\n"
            "    assert quicksort([]) == []\n"
            "    assert quicksort([1]) == [1]\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_sort()\n"
            "    print('All tests passed')\n"
        ),
    }


def _qs_wrong_bound(files, rng):
    files = dict(files)
    files["quicksort.py"] = files["quicksort.py"].replace(
        "    for j in range(lo, hi):\n",
        "    for j in range(lo, hi - 1):\n",
    )
    return files


def _bfs_golden():
    return {
        "bfs.py": (
            "from collections import deque\n"
            "\n"
            "def bfs_dist(graph, start):\n"
            "    dist = {start: 0}\n"
            "    q = deque([start])\n"
            "    while q:\n"
            "        u = q.popleft()\n"
            "        for v in graph.get(u, []):\n"
            "            if v not in dist:\n"
            "                dist[v] = dist[u] + 1\n"
            "                q.append(v)\n"
            "    return dist\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from bfs import bfs_dist\n"
            "\n"
            "def test_dist():\n"
            "    g = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}\n"
            "    d = bfs_dist(g, 'A')\n"
            "    assert d == {'A': 0, 'B': 1, 'C': 1, 'D': 2}\n"
            "\n"
            "def test_unreachable():\n"
            "    g = {'A': ['B'], 'C': []}\n"
            "    d = bfs_dist(g, 'A')\n"
            "    assert d == {'A': 0, 'B': 1}\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_dist(); test_unreachable()\n"
            "    print('All tests passed')\n"
        ),
    }


def _bfs_missing_one(files, rng):
    files = dict(files)
    files["bfs.py"] = files["bfs.py"].replace(
        "                dist[v] = dist[u] + 1\n",
        "                dist[v] = dist[u]\n",
    )
    return files


def _kadane_golden():
    return {
        "kadane.py": (
            "def max_subarray(arr):\n"
            "    if not arr:\n"
            "        return 0\n"
            "    best = cur = arr[0]\n"
            "    for x in arr[1:]:\n"
            "        cur = max(x, cur + x)\n"
            "        best = max(best, cur)\n"
            "    return best\n"
        ),
        "tests.py": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "from kadane import max_subarray\n"
            "\n"
            "def test_basic():\n"
            "    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\n"
            "    assert max_subarray([1, 2, 3]) == 6\n"
            "    assert max_subarray([-1, -2, -3]) == -1\n"
            "    assert max_subarray([]) == 0\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_basic()\n"
            "    print('All tests passed')\n"
        ),
    }


def _kadane_init_zero(files, rng):
    files = dict(files)
    files["kadane.py"] = files["kadane.py"].replace(
        "    best = cur = arr[0]\n",
        "    best = cur = 0\n",
    )
    return files


def _kadane_no_reset(files, rng):
    files = dict(files)
    files["kadane.py"] = files["kadane.py"].replace(
        "        cur = max(x, cur + x)\n",
        "        cur = cur + x\n",
    )
    return files


GOLDEN.extend([
    {"id": "topo", "bug_file": "topo.py", "entry": "tests.py",
     "files": _topo_golden(),
     "bug_patterns": [
         {"name": "dup_indeg", "desc": "统计入度时每条边把 indeg[v] 加 2 而非加 1，导致节点入度永远减不到 0 而不入队",
          "mutate": _topo_dup_indeg},
         {"name": "no_cycle_check", "desc": "删除环检测（len(order)!=len(graph) 时抛异常），含环图返回不完整结果",
          "mutate": _topo_no_check},
     ]},
    {"id": "dijkstra", "bug_file": "dijkstra.py", "entry": "tests.py",
     "files": _dij_golden(),
     "bug_patterns": [
         {"name": "dist_off_by_one", "desc": "松弛条件写成 dist[u]+w+1 < dist[v]，每条边距离系统性多算 1",
          "mutate": _dij_dist_off},
     ]},
    {"id": "merge", "bug_file": "merge.py", "entry": "tests.py",
     "files": _merge_golden(),
     "bug_patterns": [
         {"name": "no_merge", "desc": "删除区间合并逻辑，直接把每个区间 append，重叠区间不合并",
          "mutate": _merge_no_merge},
         {"name": "boundary_off", "desc": "合并判断用 s < merged[-1][1] 而非 <=，相邻区间 [a,b][b,c] 不合并",
          "mutate": _merge_boundary},
     ]},
    {"id": "quicksort", "bug_file": "quicksort.py", "entry": "tests.py",
     "files": _qs_golden(),
     "bug_patterns": [
         {"name": "wrong_bound", "desc": "partition 循环写成 range(lo, hi-1)，漏掉最后一个元素不参与比较",
          "mutate": _qs_wrong_bound},
     ]},
    {"id": "bfs", "bug_file": "bfs.py", "entry": "tests.py",
     "files": _bfs_golden(),
     "bug_patterns": [
         {"name": "missing_plus_one", "desc": "BFS 距离写成 dist[v]=dist[u] 漏了 +1，所有距离等于父距离",
          "mutate": _bfs_missing_one},
     ]},
    {"id": "kadane", "bug_file": "kadane.py", "entry": "tests.py",
     "files": _kadane_golden(),
     "bug_patterns": [
         {"name": "init_zero", "desc": "best=cur=0 而非 arr[0]，全负数组返回 0 而非最大负数",
          "mutate": _kadane_init_zero},
         {"name": "no_reset", "desc": "cur = cur + x 而非 cur = max(x, cur+x)，从不重置当前子段",
          "mutate": _kadane_no_reset},
     ]},
])


if __name__ == "__main__":
    # 自测：每个族每个 bug 模式，golden 必过、注入后必败
    import env  # noqa  (仅用于本地跑测试，实际由 CodeEnv 调用)
    rng = random.Random(1)
    print("golden 族:", len(GOLDEN))
    for f in GOLDEN:
        for bp in f["bug_patterns"]:
            files = dict(f["files"])
            files = bp["mutate"](files, rng)
            passed, error, ms, out = env.CodeEnv.run_tests(files, f["entry"])
            print("%-10s %-16s %s" % (f["id"], bp["name"], "OK(bug注入后败)" if not passed else "BUG:仍过"))
