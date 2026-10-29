"""编程 world model：环境 = 多文件代码库 + 编译器/测试。
Agent 能生成代码（不只是提交修复后的代码），通过测试验证。
难度维度由 World.make_template 控制：size/depth/hidden/scaffold/resource。
"""
import os, subprocess, sys, json, time, random, tempfile, textwrap, re, difflib
import gen  # LLM 生成器（外接编码模型当生成器）；CODE_DUEL_LLM 默认开（脑+手）
import tasks  # 任务自举：golden 池 + 程序化 bug 注入器


def change_lines(old_src, new_src):
    """两个版本之间「改了多少行」（增+删，不含未变部分）。"""
    a, b = old_src.splitlines(), new_src.splitlines()
    sm = difflib.SequenceMatcher(None, a, b)
    n = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            n += max(i2 - i1, j2 - j1)
    return n

GENE_NAMES = ("read_first","edit_scope","retry_cap","backtrack","test_cover",
              "refactor","doc_first","parallel_try","rollback_speed","scaffold")
BOUNDS = [(0,1),(0.1,1.0),(1,10),(0,1),(0,1),(0,1),(0,1),(1,5),(0,1),(0,1)]
DIMENSIONS = ("size","depth","hidden","scaffold","resource")
DIM_LABEL = {
    "size": "函数体积", "depth": "依赖深度",
    "hidden": "报错隐蔽度", "scaffold": "脚手架约束",
    "resource": "时空约束",
}

# ---------------------------------------------------------------- 多文件项目库
# 每个项目：多个文件 + 入口测试 + bug 文件 + bug 描述 + 难度标签
PROJECTS = [
    {
        "id": "calc_v1",
        "name": "基础计算器",
        "files": {
            "calculator.py": '''def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return None
    return a / b
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from calculator import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(3, 5) == -2

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-1, -1) == 1

def test_divide():
    assert divide(10, 2) == 5.0
    # 除以零必须抛异常，返回 None 是 bug
    threw = False
    try:
        divide(10, 0)
    except ZeroDivisionError:
        threw = True
    assert threw, "divide by zero must raise ZeroDivisionError"

if __name__ == "__main__":
    test_add()
    test_subtract()
    test_multiply()
    test_divide()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "calculator.py",
        "bug_desc": "divide 除以零返回 None，应该抛出 ZeroDivisionError",
        "tags": ["logic", "single_file"],
    },
    {
        "id": "calc_v2",
        "name": "多模块计算器",
        "files": {
            "ops.py": '''def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return float('inf')
    return a / b
''',
            "calc.py": '''from ops import add, subtract, multiply, divide

def compute(a, b, op):
    if op == "add":
        return add(a, b)
    elif op == "subtract":
        return subtract(a, b)
    elif op == "multiply":
        return multiply(a, b)
    elif op == "divide":
        return divide(a, b)
    else:
        return None
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from calc import compute

def test_add():
    assert compute(2, 3, "add") == 5

def test_subtract():
    assert compute(5, 3, "subtract") == 2

def test_multiply():
    assert compute(2, 3, "multiply") == 6

def test_divide():
    assert compute(10, 2, "divide") == 5.0
    # 除以零必须抛异常，返回 inf 是 bug
    threw = False
    try:
        compute(10, 0, "divide")
    except ZeroDivisionError:
        threw = True
    assert threw, "divide by zero must raise ZeroDivisionError"

if __name__ == "__main__":
    test_add()
    test_subtract()
    test_multiply()
    test_divide()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "ops.py",
        "bug_desc": "divide 除以零返回 float('inf')，应该抛出异常",
        "tags": ["multi_file", "logic"],
    },
    {
        "id": "cache_v1",
        "name": "LRU 缓存",
        "files": {
            "lru.py": '''class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return -1

    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
        else:
            if len(self.cache) >= self.capacity:
                oldest = self.order.pop(0)
                del self.cache[oldest]
            self.cache[key] = value
            self.order.append(key)
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from lru import LRUCache

def test_basic():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)
    assert c.get(2) == -1
    c.put(4, 4)
    assert c.get(1) == -1
    assert c.get(3) == 3
    assert c.get(4) == 4

if __name__ == "__main__":
    test_basic()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "lru.py",
        "bug_desc": "LRU 缓存未正确更新订单，get 后未将 key 移到最近使用位置",
        "tags": ["logic", "data_structure"],
    },
    {
        "id": "parser_v1",
        "name": "简单表达式解析器",
        "files": {
            "parser.py": '''def eval_expr(expr):
    """计算只含 + 和 - 的表达式，如 '1+2-3'"""
    tokens = re.findall(r"[0-9]+|[+-]", expr)
    result = int(tokens[0])
    i = 1
    while i < len(tokens):
        op = tokens[i]
        num = int(tokens[i + 1])
        if op == '+':
            result += num
        elif op == '-':
            result -= num
        i += 2
    return result
''',
            "tests.py": '''import sys, re
sys.path.insert(0, '.')
from parser import eval_expr

def test_add():
    assert eval_expr("1+2") == 3

def test_subtract():
    assert eval_expr("5-3") == 2

def test_mixed():
    assert eval_expr("10+5-3") == 12

if __name__ == "__main__":
    test_add()
    test_subtract()
    test_mixed()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "parser.py",
        "bug_desc": "eval_expr 使用了 re 模块但未 import re，会抛 NameError",
        "tags": ["hidden", "import_error"],
    },
    {
        "id": "graph_v1",
        "name": "图遍历",
        "files": {
            "graph.py": '''from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

def dfs(graph, start, visited=set()):
    # BUG: 可变默认参数，visited 在 def 时创建一次，所有调用共享同一 set
    visited.add(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from graph import bfs, dfs

def test_bfs():
    g = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}
    assert bfs(g, 'A') == {'A', 'B', 'C', 'D'}

def test_dfs():
    g = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': [], 'X': ['Y'], 'Y': []}
    r1 = dfs(g, 'A')
    assert r1 == {'A', 'B', 'C', 'D'}
    # 第二次独立调用不能带上第一次的节点（共享状态是 bug）
    r2 = dfs(g, 'X')
    assert r2 == {'X', 'Y'}, "dfs must not share visited across calls"

if __name__ == "__main__":
    test_bfs()
    test_dfs()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "graph.py",
        "bug_desc": "dfs 使用可变默认参数 visited=None，多次调用会共享状态",
        "tags": ["hidden", "mutable_default"],
    },
    {
        "id": "auth_v1",
        "name": "用户认证与会话",
        "files": {
            "auth.py": '''import hmac

USERS = {
    "alice": "password123",
    "bob": "qwerty456",
}


def check_password(username, password):
    stored = USERS.get(username)
    if stored is None:
        return False
    # BUG: 直接用 == 比对密码，易受时序攻击，应使用 hmac.compare_digest
    return stored == password


def authenticate(username, password):
    if check_password(username, password):
        return {"ok": True, "user": username}
    return {"ok": False, "error": "invalid credentials"}
''',
            "session.py": '''import uuid


SESSIONS = {}


def create_session(user_id):
    # BUG: 总是返回固定 session_id，会话固定漏洞，应生成随机 UUID
    session_id = "fixed-session"
    SESSIONS[session_id] = {"user_id": user_id}
    return session_id


def get_session(session_id):
    return SESSIONS.get(session_id)


def destroy_session(session_id):
    SESSIONS.pop(session_id, None)
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from auth import check_password, authenticate
from session import create_session, get_session, destroy_session


def test_check_password():
    assert check_password("alice", "password123") is True
    assert check_password("alice", "wrong") is False


def test_authenticate_ok():
    r = authenticate("alice", "password123")
    assert r["ok"] is True


def test_authenticate_fail():
    r = authenticate("alice", "wrong")
    assert r["ok"] is False


def test_session_unique():
    s1 = create_session("alice")
    s2 = create_session("bob")
    assert s1 != s2


def test_session_get_destroy():
    sid = create_session("alice")
    assert get_session(sid)["user_id"] == "alice"
    destroy_session(sid)
    assert get_session(sid) is None


if __name__ == "__main__":
    test_check_password()
    test_authenticate_ok()
    test_authenticate_fail()
    test_session_unique()
    test_session_get_destroy()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "auth.py",
        "bug_desc": "check_password 用 == 比对密码（时序攻击）；create_session 返回固定 session_id（会话固定）",
        "tags": ["security", "multi_file", "hidden"],
    },
    {
        "id": "bank_v1",
        "name": "银行转账原子性",
        "files": {
            "account.py": '''class Account:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if self.balance < amount:
            raise ValueError("insufficient funds")
        self.balance -= amount
        return True


def transfer(src, dst, amount):
    # BUG: 非原子，先扣后加，中间失败钱就丢了；没校验 dst 是否存在
    src.withdraw(amount)
    dst.deposit(amount)
''',
            "ledger.py": '''LEDGER = []


def log(txn_type, src, dst, amount, ok):
    # BUG: 只是定义了函数，从未被调用，交易不记日志
    LEDGER.append({
        "type": txn_type,
        "src": src,
        "dst": dst,
        "amount": amount,
        "ok": ok,
    })


def get_ledger():
    return LEDGER
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from account import Account, transfer
from ledger import get_ledger


def test_transfer_basic():
    a = Account("alice", 100)
    b = Account("bob", 50)
    transfer(a, b, 30)
    assert a.balance == 70
    assert b.balance == 80


def test_transfer_insufficient():
    a = Account("alice", 10)
    b = Account("bob", 50)
    try:
        transfer(a, b, 100)
        assert False, "should raise"
    except ValueError:
        pass
    assert a.balance == 10
    assert b.balance == 50


def test_ledger_recorded():
    a = Account("alice", 100)
    b = Account("bob", 50)
    transfer(a, b, 30)
    ledger = get_ledger()
    assert len(ledger) >= 1


if __name__ == "__main__":
    test_transfer_basic()
    test_transfer_insufficient()
    test_ledger_recorded()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "account.py",
        "bug_desc": "transfer 非原子（先扣后加可丢钱），未校验 src/dst；ledger 不记日志",
        "tags": ["logic", "multi_file", "atomicity"],
    },
    {
        "id": "config_v1",
        "name": "配置加载器",
        "files": {
            "config.py": '''import os
import json


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def load_config(path):
    # BUG: 路径遍历，../ 可逃出 configs 目录读取任意文件
    full_path = os.path.join(CONFIG_DIR, path)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path, data):
    full_path = os.path.join(CONFIG_DIR, path)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
''',
            "env_loader.py": '''import os


def load_env():
    env = {}
    for key, val in os.environ.items():
        if key.startswith("APP_"):
            env[key[4:].lower()] = val
    return env


def merge_config(config, env):
    # BUG: 配置覆盖环境变量，环境变量优先级应该更高
    merged = dict(env)
    merged.update(config)
    return merged
''',
            "tests.py": '''import sys, os, json
sys.path.insert(0, '.')
from config import load_config, save_config, CONFIG_DIR
from env_loader import load_env, merge_config


def test_load_config_basic():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    cfg = {"host": "localhost", "port": 8080}
    save_config("test_cfg.json", cfg)
    loaded = load_config("test_cfg.json")
    assert loaded == cfg


def test_env_priority():
    # 环境变量优先级必须高于配置文件
    config = {"host": "from-config", "port": 9000}
    env = {"host": "from-env", "debug": "true"}
    merged = merge_config(config, env)
    assert merged["host"] == "from-env", "env must override config"
    assert merged["debug"] == "true"


def test_path_traversal_blocked():
    # 在 CONFIG_DIR 外放探针文件，尝试用 ../ 逃出读取
    parent = os.path.dirname(CONFIG_DIR)
    probe = os.path.join(parent, "probe_secret.json")
    with open(probe, "w", encoding="utf-8") as f:
        json.dump({"secret": True}, f)
    blocked = False
    try:
        load_config("../probe_secret.json")
    except ValueError:
        blocked = True
    finally:
        if os.path.exists(probe):
            os.remove(probe)
    assert blocked, "path traversal (../) must be blocked with ValueError"


if __name__ == "__main__":
    test_load_config_basic()
    test_env_priority()
    test_path_traversal_blocked()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "config.py",
        "bug_desc": "load_config 允许路径遍历（../ 逃出目录）；merge_config 让配置文件覆盖环境变量，优先级反了",
        "tags": ["security", "hidden", "multi_file"],
    },
    {
        "id": "parser_v2",
        "name": "JSON 解析器",
        "files": {
            "json_parser.py": '''import re


def parse_json(text):
    """极简 JSON 解析器，只处理对象、字符串、数字、布尔、null"""
    text = text.strip()

    def parse_value(idx):
        while idx < len(text) and text[idx] in " \\t\\n\\r":
            idx += 1
        ch = text[idx]
        if ch == '{':
            return parse_object(idx)
        if ch == '"':
            return parse_string(idx)
        if ch == 't':
            return True, idx + 4
        if ch == 'f':
            return False, idx + 5
        if ch == 'n':
            return None, idx + 4
        m = re.match(r'-?\\d+(\\.\\d+)?', text[idx:])
        if m:
            num = float(m.group()) if '.' in m.group() else int(m.group())
            return num, idx + m.end()
        raise ValueError(f"unexpected char {ch!r} at {idx}")

    def parse_string(idx):
        # BUG: 转义字符处理错误，\\\\n \\\\t \\\\\\\\ \\\\" 都不正确
        assert text[idx] == '"'
        idx += 1
        chars = []
        while idx < len(text):
            ch = text[idx]
            if ch == '"':
                return ''.join(chars), idx + 1
            if ch == '\\\\':
                idx += 2
                continue
            chars.append(ch)
            idx += 1
        raise ValueError("unterminated string")

    def parse_object(idx):
        assert text[idx] == '{'
        idx += 1
        obj = {}
        while idx < len(text):
            while idx < len(text) and text[idx] in " \\t\\n\\r":
                idx += 1
            if text[idx] == '}':
                return obj, idx + 1
            key, idx = parse_string(idx)
            while idx < len(text) and text[idx] in " \\t\\n\\r":
                idx += 1
            assert text[idx] == ':'
            idx += 1
            val, idx = parse_value(idx)
            obj[key] = val
            while idx < len(text) and text[idx] in " \\t\\n\\r":
                idx += 1
            if text[idx] == ',':
                idx += 1
            elif text[idx] == '}':
                return obj, idx + 1
        raise ValueError("unterminated object")

    val, _ = parse_value(0)
    return val


if __name__ == "__main__":
    print(parse_json('{"a": 1, "b": "hello"}'))
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from json_parser import parse_json


def test_object():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_string_escape():
    assert parse_json('{"a": "hello\\\\nworld"}') == {"a": "hello\\nworld"}


def test_nested():
    assert parse_json('{"a": {"b": 2}}') == {"a": {"b": 2}}


def test_bool_null():
    assert parse_json('{"a": true, "b": null}') == {"a": True, "b": None}


def test_number():
    assert parse_json('{"a": 3.14}') == {"a": 3.14}


if __name__ == "__main__":
    test_object()
    test_string_escape()
    test_nested()
    test_bool_null()
    test_number()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "json_parser.py",
        "bug_desc": "parse_json 字符串转义处理错误（反斜杠n/t/引号），嵌套对象解析可能错",
        "tags": ["logic", "parser", "hidden"],
    },
    {
        "id": "graph_v2",
        "name": "图拓扑排序",
        "files": {
            "graph.py": '''from collections import defaultdict


class Graph:
    def __init__(self):
        self.adj = defaultdict(list)
        self.nodes = set()

    def add_node(self, n):
        self.nodes.add(n)

    def add_edge(self, src, dst):
        # BUG: 没检查 src/dst 是否是已注册节点，允许悬空边
        self.adj[src].append(dst)

    def neighbors(self, n):
        return self.adj.get(n, [])
''',
            "topo.py": '''from collections import deque


def topo_sort(graph):
    # BUG: 没有环检测，有环图会返回不完整结果而不报错
    indeg = {n: 0 for n in graph.nodes}
    for u in graph.nodes:
        for v in graph.neighbors(u):
            indeg[v] = indeg.get(v, 0) + 1

    q = deque([n for n in graph.nodes if indeg.get(n, 0) == 0])
    result = []
    while q:
        u = q.popleft()
        result.append(u)
        for v in graph.neighbors(u):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    return result  # 没检查 len(result) == len(graph.nodes)，无法发现环
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from graph import Graph
from topo import topo_sort


def test_linear():
    g = Graph()
    for n in ["A", "B", "C"]:
        g.add_node(n)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    order = topo_sort(g)
    assert order.index("A") < order.index("B") < order.index("C")


def test_cycle_detected():
    g = Graph()
    for n in ["A", "B", "C"]:
        g.add_node(n)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")  # 环
    # 有环必须检测并报错，不能返回不完整结果
    threw = False
    try:
        topo_sort(g)
    except ValueError:
        threw = True
    assert threw, "cycle must be detected and raise ValueError"


def test_independent_nodes():
    g = Graph()
    for n in ["X", "Y", "Z"]:
        g.add_node(n)
    order = topo_sort(g)
    assert set(order) == {"X", "Y", "Z"}


if __name__ == "__main__":
    test_linear()
    test_cycle_detected()
    test_independent_nodes()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "topo.py",
        "bug_desc": "topo_sort 无环检测（有环不报错），graph.add_edge 未校验节点存在",
        "tags": ["algorithm", "multi_file", "hidden"],
    },
    {
        "id": "tree_v1",
        "name": "二叉搜索树删除",
        "files": {
            "bst.py": '''class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node is None:
            return Node(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        else:
            node.right = self._insert(node.right, key)
        return node

    def delete(self, key):
        # BUG: 删除有左右两个子节点的节点时，前驱替换后没删除旧前驱节点，导致重复/树结构错
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            pred = self._max(node.left)
            node.key = pred.key
            # BUG: 忘了删除 node.left 里的 pred
            return node
        return node

    def _max(self, node):
        while node.right:
            node = node.right
        return node

    def inorder(self):
        res = []
        self._inorder(self.root, res)
        return res

    def _inorder(self, node, res):
        if node:
            self._inorder(node.left, res)
            res.append(node.key)
            self._inorder(node.right, res)
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from bst import BST


def test_insert_inorder():
    b = BST()
    for k in [5, 3, 7, 2, 4, 6, 8]:
        b.insert(k)
    assert b.inorder() == [2, 3, 4, 5, 6, 7, 8]


def test_delete_leaf():
    b = BST()
    for k in [5, 3, 7]:
        b.insert(k)
    b.delete(3)
    assert b.inorder() == [5, 7]


def test_delete_one_child():
    b = BST()
    for k in [5, 3, 7, 2]:
        b.insert(k)
    b.delete(3)
    assert b.inorder() == [2, 5, 7]


def test_delete_two_children():
    b = BST()
    for k in [5, 3, 7, 2, 4, 6, 8]:
        b.insert(k)
    b.delete(5)
    assert b.inorder() == [2, 3, 4, 6, 7, 8]
    assert len(b.inorder()) == len(set(b.inorder()))


if __name__ == "__main__":
    test_insert_inorder()
    test_delete_leaf()
    test_delete_one_child()
    test_delete_two_children()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "bst.py",
        "bug_desc": "BST.delete 两个子节点时用前驱替换但没删除旧前驱，导致重复节点/树结构错",
        "tags": ["data_structure", "logic"],
    },
    {
        "id": "queue_v1",
        "name": "有界队列",
        "files": {
            "queue.py": '''from collections import deque


class Queue:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self._data = deque()

    def push(self, item):
        # BUG: 不检查容量，无限制 push
        if len(self._data) >= self.capacity:
            raise OverflowError("queue full")
        self._data.append(item)

    def pop(self):
        # BUG: 空队列返回 None 而不是抛异常，调用者难以区分"空"和"元素为 None"
        if not self._data:
            return None
        return self._data.popleft()

    def __len__(self):
        return len(self._data)
''',
            "tests.py": '''import sys
sys.path.insert(0, '.')
from queue import Queue


def test_push_pop():
    q = Queue(3)
    q.push(1)
    q.push(2)
    assert q.pop() == 1
    assert q.pop() == 2


def test_capacity():
    q = Queue(2)
    q.push(1)
    q.push(2)
    try:
        q.push(3)
        assert False, "should raise OverflowError"
    except OverflowError:
        pass


def test_pop_empty_raises():
    q = Queue()
    try:
        q.pop()
        assert False, "should raise IndexError on empty"
    except IndexError:
        pass


def test_none_element():
    q = Queue()
    q.push(None)
    assert q.pop() is None


if __name__ == "__main__":
    test_push_pop()
    test_capacity()
    test_pop_empty_raises()
    test_none_element()
    print("All tests passed")
''',
        },
        "entry": "tests.py",
        "bug_file": "queue.py",
        "bug_desc": "Queue.pop 空队列返回 None 而非抛 IndexError，与合法 None 元素无法区分",
        "tags": ["logic"],
    },
]

class CodeEnv:
    def __init__(self, seed=1, size=1, depth=1, hidden=1, scaffold=1, resource=1,
                 use_tasks=True, champ=None, chain_ratio=0.0):
        self.rng = random.Random(seed)
        self.size = size
        self.depth = depth
        self.hidden = hidden
        self.scaffold = scaffold
        self.resource = resource
        self.use_tasks = use_tasks
        self.chain_ratio = chain_ratio
        if use_tasks:
            # 任务自举：golden 池 + bug 注入器，难度维度在 make_task 里生效。
            # champ 非 None 时世界进入对抗模式，挑克制该冠军的 bug。
            # chain_ratio > 0 时有这个概率抽到跨层隐含契约题（chains.py）。
            t = tasks.make_task(self.rng, {"size": size, "depth": depth,
                                          "hidden": hidden, "scaffold": scaffold,
                                          "resource": resource},
                                champ=champ, chain_ratio=chain_ratio)
            self.project = {"id": t["id"], "name": t["id"]}
            self.files = t["files"]
            self.entry = t["entry"]
            self.bug_file = t["bug_file"]
            self.bug_desc = t["bug_desc"]
            self.hidden_flag = t["hidden"]      # True = 盲改（不把报错喂给生成器）
            self.editable = t["editable"]
            self.counter = t.get("counter", "")  # 对抗模式：这题在克制冠军什么
        else:
            # 兼容旧死题库（探针/回归用）
            self.project = self.rng.choice(PROJECTS)
            self.files = dict(self.project["files"])
            self.entry = self.project["entry"]
            self.bug_file = self.project["bug_file"]
            self.bug_desc = self.project["bug_desc"]
            self.hidden_flag = hidden >= 2
            self.editable = list(self.files.keys())
        # 作弊计数：笔试图改白名单以外的文件（尤其测试入口）时 +1。
        # 不参与 fit 计算，只作为「评估器被攻击」的观测指标。
        self.cheat_attempts = 0
        self.cheat_files = []
        self._apply_difficulty()

    def _apply_difficulty(self):
        # tasks 路径的难度（size padding / hidden / scaffold）已在 make_task 里处理；
        # 旧 PROJECTS 路径在此补 size padding。
        if self.use_tasks:
            return
        if self.size >= 3:
            pad = "\n".join(f"# padding line {i}\n" for i in range(self.size * 3))
            self.files[self.bug_file] = self.files[self.bug_file] + pad
        # depth/hidden/scaffold/resource 维度语义在 submit/write_fix 中体现

    @staticmethod
    def run_tests(files_dict, entry, timeout=10.0):
        """在临时目录写入文件，运行测试入口，返回 (passed, error, ms, output)"""
        with tempfile.TemporaryDirectory() as td:
            for fname, content in files_dict.items():
                path = os.path.join(td, fname)
                d = os.path.dirname(path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
            entry_path = os.path.join(td, entry)
            t0 = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, entry_path],
                    capture_output=True, text=True, timeout=timeout,
                    cwd=td
                )
                ms = (time.time() - t0) * 1000
                passed = "All tests passed" in proc.stdout
                error = proc.stderr.strip()[:500] if proc.stderr else ""
                return passed, error, ms, proc.stdout
            except subprocess.TimeoutExpired:
                return False, "TIMEOUT", timeout * 1000, ""
            except Exception as e:
                return False, str(e)[:500], 0, ""

class Bot:
    """编程 agent：根据基因决定如何读代码、怎么改、什么时候回滚。"""
    def __init__(self, gene):
        self.gene = list(gene)
        (self.read_first, self.edit_scope, self.retry_cap, self.backtrack,
         self.test_cover, self.refactor, self.doc_first,
         self.parallel_try, self.rollback_speed, self.scaffold) = self.gene
        self._rng = random.Random()

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, v):
        self._rng = v

    def write_fix(self, env, base_error=None):
        """根据环境和基因，生成修复代码。这是「能写东西」的核心。

        路径优先级：
          1. LLM 生成器（gen.generate_patch）—— 外接编码模型当生成器，真写代码
          2. 查表模板（_generate_fix）—— 已验证的 few-shot / 兜底
          3. 回滚（backtrack 命中）—— 不修改
        """
        if self.rng.random() < self.backtrack:
            # 回滚：不做修改
            return dict(env.files)

        # ---- 主路径：本地脑思考 → 外部 LLM 笔书写 ----
        # base_error 已由 submit 按 hidden 维度处理（盲改时为 None，逼笔盲修）
        test_error = base_error
        patch = gen.generate_patch(env.project, env.files, env.bug_desc,
                                   test_error, self, bug_file=env.bug_file)
        patch = self._vet_patch(patch, env)
        if patch:
            new_files = dict(env.files)
            for fname, content in patch.items():
                if self.edit_scope > 0.7:
                    content = self._refactor(content)
                new_files[fname] = content
            if self.scaffold > 0.7 and self.rng.random() < 0.3:
                new_files["extra_check.py"] = self._extra_check(env)
            return new_files

        # ---- fallback：查表模板 ----
        buggy_code = env.files[env.bug_file]
        fixed = self._generate_fix(buggy_code, env)

        new_files = dict(env.files)
        if isinstance(fixed, dict):
            for fname, content in fixed.items():
                if self.edit_scope > 0.7:
                    content = self._refactor(content)
                new_files[fname] = content
        else:
            fixed_code = fixed
            if self.edit_scope > 0.7:
                fixed_code = self._refactor(fixed_code)
            new_files[env.bug_file] = fixed_code

        # scaffold 高：可能添加辅助测试文件
        if self.scaffold > 0.7 and self.rng.random() < 0.3:
            new_files["extra_check.py"] = self._extra_check(env)

        return new_files

    def _vet_patch(self, patch, env):
        """白名单过滤：只允许改 env.editable 里列出的文件。

        这条不是洁癖，是堵一个能直接刷满分的洞。
        `env.editable` 早就写好了（tasks.make_task 里等于 [bug_file]），
        但 write_fix 从来没查过它 —— 也就是说笔只要返回一格
        `# file: tests.py` 并把断言删掉，或者写成 `print("All tests passed")`，
        fit 立刻 1.0。这正是 coding agent 最经典的作弊姿势：
        **测试不过就改测试**（业界文档里点名的那种）。

        越界的文件名记在 env.cheat_files 上，不静默丢弃 ——
        它是「评估器正在被攻击」的早期信号，比 fit 更值得盯。
        """
        if not patch:
            return None
        allowed = set(getattr(env, "editable", None) or env.files.keys())
        kept = {f: c for f, c in patch.items() if f in allowed}
        refused = [f for f in patch if f not in allowed]
        if refused:
            env.cheat_attempts = getattr(env, "cheat_attempts", 0) + len(refused)
            env.cheat_files = (list(getattr(env, "cheat_files", [])) + refused)[-5:]
        if not kept:
            return None   # 全被拒 = 这笔等于没写成，交给查表兜底
        return kept

    def _generate_fix(self, code, env):
        """根据 bug 描述生成修复代码。模拟 LLM 写代码能力。"""
        project_id = env.project["id"]
        bug_desc = env.bug_desc

        # 根据已知 bug 模式生成修复（可扩展为真实 LLM 调用）
        if project_id == "calc_v1":
            if "除以零" in bug_desc:
                return code.replace(
                    "    if b == 0:\n        return None",
                    "    if b == 0:\n        raise ZeroDivisionError('division by zero')"
                )
        elif project_id == "calc_v2":
            if "除以零" in bug_desc:
                return code.replace(
                    "    if b == 0:\n        return float('inf')",
                    "    if b == 0:\n        raise ZeroDivisionError('division by zero')"
                )
        elif project_id == "cache_v1":
            if "未正确更新订单" in bug_desc or "LRU" in bug_desc:
                old = '''    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return -1'''
                new = '''    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1'''
                return code.replace(old, new)
        elif project_id == "parser_v1":
            if "import re" in bug_desc or "NameError" in bug_desc:
                return "import re\n" + code
        elif project_id == "graph_v1":
            if "可变默认参数" in bug_desc or "共享状态" in bug_desc:
                old = '''def dfs(graph, start, visited=set()):
    # BUG: 可变默认参数，visited 在 def 时创建一次，所有调用共享同一 set
    visited.add(start)'''
                new = '''def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)'''
                return code.replace(old, new)

        # ---------- 新项目 fix ----------
        elif project_id == "auth_v1":
            fixes = {}
            fixes["auth.py"] = code.replace(
                "    return stored == password",
                "    return hmac.compare_digest(stored, password)"
            )
            session_code = env.files["session.py"]
            fixes["session.py"] = session_code.replace(
                'session_id = "fixed-session"',
                'session_id = str(uuid.uuid4())'
            )
            return fixes

        elif project_id == "bank_v1":
            fixed = code.replace(
                "def transfer(src, dst, amount):\n    # BUG: 非原子，先扣后加，中间失败钱就丢了；没校验 dst 是否存在\n    src.withdraw(amount)\n    dst.deposit(amount)",
                "from ledger import log\n\n\ndef transfer(src, dst, amount):\n    if src is None or dst is None:\n        raise ValueError('account not found')\n    src.withdraw(amount)\n    dst.deposit(amount)\n    log(\"transfer\", src.owner, dst.owner, amount, True)"
            )
            return fixed

        elif project_id == "config_v1":
            fixed = code.replace(
                "def load_config(path):\n    # BUG: 路径遍历，../ 可逃出 configs 目录读取任意文件\n    full_path = os.path.join(CONFIG_DIR, path)\n    with open(full_path, \"r\", encoding=\"utf-8\") as f:\n        return json.load(f)",
                "def load_config(path):\n    full_path = os.path.normpath(os.path.join(CONFIG_DIR, path))\n    if not full_path.startswith(os.path.normpath(CONFIG_DIR)):\n        raise ValueError('path traversal blocked')\n    with open(full_path, \"r\", encoding=\"utf-8\") as f:\n        return json.load(f)"
            )
            fixed = fixed.replace(
                "def save_config(path, data):\n    full_path = os.path.join(CONFIG_DIR, path)\n    with open(full_path, \"w\", encoding=\"utf-8\") as f:\n        json.dump(data, f)",
                "def save_config(path, data):\n    full_path = os.path.join(CONFIG_DIR, path)\n    os.makedirs(os.path.dirname(full_path), exist_ok=True)\n    with open(full_path, \"w\", encoding=\"utf-8\") as f:\n        json.dump(data, f)"
            )
            env_code = env.files.get("env_loader.py", "")
            env_fixed = env_code.replace(
                "    merged = dict(env)\n    merged.update(config)\n    return merged",
                "    merged = dict(config)\n    merged.update(env)\n    return merged"
            )
            return {"config.py": fixed, "env_loader.py": env_fixed}

        elif project_id == "parser_v2":
            fixed = code.replace(
                "            if ch == '\\\\':\n                idx += 2\n                continue",
                "            if ch == '\\\\':\n                idx += 1\n                if idx < len(text):\n                    nxt = text[idx]\n                    esc = {'n': '\\n', 't': '\\t', 'r': '\\r', '\\\\': '\\\\', '\"': '\"'}\n                    chars.append(esc.get(nxt, nxt))\n                    idx += 1\n                    continue"
            )
            return fixed

        elif project_id == "graph_v2":
            fixed = code.replace(
                "    return result  # 没检查 len(result) == len(graph.nodes)，无法发现环",
                "    if len(result) != len(graph.nodes):\n        raise ValueError(\"cycle detected in graph\")\n    return result"
            )
            return fixed

        elif project_id == "tree_v1":
            fixed = code.replace(
                "            node.key = pred.key\n            # BUG: 忘了删除 node.left 里的 pred\n            return node",
                "            node.key = pred.key\n            node.left = self._delete(node.left, pred.key)\n            return node"
            )
            return fixed

        elif project_id == "queue_v1":
            fixed = code.replace(
                "    def pop(self):\n        # BUG: 空队列返回 None 而不是抛异常，调用者难以区分\"空\"和\"元素为 None\"\n        if not self._data:\n            return None\n        return self._data.popleft()",
                "    def pop(self):\n        if not self._data:\n            raise IndexError(\"pop from empty queue\")\n        return self._data.popleft()"
            )
            return fixed

        # 通用 fallback：返回原代码（无法自动修复）
        return code

    def _refactor(self, code):
        """简单重构：添加类型注解和文档"""
        return code + "\n\n# 已重构：补充类型注解与文档\n"

    def _extra_check(self, env):
        return f"# 额外检查 for {env.project['name']}\n"

    def submit(self, env):
        """提交代码并运行测试，返回 (passed, error, ms, output, files)"""
        # baseline：先跑一次 buggy 测试拿报错（read_first 基因想看时）
        base_err = None
        if gen.LLM_ENABLED and self.read_first > 0.5:
            try:
                _, base_err, _, _ = env.run_tests(env.files, env.entry, timeout=20)
            except Exception:
                base_err = None
        # hidden 维度生效：盲改（hidden_flag=True）时不把报错喂给生成器
        if getattr(env, "hidden_flag", False):
            base_err = None
        new_files = self.write_fix(env, base_error=base_err)
        timeout = 5.0 if env.resource >= 2 else 30.0
        passed, error, ms, output = env.run_tests(new_files, env.entry, timeout=timeout)

        # resource 维度：改动行数预算。
        # 原来这一维只控制测试超时（5s vs 30s），可绝大多数题 5s 都用不完，
        # 等于没约束。这里补一条真实工程里最常见的约束：**最小改动**。
        # 重写整个文件碰巧对了，不算本事 —— 它没定位到 bug。
        if passed and getattr(env, "resource", 0) >= 2:
            limit = 0.6 if env.resource == 2 else 0.3
            old_src = env.files.get(env.bug_file, "")
            new_src = new_files.get(env.bug_file, "")
            budget = max(4, int(len(old_src.splitlines()) * limit))
            used = change_lines(old_src, new_src)
            if used > budget:
                passed = False
                error = "改动超预算：%d 行 > 上限 %d 行（resource=%d）" % (used, budget, env.resource)
        return passed, error, ms, output, new_files

    @classmethod
    def random(cls, rng):
        g = [rng.random() for _ in range(10)]
        g[2] = int(rng.integers(1, 11))   # retry_cap
        g[7] = int(rng.integers(1, 6))     # parallel_try
        for i, (lo, hi) in enumerate(BOUNDS):
            g[i] = max(lo, min(hi, g[i]))
        return cls(g)


def random_gene(rng):
    g = [rng.random() for _ in range(10)]
    g[2] = int(rng.integers(1, 11))
    g[7] = int(rng.integers(1, 6))
    for i, (lo, hi) in enumerate(BOUNDS):
        g[i] = max(lo, min(hi, g[i]))
    return g