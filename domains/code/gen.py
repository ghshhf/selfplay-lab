"""本地编程模型（脑） + 外部 LLM（笔）。

架构定位（主人 2026-10-29 纠正）：
- **本地训练的编程模型（code_duel 进化系统 / Brain）= 脑**：负责思考、诊断、决策。
  读源码 + bug 描述 + 测试报错，基于基因（思考风格参数）产出「修复指令」。
- **外部 LLM = 笔**：只负责按脑给的指令把代码写出来（书写），本地跑测试验证。
  它不参与决策，只执行。

所以本模块分两层：
  - brain_plan() ：本地大脑思考，产出结构化修复指令（不写代码）。
  - pen_write() ：LLM 当笔，按指令书写代码。
  - generate_patch() ：编排 脑→笔→返回 patch；进化系统训练的是脑的思考策略基因。

复用 device-fleet 已验证的 compute 模块拿端点/key（8787 zen-gate）。

开关：CODE_DUEL_LLM=1 才真调 LLM 当笔；否则 Brain 只能靠本地知识库（_generate_fix 降级）自己写。
"""

import os
import sys
import json
import re
import time
import hashlib
import urllib.request
import urllib.error

# ---- 接入 device-fleet 的 compute（已验证的端点/key 管理） ----
_FLEET_DIR = os.environ.get("CODE_DUEL_FLEET_DIR", r"D:/code/device-fleet")
if _FLEET_DIR not in sys.path:
    sys.path.insert(0, _FLEET_DIR)

LLM_ENABLED = os.environ.get("CODE_DUEL_LLM", "1") == "1"

_compute = None
_compute_tried = False
_ep = None
_ep_cooldown = {}   # endpoint id -> 坏端点冷却到的 Unix 时间戳


class EnvEndpoint:
    """不依赖 device-fleet 的最小端点：全靠环境变量。

    CODE_DUEL_API_BASE   到 /v1 为止（如 https://api.example.com/v1）
    CODE_DUEL_API_KEY    可空（pollinations 这类免 key 端点）
    CODE_DUEL_MODEL      模型名，可空

    存在的理由：这个仓库要能被别人拉下来直接跑。原实现硬依赖一台机器上的
    device-fleet 端点池，外人拿不到，等于开箱即死。
    """

    __slots__ = ("id", "title", "base_url", "model", "need_key", "key", "enabled")

    def __init__(self):
        self.id = "env"
        self.title = "从环境变量配置的端点"
        self.base_url = (os.environ.get("CODE_DUEL_API_BASE") or "").rstrip("/")
        self.model = os.environ.get("CODE_DUEL_MODEL") or ""
        self.key = os.environ.get("CODE_DUEL_API_KEY") or ""
        self.need_key = False
        self.enabled = bool(self.base_url)

    def url(self, path):
        return self.base_url + "/" + path.lstrip("/")

    def has_key(self):
        return bool(self.key)

    def ready(self):
        return self.enabled


def _hdrs(ep):
    """请求头。不依赖 compute —— env 端点也要能走这条路。"""
    h = {"Content-Type": "application/json"}
    if getattr(ep, "key", ""):
        h["Authorization"] = "Bearer " + ep.key
    return h


def _candidate_eps():
    """按可用性排序的候选端点（Endpoint 或 EnvEndpoint）。

    笔（LLM）必须稳定能写出，不能吊死在一个被限流的端点上：
    - 环境变量配的端点排最前（用户显式指定，优先于内置池）；
    - pollinations / ovh 是免 key、直连可达的云端端点；
    - zen-gate 是本机 40B 网关，但它背后的免费模型会触发
      PROMOTION_MODEL_LIMIT_REACHED（额度耗尽）硬限流，所以放最后。

    某个端点若连续失败，会被冷却 60s，期间跳过，自动换下一个。
    """
    global _compute, _compute_tried
    now = time.time()
    cands = []

    env_ep = EnvEndpoint()
    if env_ep.ready() and _ep_cooldown.get(env_ep.id, 0) < now:
        cands.append(env_ep)

    if not _compute_tried:
        _compute_tried = True
        try:
            import fleet.compute as cp
            _compute = cp
        except Exception as e:
            print("[gen] 没有 device-fleet compute（%s），只用环境变量端点" % e)

    if _compute is not None:
        for eid in ("pollinations", "ovh", "zen-gate"):
            e = _compute.get(eid)
            if e is not None and e.ready() and _ep_cooldown.get(eid, 0) < now:
                cands.append(e)
    return cands


def _ensure_ep():
    """兼容旧调用：返回第一个候选（没有就 None）。"""
    cands = _candidate_eps()
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# 缓存：同一 (题, bug描述, 是否带报错, 脑指令hash) 只烧一次 token
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "gen_cache.json")
_cache = None


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
    _cache = {}
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def _save_cache():
    if _cache is None:
        return
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 本地大脑（Brain）：思考、诊断、决策 —— 产出「修复指令」
# ---------------------------------------------------------------------------

# 脑的本地知识库：关键词 -> (修复意图, 目标文件提示, 补充提示)
# 这是 Brain 的先验经验，进化系统训练的是「如何调用这些经验 + 基因参数」。
_DIAGNOSE_RULES = [
    # ---- tasks.py 各 bug 族（按 desc 关键词）----
    ("返回 None 而不是抛", "zerodiv", "除数为零时抛出 ZeroDivisionError，而不是返回 None", None),
    ("多算了 1", "offbyone", "修正 off-by-one：add 多加了 1，检查循环或边界", None),
    ("少算了 1", "mul", "修正 multiply 的运算逻辑错误（少算 1）", None),
    ("移到最近位置", "lru", "get 命中后把 key 移到访问顺序末尾（最近使用放最后）", None),
    ("容量判断", "cap", "容量边界修正：满时用 >= 触发淘汰，而非 >", None),
    ("二分查找", "bisect", "修正二分查找边界：循环条件应为 lo<=hi，未命中返回 -1", None),
    ("返回 hi 而非 -1", "retneg", "没找到时返回 -1 而非 hi", None),
    ("丢失原始顺序", "order", "去重时保留原始顺序（不能用 set 直接转 list 破坏顺序）", None),
    ("少处理最后一个", "loop", "循环边界修正：处理到最后一个元素", None),
    ("转小写", "lower", "比较前统一转小写以消除大小写敏感", None),
    ("过滤非字母数字", "symbol", "过滤掉非字母数字字符后再判断回文", None),
    ("基准错误", "base", "修正 fib 基准条件：fib(0)=0, fib(1)=1", None),
    ("fib(n-2)+fib(n-3)", "recur", "修正递归式，应为 fib(n-1)+fib(n-2)", None),
    ("不检查是否匹配", "match", "遇到右括号时检查栈顶是否匹配，不匹配则非法", None),
    ("配对表写错", "pair", "修正括号配对表：) 对应 (，] 对应 [，} 对应 {", None),
    # ---- 旧 PROJECTS（保持兼容）----
    ("NameError", "re", "在文件顶部补充缺失的 import（如 re/hmac）", None),
    ("除以零", "zerodiv", "除数为零时抛出 ZeroDivisionError，而不是返回 None 或 inf", None),
    ("ZeroDivisionError", "zerodiv", "除数为零时抛出 ZeroDivisionError", None),
    ("LRU", "lru", "get 命中后把 key 移到访问顺序末尾（LRU 语义）", None),
    ("路径遍历", "traversal", "用 os.path.normpath 规范化路径并校验前缀，阻止 ../ 逃逸", None),
    ("traversal", "traversal", "用 normpath 规范化并校验前缀，阻止 ../ 逃逸", None),
    ("环", "cycle", "拓扑排序必须检测环：结果节点数不等于总节点数时抛出 ValueError", None),
    ("cycle", "cycle", "拓扑排序检测环，发现环抛异常而非返回不完整结果", None),
    ("共享状态", "mutdefault", "把可变默认参数改成函数内 if visited is None: visited=set()", None),
    ("可变默认参数", "mutdefault", "把可变默认参数改成函数内初始化", None),
    ("非原子", "atomic", "先校验账户存在，再 withdraw 再 deposit，保证原子", None),
    ("原子", "atomic", "转账前校验账户存在，保证扣款与存款原子", None),
    ("空队列", "emptyq", "空队列 pop 时抛 IndexError，而不是返回 None", None),
    ("转义", "escape", "处理字符串转义：\\\\n 换行、\\\\t 制表符、\\\\\\\\ 反斜杠、\\\\\" 引号", None),
    ("BST", "bst", "BST 用前驱替换后，递归删除左子树中的前驱节点", None),
    ("前驱", "bst", "BST 用前驱替换后，递归删除左子树中的前驱节点", None),
    ("优先级", "merge", "环境变量优先级高于配置文件：merged = dict(env); merged.update(config)", None),
    ("环境", "merge", "环境变量覆盖配置文件：先 copy env 再 update config", None),
    ("未定义", "import", "在文件顶部补充缺失的 import", None),
]


def _find_import_target(files):
    """脑的辅助：找最可能是 import 目标的文件（含 bug_desc 关键词或最短的 .py）。"""
    cands = [f for f in files if f.endswith(".py") and f != "tests.py"]
    if not cands:
        return None
    # 优先 main/逻辑文件而非测试
    for f in cands:
        if "test" not in f:
            return f
    return cands[0]


def brain_plan(project, files, bug_desc, test_error, gene, bug_file=None):
    """本地大脑思考：读信息，基于基因决策，产出修复指令（不写代码）。

    返回 dict：
      intent   : str  修复意图（自然语言，交给笔去执行）
      target   : str|None  目标文件（None=让笔自己判断）
      hints    : list[str] 补充提示
      add_test : bool  是否让笔顺带加一个辅助测试文件（基因 scaffold 决策）
      multi    : bool  是否允许多文件修改（基因 edit_scope 决策）
    """
    # ---- 基因：思考风格参数（进化训练的对象）----
    read_first = bool(getattr(gene, "read_first", 0.0) > 0.5)   # 是否先看报错再想
    edit_scope = float(getattr(gene, "edit_scope", 0.0))          # 改动范围（大=多文件）
    scaffold = bool(getattr(gene, "scaffold", 0.0) > 0.5)        # 是否加辅助测试

    # ---- 脑的诊断：从 bug_desc + 报错 提取意图 ----
    text = (bug_desc or "") + " " + (test_error or "")
    intent = "根据 bug 描述修复逻辑错误"
    tag = None
    target_hint = None
    hints = []
    for kw, t, intent_txt, tgt in _DIAGNOSE_RULES:
        if kw in text:
            intent = intent_txt
            tag = t
            if tgt:
                target_hint = tgt
            break

    # 没从关键词命中，但脑仍可基于 bug_desc 给一个泛化意图
    if tag is None and bug_desc:
        intent = "分析并修复以下 bug：" + bug_desc

    # ---- 基因决策：目标文件（脑的定位思考）----
    # 优先用任务自举已定位的 bug 文件，其次用诊断提示推断
    target = bug_file
    if target is None and target_hint:
        for f in files:
            if target_hint in f:
                target = f
                break
    if target is None and tag == "import":
        target = _find_import_target(files)

    # ---- 基因决策：多文件 / 加测试 ----
    multi = bool(edit_scope > 0.5)
    add_test = bool(scaffold)

    plan = {
        "intent": intent,
        "target": target,
        "hints": hints,
        "add_test": add_test,
        "multi": multi,
    }
    # 思考过程日志（让主人看到本地模型在当脑）
    print("[brain] 思考: intent=%s | target=%s | 多文件=%s | 加测试=%s | 报错=%s"
          % (intent[:40], target, multi, add_test,
             ("有" if (read_first and test_error) else "无/盲")))
    return plan


# ---------------------------------------------------------------------------
# 外部 LLM（笔）：只按脑的指令书写代码
# ---------------------------------------------------------------------------
_PEN_SYSTEM = (
    "你是一个资深 Python 工程师，但在本任务里你只是「笔」——不负责思考修复方案，"
    "只负责按下面给出的【修复指令】把代码改好并写出来。\n"
    "你已经收到一份修复指令（由分析大脑给出）和项目源码。请严格按指令执行：\n"
    "1. 只输出【需要修改的文件】的完整新内容，格式：\n"
    "   ```python\n   # file: 文件名.py\n   <该文件完整新代码>\n   ```\n"
    "2. 输出文件必须包含该文件【全部】代码（不能只给 diff/片段）；\n"
    "3. 不要另起修复方案、不要解释、不要输出 ``` 之外的内容；\n"
    "4. 如果指令要求加辅助测试文件，就额外输出一个 # file: extra_check.py；\n"
    "5. 修复要真解决问题，不要删测试、注释断言、或 return 假值绕过。"
)


def _pen_user(plan, project, files, bug_desc):
    src_parts = ["# ===== file: %s =====\n%s" % (f, c) for f, c in files.items()]
    u = "【修复指令】\n"
    u += "意图：%s\n" % plan["intent"]
    if plan["target"]:
        u += "目标文件：%s\n" % plan["target"]
    else:
        u += "目标文件：请自行判断（可多文件）\n"
    if plan["hints"]:
        u += "补充提示：%s\n" % "; ".join(plan["hints"])
    if plan["add_test"]:
        u += "额外要求：顺带生成一个 extra_check.py 做辅助校验。\n"
    u += "\n【Bug 描述】%s\n\n" % (bug_desc or "")
    u += "【项目源文件】\n" + "\n\n".join(src_parts) + "\n\n请按指令输出修复后的文件。"
    return u


def _chat_one(ep, system, user, timeout):
    """向单个端点发一次 chat。成功返回 content，失败抛异常。"""
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 2200,
        "stream": False,
    }
    if ep.model:
        payload["model"] = ep.model
    req = urllib.request.Request(
        ep.url("chat/completions"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers=_hdrs(ep),
    )
    raw = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
    d = json.loads(raw)
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    if not content and msg.get("reasoning_content"):
        content = msg["reasoning_content"]
    if not content:
        raise RuntimeError("模型返回空 content（可能抽风/被截断）")
    return content


def chat(system, user, timeout=90):
    """笔的通信：遍历候选端点，遇限流/网关/连接错误自动换下一个。

    返回 content；所有候选都失败则抛 RuntimeError。
    """
    cands = _candidate_eps()
    if not cands:
        raise RuntimeError("没有可用端点（CODE_DUEL_LLM 未开或端点都不可用）")
    last_err = None
    for ep in cands:
        try:
            content = _chat_one(ep, system, user, timeout)
            return content
        except Exception as e:
            last_err = str(e)
            # 限流(429)/网关(5xx)/连接类错误：冷却这个端点，换下一个
            tag = last_err[:64]
            if ("429" in tag or tag[:3] in ("400", "500", "502", "503", "504")
                    or "连接" in tag or "Connection" in tag or "refused" in tag):
                _ep_cooldown[ep.id] = time.time() + 60
                print("[gen] 端点 %s 暂不可用（%s），冷却 60s 换下一个" % (ep.id, tag))
            continue
    raise RuntimeError("所有候选端点均失败：%s" % last_err)


def _parse_patch(reply):
    """从回复里提取 `# file: xxx.py` 标记的完整文件代码，返回 {fname: code}。

    兼容两种写法（pollinations 两种都来）：
      · 标准：```python\n# file: a.py\n...```
      · 裸写：直接 `# file: a.py\n...`（没有 ``` 包裹）
    标记也可能不在块的最开头（前面有空行/注释），所以用 search 而非 match。
    """
    out = {}
    blocks = re.findall(r"```[a-zA-Z]*\s*(.*?)```", reply, re.S)
    if not blocks:
        blocks = [reply]          # 没有 ``` 包裹，把整段当成一个块
    for b in blocks:
        m = re.search(r"#\s*(?:file:\s*)?([\w./\\-]+\.py)\s*\n(.*)", b, re.S)
        if m:
            out[m.group(1)] = m.group(2).rstrip() + "\n"
    return out


def pen_write(plan, project, files, bug_desc, gene, timeout=90):
    """LLM 当笔：按脑的指令书写代码。返回 {fname: code} 或 None。"""
    retry_cap = int(getattr(gene, "retry_cap", 1))
    retry_cap = max(1, min(10, retry_cap))
    user = _pen_user(plan, project, files, bug_desc)
    last_err = None
    for attempt in range(retry_cap):
        try:
            reply = chat(_PEN_SYSTEM, user, timeout=timeout)
            patched = _parse_patch(reply)
            if patched:
                return patched
            last_err = "解析不到 # file: 块"
        except Exception as e:
            last_err = str(e)
            # 指数退避：应对 zen-gate 限流(429)/网关抖动(502)，给端点恢复时间
            time.sleep(min(12, 1.5 * (2 ** attempt)))
    print("[gen] 笔书写失败（%d 次）：%s" % (retry_cap, last_err))
    return None


# ---------------------------------------------------------------------------
# 对外：脑→笔 编排
# ---------------------------------------------------------------------------
def generate_patch(project, files, bug_desc, test_error, gene, entry=None, bug_file=None):
    """本地大脑思考 + 外部 LLM 书写。

    1. 脑（本地）：brain_plan 产出修复指令（思考/决策）。
    2. 笔（LLM）：pen_write 按指令写代码（书写）。
    若 LLM 不可用，返回 None，由调用方回退到本地知识库（_generate_fix，脑自己写）。
    """
    if not LLM_ENABLED:
        return None
    ep = _ensure_ep()
    if ep is None:
        return None

    # ---- 脑思考 ----
    plan = brain_plan(project, files, bug_desc, test_error, gene, bug_file=bug_file)

    # 缓存键：含脑指令 hash（同一题同一指令只烧一次 token）
    plan_hash = hashlib.md5(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8]
    with_error = bool(test_error)
    key = "%s|%s|%d|%s" % (project["id"], bug_desc, 1 if with_error else 0, plan_hash)
    cache = _load_cache()
    if key in cache:
        return cache[key] or None

    # ---- 笔书写 ----
    patch = pen_write(plan, project, files, bug_desc, gene, timeout=90)
    cache[key] = patch
    _save_cache()
    return patch


def endpoint_info():
    ep = _ensure_ep()
    if ep is None:
        return {"enabled": LLM_ENABLED, "ep": None}
    return {"enabled": LLM_ENABLED, "id": ep.id, "base": ep.base_url, "model": ep.model}


if __name__ == "__main__":
    os.environ["CODE_DUEL_LLM"] = "1"
    import env
    p = next(x for x in env.PROJECTS if x["id"] == "calc_v1")
    e = env.CodeEnv(seed=1)
    e.project = p
    e.files = dict(p["files"])
    e.bug_file = p["bug_file"]
    e.bug_desc = p["bug_desc"]
    e.entry = p["entry"]
    print("endpoint:", endpoint_info())
    patch = generate_patch(p, e.files, e.bug_desc, None,
                           type("G", (), {"read_first": 0.0, "retry_cap": 3,
                                          "edit_scope": 0.0, "scaffold": 0.0})())
    print("patch keys:", list(patch.keys()) if patch else None)
    if patch:
        for k, v in patch.items():
            print("--- %s ---" % k)
            print(v[:400])
