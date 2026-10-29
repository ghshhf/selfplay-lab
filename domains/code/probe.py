"""探针模式（冠军扫描） —— 固定最优「运用生成器」基因，扫描全部题目，输出三层对比报告。

回答主人质疑「每次都这么快吗？真的这么强吗？还是题太简单了？」：
  - buggy 层：强制回滚（不改代码），确认 bug 真的存在（应 FAIL）。
  - table 层：关 LLM，走查表模板（背答案），确认查表基线。
  - llm 层  ：开 LLM 生成器（真写代码），确认生成器独立能修多少。

报告三张：
  - 难度曲线：按 difficulty 分档，各档三层的通过率。
  - 平衡性  ：永不过（无效题）/ 秒过（太简单，table 和 llm 都过的简单题）。
  - exploit  ：可疑 fix 信号（动了 tests.py / 出现 return None 假值 / 注释断言）。

用法：
  CODE_DUEL_LLM=1 python probe.py        # 跑完整三层（llm 层需要 LLM）
  python probe.py                        # 只跑 buggy + table（无需 LLM）
"""

import os
import sys
import json
import time
import datetime

import env
import gen

# 固定基因：read_first=1（读报错喂生成器）、retry_cap=6、backtrack=0（不回滚）、
# edit_scope=0、scaffold=0、parallel_try=1。这是「最优运用生成器」的策略。
GENE = [1.0, 0.0, 6, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0]

# 题目难度分级（1=单点逻辑，2=多文件/算法，3=安全/原子性/隐蔽）
DIFFICULTY = {
    "calc_v1": 1, "calc_v2": 1, "parser_v1": 1, "queue_v1": 1,
    "graph_v1": 2, "cache_v1": 2, "tree_v1": 2, "auth_v1": 2,
    "config_v1": 3, "bank_v1": 3, "parser_v2": 3, "graph_v2": 3,
}


def _setup(p, seed=1):
    e = env.CodeEnv(seed=seed)
    e.project = p
    e.files = dict(p["files"])
    e.bug_file = p["bug_file"]
    e.bug_desc = p["bug_desc"]
    e.entry = p["entry"]
    return e


def run_buggy(p):
    """强制回滚（不改代码），bug 应存在 → 返回 True 表示 buggy 正确判败。"""
    e = _setup(p)
    bot = env.Bot([0.0, 0.0, 1, 1.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0])  # backtrack=1
    passed, error, ms, o, f = bot.submit(e)
    return (not passed), error


def run_table(p):
    """关 LLM，走查表模板。"""
    gen.LLM_ENABLED = False
    e = _setup(p)
    bot = env.Bot(GENE)
    passed, error, ms, o, f = bot.submit(e)
    return passed, error, f


def run_llm(p):
    """开 LLM 生成器，屏蔽查表 fallback（隔离测生成器）。"""
    gen.LLM_ENABLED = True
    orig = env.Bot._generate_fix
    env.Bot._generate_fix = lambda self, code, e: code  # 返回原代码=无效 fallback
    e = _setup(p)
    bot = env.Bot(GENE)
    passed, error, ms, o, f = bot.submit(e)
    env.Bot._generate_fix = orig
    return passed, error, f


def _suspicious(patch_files, orig_files):
    """exploit 信号：真改了测试文件 / 注释掉断言。

    注意：LLM 常把未修改的文件（含 tests.py）原样完整输出，这不算 exploit，
    只有输出的测试文件内容与原始不同（删断言/改期望）才算可疑。
    """
    sig = []
    for fn, content in patch_files.items():
        if "test" in fn and fn in orig_files:
            if content.strip() != orig_files[fn].strip():
                sig.append("真改了测试文件:%s" % fn)
    # 注释掉的断言（无论哪个文件）
    blob = "\n".join(patch_files.values())
    if "# assert" in blob or "assert #" in blob:
        sig.append("出现被注释的断言（疑似绕过）")
    return sig


def main():
    os.makedirs("reports", exist_ok=True)
    llm_on = os.environ.get("CODE_DUEL_LLM", "0") == "1"
    print("探针启动  LLM层=%s  题目=%d" % (llm_on, len(env.PROJECTS)))

    rows = []
    for p in env.PROJECTS:
        pid = p["id"]
        diff = DIFFICULTY.get(pid, 2)
        buggy_fail, berr = run_buggy(p)
        table_pass, terr, tfiles = run_table(p)
        llm_pass, lerr, lfiles = (False, "LLM 未启用", {}) if not llm_on else run_llm(p)

        patch_files = lfiles if (llm_on and llm_pass) else tfiles
        sig = _suspicious(patch_files, p["files"]) if (llm_pass or table_pass) else []

        rows.append({
            "id": pid,
            "difficulty": diff,
            "tags": p.get("tags", []),
            "buggy_fail": buggy_fail,
            "table_pass": table_pass,
            "llm_pass": llm_pass,
            "llm_patch_files": list(lfiles.keys()) if llm_on else [],
            "suspicious": sig,
        })
        print("%-12s d%d  buggy_fail=%-5s table=%-5s llm=%-5s %s" % (
            pid, diff, buggy_fail, table_pass, llm_pass,
            ("⚠ " + ";".join(sig)) if sig else ""))

    # ---- 汇总 ----
    n = len(rows)
    valid = [r for r in rows if r["buggy_fail"]]           # 有效题（bug 真存在）
    n_valid = len(valid)
    table_ok = sum(1 for r in valid if r["table_pass"])
    llm_ok = sum(1 for r in valid if r["llm_pass"])
    # 平衡性：table 和 llm 都过的简单题（difficulty<=1）
    trivial = [r["id"] for r in valid if r["table_pass"] and r["llm_pass"] and r["difficulty"] <= 1]
    invalid = [r["id"] for r in rows if not r["buggy_fail"]]  # 无效题
    exploits = [r["id"] for r in rows if r["suspicious"]]

    # 难度曲线：按档统计三层通过率
    curve = {}
    for d in (1, 2, 3):
        grp = [r for r in valid if r["difficulty"] == d]
        if not grp:
            continue
        curve[d] = {
            "n": len(grp),
            "buggy_fail": sum(1 for r in grp if r["buggy_fail"]),
            "table_pass": sum(1 for r in grp if r["table_pass"]),
            "llm_pass": sum(1 for r in grp if r["llm_pass"]),
        }

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "llm_layer": llm_on,
        "endpoint": gen.endpoint_info() if llm_on else None,
        "gene": {env.GENE_NAMES[i]: GENE[i] for i in range(len(GENE))},
        "summary": {
            "total": n,
            "valid_buggy_fail": n_valid,
            "table_pass": table_ok,
            "llm_pass": llm_ok,
            "table_rate": round(table_ok / n_valid, 3) if n_valid else 0,
            "llm_rate": round(llm_ok / n_valid, 3) if n_valid else 0,
            "trivial_easy": trivial,
            "invalid_questions": invalid,
            "exploit_suspects": exploits,
        },
        "difficulty_curve": curve,
        "projects": rows,
    }

    with open("reports/probe_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    # 文本报告
    lines = []
    lines.append("=" * 64)
    lines.append("编程世界模型 · 探针报告  %s" % report["generated_at"])
    lines.append("LLM层=%s  端点=%s" % (llm_on, (report["endpoint"] or {}).get("id")))
    lines.append("=" * 64)
    s = report["summary"]
    lines.append("题目总数=%d  有效题(bug真存在)=%d" % (s["total"], s["valid_buggy_fail"]))
    lines.append("查表通过=%d (%.0f%%)   LLM生成器通过=%d (%.0f%%)" % (
        s["table_pass"], s["table_rate"] * 100, s["llm_pass"], s["llm_rate"] * 100))
    if s["invalid_questions"]:
        lines.append("⚠ 无效题(测不出bug): %s" % ", ".join(s["invalid_questions"]))
    if s["trivial_easy"]:
        lines.append("· 简单题(table+llm都过, d<=1): %s" % ", ".join(s["trivial_easy"]))
    if s["exploit_suspects"]:
        lines.append("⚠ exploit嫌疑: %s" % ", ".join(s["exploit_suspects"]))
    lines.append("")
    lines.append("难度曲线(按档):")
    for d, c in curve.items():
        lines.append("  d%d  n=%d  查表 %d/%d  LLM %d/%d" % (
            d, c["n"], c["table_pass"], c["n"], c["llm_pass"], c["n"]))
    lines.append("")
    lines.append("逐题:")
    for r in rows:
        lines.append("  %-12s d%d bug_fail=%-5s tbl=%-5s llm=%-5s %s" % (
            r["id"], r["difficulty"], r["buggy_fail"], r["table_pass"], r["llm_pass"],
            ("⚠" + ";".join(r["suspicious"])) if r["suspicious"] else ""))
    with open("reports/probe_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n报告已写：reports/probe_report.json / .txt")
    print("有效题 %d · 查表 %d(%.0f%%) · LLM %d(%.0f%%)" % (
        n_valid, table_ok, s["table_rate"] * 100, llm_ok, s["llm_rate"] * 100))


if __name__ == "__main__":
    main()
