# -*- coding: utf-8 -*-
"""验证新硬核族是否拉出难度梯度：对比 喂报错 vs 盲改 下 LLM 生成器通过率。"""
import random
import time
import env
import tasks
import gen

NEW = ["topo", "dijkstra", "merge", "quicksort", "bfs", "kadane"]
OLD = ["calc", "lru", "bisect", "dedupe", "palindrome", "fib", "stack"]

rng = random.Random(7)

def run(fam, blind):
    bp = fam["bug_patterns"][0]
    e = env.CodeEnv(seed=abs(hash(fam["id"])) % 100003, hidden=3 if blind else 1, use_tasks=True)
    e.files = bp["mutate"](dict(fam["files"]), rng)
    e.bug_file = fam["bug_file"]
    e.bug_desc = bp["desc"]
    e.entry = fam["entry"]
    e.hidden_flag = blind
    # 基因：read_first=1（想先看报错），retry=6，edit_scope=0（单文件），backtrack=0
    bot = env.Bot([1.0, 0.0, 6, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0])
    passed, error, ms, o, f = bot.submit(e)
    return passed

print("族            喂报错   盲改")
feed = []; blind = []
fams = [next(x for x in tasks.GOLDEN if x["id"] == i) for i in NEW]
for fam in fams:
    p1 = run(fam, False)
    time.sleep(1.5)
    p2 = run(fam, True)
    time.sleep(1.5)
    if p1: feed.append(fam["id"])
    if p2: blind.append(fam["id"])
    print("%-12s %-8s %-8s" % (fam["id"], "PASS" if p1 else "FAIL", "PASS" if p2 else "FAIL"))

print()
print("喂报错通过: %d/%d" % (len(feed), len(NEW)))
print("盲改通过:   %d/%d" % (len(blind), len(NEW)))
print("梯度差(喂-盲): %d" % (len(feed) - len(blind)))
print("盲改失手族:", [i for i in NEW if i in feed and i not in blind])
