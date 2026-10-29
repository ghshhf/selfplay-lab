"""验证 hidden 维度是否真拉开曲线：同一 task，盲改 vs 喂报错，LLM 通过率差异。"""
import env, tasks

rng_base = 1000
feed_pass = []
blind_pass = []
rows = []
for i in range(7):
    seed = rng_base + i * 101
    # 喂报错（hidden=1 -> hidden_flag=False）
    ef = env.CodeEnv(seed=seed, hidden=1, use_tasks=True)
    bot = env.Bot([1.0, 0.0, 6, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0])  # read_first=1
    pf, ef_err, _, _, _ = bot.submit(ef)
    # 盲改（hidden=3 -> hidden_flag=True，不喂报错）
    eb = env.CodeEnv(seed=seed, hidden=3, use_tasks=True)
    bot2 = env.Bot([1.0, 0.0, 6, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 0.0])
    pb, eb_err, _, _, _ = bot2.submit(eb)
    if pf:
        feed_pass.append(ef.project["id"])
    if pb:
        blind_pass.append(eb.project["id"])
    rows.append((ef.project["id"], pf, pb,
                 (ef_err or "")[:50], (eb_err or "")[:50]))

print("task             feed   blind")
for tid, pf, pb, fe, be in rows:
    print("%-14s  %-5s  %-5s" % (tid, "PASS" if pf else "FAIL", "PASS" if pb else "FAIL"))
print()
print("喂报错通过: %d/%d" % (len(feed_pass), len(rows)))
print("盲改通过:   %d/%d" % (len(blind_pass), len(rows)))
print("梯度差(喂-盲): %d" % (len(feed_pass) - len(blind_pass)))
