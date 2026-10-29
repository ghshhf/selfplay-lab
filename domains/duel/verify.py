"""verify.py —— 对打环境的自检：规则对不对、数据能不能对账、动作有没有偏科"""
import numpy as np
import duel as D
from bot import Bot

def run(n=200, seed=1, verbose=False):
    rng = np.random.default_rng(seed)
    stats = {"acts": np.zeros((2, 6), dtype=np.int64), "hits": 0, "blocked": 0,
             "frames": [], "ko": 0, "draw": 0, "win": [0, 0], "dmg": [],
             "xmin": 1e9, "xmax": -1e9, "balance_err": []}
    for ep in range(n):
        env = D.Duel(seed=int(rng.integers(1 << 30)))
        env.reset()
        b0, b1 = Bot.random(rng), Bot.random(rng)
        while True:
            a0, a1 = b0(env.p0, env.p1), b1(env.p1, env.p0)
            stats["acts"][0, a0] += 1
            stats["acts"][1, a1] += 1
            _, _, _, _, done, info = env.step(a0, a1)
            stats["xmin"] = min(stats["xmin"], env.p0.x, env.p1.x)
            stats["xmax"] = max(stats["xmax"], env.p0.x, env.p1.x)
            if done:
                break
        stats["frames"].append(info["frames"])
        stats["hits"] += info["hits"]
        stats["blocked"] += sum(1 for e in env.events if e[3])
        stats["dmg"].append(sum(info["damage"]))
        w = info["winner"]
        if w == -1:
            stats["draw"] += 1
        else:
            stats["win"][w] += 1
        if info["frames"] < env.max_frames:
            stats["ko"] += 1
        # 对账：初始血量 - 剩余血量 必须等于记录的总伤害
        expect = 2 * D.MAX_HP - (info["hp"][0] + info["hp"][1])
        stats["balance_err"].append(abs(expect - sum(info["damage"])))
        if verbose and ep < 3:
            print("  ep%d 帧=%d hp=%.0f/%.0f winner=%s 伤害=%.1f 对账误差=%.2e" %
                  (ep, info["frames"], info["hp"][0], info["hp"][1], w,
                   sum(info["damage"]), stats["balance_err"][-1]))
    return stats


if __name__ == "__main__":
    s = run(200, seed=1, verbose=True)
    f = np.array(s["frames"])
    print()
    print("① 伤害对账      最大误差 %.2e  （必须为 0）" % max(s["balance_err"]))
    print("② 局长度         中位 %.1fs  min %.1fs  max %.1fs" %
          (np.median(f) / 60, f.min() / 60, f.max() / 60))
    print("③ 结局           KO %d/%d (%.0f%%)  平局 %d  胜方分布 %s" %
          (s["ko"], len(f), 100 * s["ko"] / len(f), s["draw"], s["win"]))
    print("④ 总命中 %d 次，其中被格挡 %d 次 (%.0f%%)" %
          (s["hits"], s["blocked"], 100 * s["blocked"] / max(1, s["hits"])))
    print("⑤ 位置范围      [%.2f, %.2f]  场地 ±%.1f" % (s["xmin"], s["xmax"], D.ARENA_HALF))
    tot = s["acts"].sum()
    print("⑥ 动作分布：")
    for i, name in enumerate(D.ACTION_NAMES):
        pct = 100 * s["acts"][:, i].sum() / tot
        print("   %-6s %7d  %5.1f%%  %s" % (name, s["acts"][:, i].sum(), pct, "█" * int(pct / 2)))
    avg_fight = f.mean() / 60
    print("⑦ 总帧数 %d  平均每局 %.1fs  单核速度待测" % (tot, avg_fight))
