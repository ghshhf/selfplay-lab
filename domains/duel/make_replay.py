"""
make_replay.py —— 跑若干局对打，挑有代表性的几局，生成可双击回放的 HTML

挑选标准（每类各取一场，避免全是"秒杀"或全是"拉锯"）：
  最短 KO / 最长拉锯 / 格挡最多 / 命中最多 / 双方各赢一场
回放里能看到：两个火柴人的动作姿态、头顶血条、命中闪光、底部攻防节奏带。
"""

import json
import numpy as np

import duel as D
from bot import Bot

FAMILY = ["FWD", "BACK", "PUNCH", "KICK", "BLOCK", "IDLE"]
PHASE_CODE = {"": 0, "windup": 1, "active": 2, "recover": 3}
# 招式码：1=拳 2=踢（不能用动作编号，PUNCH=2/KICK=3 会把踢写成非法值）
MOVE_CODE = {D.PUNCH: 1, D.KICK: 2}


def simulate(env, b0, b1):
    """跑完一局，返回逐帧渲染数据 + 统计"""
    frames = []
    n_hit = 0
    while True:
        a0, a1 = b0(env.p0, env.p1), b1(env.p1, env.p0)
        _, _, _, _, done, info = env.step(a0, a1)
        p0, p1 = env.p0, env.p1
        hit = 0
        if len(env.events) > n_hit:
            e = env.events[-1]                   # (frame, move, dmg, blocked, attacker)
            n_hit += 1
            victim = 1 - e[4]                    # 被打的是攻击者的对面
            hit = (victim + 1) * (-1 if e[3] else 1)
        frames.append([
            round(p0.x, 3), round(p1.x, 3),
            round(p0.hp, 1), round(p1.hp, 1),
            p0.state, p1.state,
            PHASE_CODE.get(p0.phase, 0), PHASE_CODE.get(p1.phase, 0),
            MOVE_CODE.get(p0.move, 0),
            MOVE_CODE.get(p1.move, 0),
            a0, a1, hit,
        ])
        if done:
            break
    blocked = sum(1 for e in env.events if e[3])
    return {
        "frames": frames,
        "winner": info["winner"],
        "n": len(frames),
        "hits": info["hits"],
        "blocked": blocked,
        "hp": [round(info["hp"][0], 1), round(info["hp"][1], 1)],
        "damage": [round(d, 1) for d in info["damage"]],
        "ko": info["frames"] < env.max_frames,
        "bot0": {k: (round(v, 2) if isinstance(v, float) else v)
                 for k, v in b0.__dict__.items() if k != "rng"},
        "bot1": {k: (round(v, 2) if isinstance(v, float) else v)
                 for k, v in b1.__dict__.items() if k != "rng"},
    }


def pick(pool, key, reverse=False, used=None):
    used = used or set()
    cand = [x for x in pool if x["_id"] not in used]
    if not cand:
        return None
    return sorted(cand, key=key, reverse=reverse)[0]


def main(n_pool=140, seed=11):
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n_pool):
        env = D.Duel(seed=int(rng.integers(1 << 30)))
        env.reset()
        b0, b1 = Bot.random(rng), Bot.random(rng)
        r = simulate(env, b0, b1)
        r["_id"] = len(pool)
        pool.append(r)

    used = set()
    picks = []

    def take(cand, why):
        if cand is None:
            return
        if cand["_id"] in used:
            return
        used.add(cand["_id"])
        c = dict(cand)
        c["why"] = why
        picks.append(c)

    take(pick(pool, lambda r: r["n"], reverse=False, used=used), "最快 KO")
    take(pick(pool, lambda r: r["n"], reverse=True, used=used), "最长的一局")
    take(pick(pool, lambda r: r["blocked"], reverse=True, used=used), "格挡最多")
    take(pick(pool, lambda r: r["hits"], reverse=True, used=used), "命中最多")
    take(pick([r for r in pool if r["winner"] == 0], lambda r: r["hp"][1], used=used), "蓝方大胜")
    take(pick([r for r in pool if r["winner"] == 1], lambda r: r["hp"][0], used=used), "红方大胜")

    data = {
        "fps": D.FPS,
        "arena": D.ARENA_HALF,
        "maxHp": D.MAX_HP,
        "moves": {str(k): v for k, v in D.MOVES.items()},
        "replays": picks,
    }
    html = TPL.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out = "duel_replay.html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    print("生成 %s（%d 局）" % (out, len(picks)))
    for p in picks:
        print("  [%s] 帧=%4d %.1fs  胜方=%s  hp=%.0f/%.0f  命中=%d 格挡=%d" %
              (p["why"], p["n"], p["n"] / D.FPS, p["winner"],
               p["hp"][0], p["hp"][1], p["hits"], p["blocked"]))


TPL = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MiniDuel 回放 · 两个小人对打</title>
<style>
  :root{
    --bg:#12161d; --panel:#1a2029; --line:#2b3442; --txt:#e6edf3; --dim:#8b96a5;
    --blue:#4ea8ff; --red:#ff6b5a; --gold:#ffcc55; --green:#4ade80;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--txt);
    font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding:20px 18px 40px}
  h1{font-size:19px;margin:0 0 4px;font-weight:600}
  .sub{font-size:12.5px;color:var(--dim);margin-bottom:14px}
  .tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
  .tab{font-size:12px;padding:6px 11px;border-radius:6px;cursor:pointer;
    background:var(--panel);border:1px solid var(--line);color:var(--dim);
    transition:.15s}
  .tab:hover{color:var(--txt);border-color:#3d4a5c}
  .tab.on{background:#243044;border-color:#3f5f8a;color:var(--txt)}
  canvas{width:100%;display:block;border-radius:10px;border:1px solid var(--line);
    background:#151b24}
  .bar{display:flex;align-items:center;gap:10px;margin-top:10px;flex-wrap:wrap}
  button{background:var(--panel);color:var(--txt);border:1px solid var(--line);
    border-radius:6px;padding:6px 13px;font-size:12.5px;cursor:pointer}
  button:hover{border-color:#3f5f8a}
  .info{font-size:12px;color:var(--dim);margin-left:auto;font-variant-numeric:tabular-nums}
  .why{font-size:12px;color:var(--gold);margin-bottom:8px}
  .legend{display:flex;gap:14px;font-size:11.5px;color:var(--dim);margin-top:8px;flex-wrap:wrap}
  .dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
</style>
</head>
<body>
<div class="wrap">
  <h1>MiniDuel · 两个小人对打</h1>
  <div class="sub">规则：走位 / 拳（快、近） / 踢（慢、远、重） / 格挡（挡下 72% 伤害） / 受击硬直。每局最多 30 秒。</div>
  <div class="tabs" id="tabs"></div>
  <div class="why" id="why"></div>
  <canvas id="cv" width="1000" height="440"></canvas>
  <div class="bar">
    <button id="play">暂停</button>
    <button id="again">重播</button>
    <button id="slow">0.5×</button>
    <span class="info" id="info"></span>
  </div>
  <div class="legend">
    <span><i class="dot" style="background:var(--blue)"></i>蓝方（左）</span>
    <span><i class="dot" style="background:var(--red)"></i>红方（右）</span>
    <span><i class="dot" style="background:var(--gold)"></i>命中</span>
    <span><i class="dot" style="background:#7d8ea3"></i>格挡</span>
    <span>底部色带 = 每帧动作：<b>蓝走·蓝打·灰挡</b> / <b>红走·红打·灰挡</b></span>
  </div>
</div>
<script>
const DATA = __DATA__;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const W = cv.width, H = cv.height;
const GROUND = H - 96;
const AR = DATA.arena;
const SX = (W * 0.40) / AR;          // 世界 x -> 像素

let cur = 0, frame = 0, playing = true, speed = 1, acc = 0, last = 0;

/* ---------- 顶部标签 ---------- */
const tabs = document.getElementById('tabs');
DATA.replays.forEach((r, i) => {
  const d = document.createElement('div');
  d.className = 'tab' + (i === 0 ? ' on' : '');
  d.textContent = (i + 1) + '. ' + r.why;
  d.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('on'));
    d.classList.add('on');
    cur = i; frame = 0; playing = true;
    document.getElementById('play').textContent = '暂停';
    updateWhy();
  };
  tabs.appendChild(d);
});

function updateWhy() {
  const r = DATA.replays[cur];
  const w = r.winner < 0 ? '平' : (r.winner === 0 ? '蓝方胜' : '红方胜');
  document.getElementById('why').textContent =
    r.why + ' · ' + (r.n / DATA.fps).toFixed(1) + ' 秒 · ' + w +
    ' · 命中 ' + r.hits + ' 次（被格挡 ' + r.blocked + '）';
}
updateWhy();

/* ---------- 画火柴人 ---------- */
// 一帧的渲染态 -> 关节坐标
function joints(o) {
  // o: {x, state, phase, move, facing, t(帧内相位)}
  const px = W / 2 + o.x * SX;
  const hipY = GROUND - 64;
  const hip = [px, hipY];

  let lean = 0, crouch = 0;
  let armExt = 0, armUp = 0;      // 前手：伸展 / 抬起（格挡）
  let legUp = 0, legExt = 0;
  let headOff = 0;

  const ST = ['IDLE', 'WALK', 'ATTACK', 'BLOCK', 'HURT', 'KO'];
  const st = ST[o.state];

  if (st === 'WALK') {
    lean = 0.10 * o.facing;
    armExt = 0.15;
  } else if (st === 'ATTACK') {
    if (o.move === 1) {                      // 1=拳
      if (o.phase === 1) { armExt = -0.42; lean = -0.12 * o.facing; }
      else if (o.phase === 2) { armExt = 1.0; lean = 0.28 * o.facing; }
      else { armExt = 0.40; lean = 0.10 * o.facing; }
    } else if (o.move === 2) {                // 2=踢
      if (o.phase === 1) { legUp = 0.55; legExt = -0.35; lean = -0.18 * o.facing; }
      else if (o.phase === 2) { legUp = 1.0; legExt = 1.0; lean = -0.22 * o.facing; armExt = -0.3; }
      else { legUp = 0.40; legExt = 0.25; lean = -0.08 * o.facing; }
    }
  } else if (st === 'BLOCK') {
    armUp = 1.0; lean = -0.14 * o.facing; crouch = 6;
  } else if (st === 'HURT') {
    lean = -0.42 * o.facing; headOff = -5; armExt = -0.25;
  } else if (st === 'KO') {
    lean = -1.35 * o.facing; crouch = 40;
  }

  const neck = [hip[0] + lean * 52, hipY - 56 + crouch * 0.3];
  const head = [neck[0] + lean * 12 + headOff * 0.4, neck[1] - 22 + crouch * 0.2];
  const sh = neck;

  // 前手
  const dirF = o.facing;
  let elbowF, handF;
  if (armUp > 0.5) {                        // 格挡：前臂竖在身前
    elbowF = [sh[0] + dirF * 12, sh[1] + 20];
    handF = [sh[0] + dirF * 6, sh[1] - 12];
  } else {
    const ex = armExt;
    elbowF = [sh[0] + dirF * (14 + ex * 22), sh[1] + 20 - ex * 6];
    handF = [sh[0] + dirF * (14 + ex * 52), sh[1] + 22 - ex * 20];
  }
  // 后手
  const elbowB = [sh[0] - dirF * 10, sh[1] + 22];
  const handB = [sh[0] - dirF * 20 - dirF * armExt * 8, sh[1] + 26];

  // 前腿
  let kneeF, footF;
  if (legUp > 0.5) {                        // 踢
    const e = legExt;
    kneeF = [hip[0] + dirF * (14 + e * 26), hipY + 20 - e * 22];
    footF = [hip[0] + dirF * (10 + e * 74), hipY + 40 - e * 56];
  } else {
    kneeF = [hip[0] + dirF * 12, hipY + 30];
    footF = [hip[0] + dirF * 20, GROUND];
  }
  // 后腿
  const kneeB = [hip[0] - dirF * 12, hipY + 30 + crouch * 0.3];
  const footB = [hip[0] - dirF * 22, GROUND - crouch * 0.5];

  return {hip, neck, head, elbowF, handF, elbowB, handB, kneeF, footF, kneeB, footB};
}

function line(a, b, w, c) {
  ctx.strokeStyle = c; ctx.lineWidth = w; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
}

function drawFighter(o, color, glow) {
  const j = joints(o);
  if (glow > 0) {
    ctx.shadowColor = color; ctx.shadowBlur = 18 * glow;
  }
  // 躯干
  line(j.hip, j.neck, 9, color);
  // 手臂
  line(j.neck, j.elbowB, 6, shade(color, -0.25));
  line(j.elbowB, j.handB, 5, shade(color, -0.25));
  line(j.neck, j.elbowF, 7, color);
  line(j.elbowF, j.handF, 6, color);
  // 腿
  line(j.hip, j.kneeB, 7, shade(color, -0.25));
  line(j.kneeB, j.footB, 6, shade(color, -0.25));
  line(j.hip, j.kneeF, 8, color);
  line(j.kneeF, j.footF, 7, color);
  // 头
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(j.head[0], j.head[1], 11, 0, Math.PI * 2); ctx.fill();
  // 朝向小眼睛
  ctx.fillStyle = '#0d1219';
  ctx.beginPath();
  ctx.arc(j.head[0] + o.facing * 4.5, j.head[1] - 1.5, 2.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
}

function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  r = Math.max(0, Math.min(255, r + amt * 255));
  g = Math.max(0, Math.min(255, g + amt * 255));
  b = Math.max(0, Math.min(255, b + amt * 255));
  return 'rgb(' + (r | 0) + ',' + (g | 0) + ',' + (b | 0) + ')';
}

/* ---------- 场景 ---------- */
function drawArena() {
  const g = ctx.createLinearGradient(0, 0, 0, GROUND);
  g.addColorStop(0, '#101720'); g.addColorStop(1, '#1a232f');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, GROUND);

  // 地面
  ctx.fillStyle = '#0d131a';
  ctx.fillRect(0, GROUND, W, H - GROUND);
  ctx.strokeStyle = '#31404f'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, GROUND); ctx.lineTo(W, GROUND); ctx.stroke();

  // 中线和距离刻度
  ctx.strokeStyle = 'rgba(120,140,165,.16)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(W / 2, GROUND - 6); ctx.lineTo(W / 2, GROUND + 10); ctx.stroke();
  for (let d = 1; d <= AR; d++) {
    [1, -1].forEach(s => {
      const x = W / 2 + s * d * SX;
      ctx.globalAlpha = d === AR ? .5 : .28;
      ctx.beginPath(); ctx.moveTo(x, GROUND); ctx.lineTo(x, GROUND + (d % 2 ? 8 : 14)); ctx.stroke();
    });
  }
  ctx.globalAlpha = 1;

  // 1 米参照
  ctx.fillStyle = 'rgba(140,160,185,.45)';
  ctx.font = '11px "Segoe UI",system-ui';
  ctx.textAlign = 'center';
  for (let d = 1; d <= AR; d++) {
    ctx.fillText(d + 'm', W / 2 + d * SX, GROUND + 27);
    ctx.fillText(d + 'm', W / 2 - d * SX, GROUND + 27);
  }
}

function hpBar(x, y, w, hp, color, name, flip) {
  const h = 9;
  ctx.fillStyle = 'rgba(0,0,0,.55)';
  ctx.fillRect(x, y, w, h);
  const frac = Math.max(0, hp / DATA.maxHp);
  const fw = w * frac;
  ctx.fillStyle = color;
  if (flip) ctx.fillRect(x + w - fw, y, fw, h);
  else ctx.fillRect(x, y, fw, h);
  ctx.strokeStyle = 'rgba(255,255,255,.28)'; ctx.lineWidth = 1;
  ctx.strokeRect(x + .5, y + .5, w - 1, h - 1);

  ctx.font = '600 13px "Segoe UI",system-ui';
  ctx.fillStyle = '#dfe8f2';
  ctx.textAlign = flip ? 'right' : 'left';
  ctx.fillText(name, flip ? x + w : x, y - 8);
  ctx.font = '12px "Segoe UI",system-ui';
  ctx.fillStyle = color;
  ctx.fillText(Math.round(hp), flip ? x + w - 56 : x + 56, y - 8);
}

/* ---------- 主循环 ---------- */
function draw() {
  const R = DATA.replays[cur];
  const F = R.frames;
  const f = F[Math.min(frame, F.length - 1)];

  drawArena();

  const p0 = {x: f[0], state: f[4], phase: f[6], move: f[8], facing: f[1] >= f[0] ? 1 : -1};
  const p1 = {x: f[1], state: f[5], phase: f[7], move: f[9], facing: f[0] >= f[1] ? -1 : 1};

  // 影子
  [p0, p1].forEach((p, i) => {
    const x = W / 2 + p.x * SX;
    ctx.fillStyle = 'rgba(0,0,0,.35)';
    ctx.beginPath(); ctx.ellipse(x, GROUND + 3, 26, 5, 0, 0, Math.PI * 2); ctx.fill();
  });

  // 命中闪光：|hit|-1 = 被打的是谁，hit<0 = 格挡成功
  if (f[12] !== 0) {
    const victim = Math.abs(f[12]) - 1;
    const blocked = f[12] < 0;
    const vx = W / 2 + (victim === 0 ? f[0] : f[1]) * SX;
    const col = blocked ? '#8b9aad' : '#ffcc55';
    ctx.save();
    ctx.globalAlpha = blocked ? .32 : .55;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(vx, GROUND - 100, blocked ? 26 : 34, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    ctx.font = '700 15px "Segoe UI",system-ui';
    ctx.fillStyle = blocked ? '#a9b7c8' : '#ffd977'; ctx.textAlign = 'center';
    ctx.fillText(blocked ? 'BLOCK' : 'HIT', vx, GROUND - 140);
  }

  drawFighter(p0, '#4ea8ff', f[4] === 2 ? 1 : 0);
  drawFighter(p1, '#ff6b5a', f[5] === 2 ? 1 : 0);

  // 血条
  hpBar(60, 44, 330, f[2], '#4ea8ff', 'BLUE', false);
  hpBar(W - 390, 44, 330, f[3], '#ff6b5a', 'RED', true);

  // 计时
  ctx.font = '600 15px "Segoe UI",system-ui';
  ctx.fillStyle = '#c8d4e2'; ctx.textAlign = 'center';
  ctx.fillText((Math.min(frame, F.length - 1) / DATA.fps).toFixed(1) + 's / '
               + (F.length / DATA.fps).toFixed(1) + 's', W / 2, 56);

  // 结束横幅
  if (frame >= F.length - 1) {
    const w = R.winner;
    const txt = w < 0 ? 'DRAW' : (w === 0 ? 'BLUE WINS' : 'RED WINS');
    const col = w < 0 ? '#c8d4e2' : (w === 0 ? '#4ea8ff' : '#ff6b5a');
    ctx.save();
    ctx.globalAlpha = .82;
    ctx.fillStyle = '#0d131a';
    ctx.fillRect(0, H / 2 - 46, W, 84);
    ctx.restore();
    ctx.font = '700 34px "Segoe UI",system-ui';
    ctx.fillStyle = col; ctx.textAlign = 'center';
    ctx.fillText(txt, W / 2, H / 2 + 8);
  }

  drawRhythm(F, frame);

  document.getElementById('info').textContent =
    '帧 ' + Math.min(frame + 1, F.length) + '/' + F.length +
    '   HP ' + Math.round(f[2]) + ' : ' + Math.round(f[3]);
}

/* ---------- 底部攻防节奏带 ---------- */
function drawRhythm(F, at) {
  const y0 = H - 46, h = 30;
  const w = W - 120;
  const x0 = 60;
  const n = F.length;
  const bw = Math.max(1, w / n);

  ctx.fillStyle = 'rgba(0,0,0,.35)';
  ctx.fillRect(x0, y0, w, h);

  const colFor = (st, who) => {
    // 0走 1走 2攻 3挡 4挨 5倒
    if (st === 2) return who === 0 ? '#4ea8ff' : '#ff6b5a';
    if (st === 3) return '#6b7a8d';
    if (st === 4) return '#3a2b2b';
    if (st === 5) return '#221a1a';
    return who === 0 ? '#243449' : '#3d2830';
  };

  for (let i = 0; i < n; i += Math.max(1, Math.floor(1 / bw))) {
    const x = x0 + i * bw;
    ctx.fillStyle = colFor(F[i][4], 0);
    ctx.fillRect(x, y0, Math.max(bw, .7), h / 2);
    ctx.fillStyle = colFor(F[i][5], 1);
    ctx.fillRect(x, y0 + h / 2, Math.max(bw, .7), h / 2);
  }

  // 播放头
  const px = x0 + Math.min(at, n - 1) * bw;
  ctx.strokeStyle = '#ffd977'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(px, y0 - 3); ctx.lineTo(px, y0 + h + 3); ctx.stroke();

  ctx.font = '10.5px "Segoe UI",system-ui';
  ctx.fillStyle = 'rgba(150,170,195,.75)';
  ctx.textAlign = 'left';
  ctx.fillText('攻防节奏', x0, y0 - 8);
}

/* ---------- 播放控制 ---------- */
function tick(ts) {
  const dt = last ? Math.min(50, ts - last) : 16;
  last = ts;
  if (playing) {
    acc += dt * speed;
    while (acc >= 1000 / DATA.fps) {
      acc -= 1000 / DATA.fps;
      frame++;
      if (frame >= DATA.replays[cur].frames.length - 1) {
        frame = DATA.replays[cur].frames.length - 1;
        playing = false;
        document.getElementById('play').textContent = '播放';
        break;
      }
    }
  }
  draw();
  requestAnimationFrame(tick);
}

document.getElementById('play').onclick = e => {
  if (frame >= DATA.replays[cur].frames.length - 1) { frame = 0; }
  playing = !playing;
  e.target.textContent = playing ? '暂停' : '播放';
};
document.getElementById('again').onclick = () => {
  frame = 0; playing = true;
  document.getElementById('play').textContent = '暂停';
};
document.getElementById('slow').onclick = e => {
  speed = speed === 1 ? 0.5 : 1;
  e.target.textContent = speed + '×';
};

requestAnimationFrame(tick);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
