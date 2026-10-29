#!/bin/bash
# 训练循环 · 三段式：训练 → 实战验收 → 保留或回滚
#
# 🔴 为什么不能"训完就采纳"（2026-10-30 A/B 实测）：
#   同一裁判、各 40 局：
#     第 1 轮权重  验证准确率 67.8%  →  实战 38.9 手
#     第 2 轮权重  验证准确率 69.7%  →  实战 25.4 手
#   准确率涨了 1.9 个点，手数掉了 13.5 手。
#   ⇒ 验证准确率是假指标。DAgger 样本 ×9 重采样，越训越只认那 300 局。
#   ⇒ 以后验收只看手数，且必须实测，不能拿 loss/acc 代替。
#
# 🔴 评测条件必须干净：同一时刻只跑一路，否则 CPU 争用时
#   同一份权重能从 38.9 手测成 22 手，测量本身在撒谎。
#
# 🔴 同一时刻只能有一路训练：两个 train_sl 并发时，后起的抢不到 CUDA
#   会静默回落 CPU（370 秒/代）。

cd /e/selfevo || exit 1
PY=/e/FlatWorld/.venv/Scripts/python.exe
ENG=rapfi/pbrain-rapfi-windows-avx2.exe
LOG=results/train_loop.log
BEST=results/net_best.pt
TRY=results/net_try.pt
EVAL_GAMES=20          # 20 局标准误约 2.4 手
MARGIN=1.5             # 新权重至少要快这么多手才值得替换

DAGGERS="results/dagger_1.jsonl results/dagger_2.jsonl results/dagger_3.jsonl"

# 平均存活手数（越大越强）。$1 = 权重路径
strokes_of() {
  rm -f results/_ev_ladder.json
  "$PY" match_rapfi.py --engine "$ENG" --games "$EVAL_GAMES" --movetime 300 \
      --net "$1" --ladder results/_ev_ladder.json --out results/_ev.json \
      > results/_ev_stdout.log 2>&1
  grep -o "共 [0-9]*手" results/_ev_stdout.log | grep -o "[0-9]*" \
      | awk '{s+=$1; n++} END{ if (n>0) printf "%.1f", s/n; else printf "0" }'
}

echo "=== 训练循环启动（三段式 · 只看手数）$(date '+%H:%M:%S') ===" >> "$LOG"
[ -f "$BEST" ] || { cp results/net_sl.pt "$BEST" 2>/dev/null && echo "# 首次：以当前网为基准" >> "$LOG"; }

round=0
while true; do
  round=$((round+1))
  n1=$(wc -l < results/rapfi_moves.jsonl 2>/dev/null || echo 0)
  n2=$(wc -l < results/rapfi_selfplay.jsonl 2>/dev/null || echo 0)
  nd=0
  for f in $DAGGERS; do [ -f "$f" ] && nd=$((nd + $(wc -l < "$f"))); done

  # DAgger 重采样倍数：目标约 3 万条、上限 8 倍（×9 实测过拟合）
  if [ "$nd" -gt 0 ]; then
    mult=$(( 30000 / nd )); [ "$mult" -lt 1 ] && mult=1; [ "$mult" -gt 8 ] && mult=8
    DATA="results/rapfi_moves.jsonl,results/rapfi_selfplay.jsonl"
    for f in $DAGGERS; do [ -f "$f" ] && DATA="$DATA,$f*$mult"; done
  else
    mult=0; DATA="results/rapfi_moves.jsonl,results/rapfi_selfplay.jsonl"
  fi
  n=$(( n1 + n2 + nd * mult ))

  # 只训 6 代。实测 10 代就会把那 300 局背下来，手数反而掉。
  ep=6

  echo "" >> "$LOG"
  echo "### 第 $round 轮  训练 $(date '+%m-%d %H:%M:%S')  教材 ${n}（对局${n1} + 自对弈${n2} + DAgger${nd}×${mult}）  ${ep}代" >> "$LOG"
  cp "$BEST" "$TRY"
  "$PY" train_sl.py \
      --data "$DATA" --init "$TRY" \
      --epochs "$ep" --ch 48 --threads 2 --device cuda --bs 512 \
      --lr 4e-5 --vw 0.3 \
      --out "$TRY" --log results/train.log >> "$LOG" 2>&1

  if tail -6 results/train.log | grep -q "设备 CPU"; then
    echo "### ⚠ 掉到 CPU 上了，本轮作废（检查并发训练）" >> "$LOG"
    rm -f "$TRY"; sleep 30; continue
  fi

  # ---- 实战验收：只比手数 ----
  base=$(strokes_of "$BEST")
  cand=$(strokes_of "$TRY")
  better=$(awk -v c="$cand" -v b="$base" -v m="$MARGIN" 'BEGIN{print (c >= b + m) ? 1 : 0}')
  echo "### 验收  基准 ${base} 手  →  候选 ${cand} 手  ${better:+(差 ${MARGIN} 手以上，采纳)}" >> "$LOG"

  if [ "$better" = "1" ]; then
    cp "$TRY" "$BEST"          # 新的最好
    cp "$TRY" results/net_sl.pt
    echo "### ✅ 采纳：${base} → ${cand} 手，已写入 net_best.pt / net_sl.pt" >> "$LOG"
  else
    echo "### ✗ 不采纳：候选没比基准快 ${MARGIN} 手，回滚，最好的仍是 ${base} 手" >> "$LOG"
  fi
  rm -f "$TRY"
  sleep 10
done
