# SelfPlay Lab

**五个域，一套命题：自己跟自己打，让"世界"在冠军打满时自动变难。**

这个仓库想回答一个问题：

> **自进化 / 世界模型这条路，到底能不能走通？**

结论先说：**机制走通了，瓶颈在评估器。** 详细证据见 [`docs/FINDINGS.md`](docs/FINDINGS.md)。

前三个域用**遗传搜索**（基因 → 个体 → 对战 → 适应度 → 选择 → 变异 → 世界升级）；第四个域 gomoku 换成**梯度方法**（双头网络 + 搜索 + DAgger）；第五个域 neuro 是**纯 NumPy 神经网络自博弈**（PPO + 行为克隆预热 + 冻结 league + one-step 世界模型），用同一个 duel 环境验证"把参数化策略直接交给梯度"能不能跑通，以及评估器健康判据是否仍然成立。

---

## 一、五个域

同一套代码骨架，换五个"世界"：

| 域 | 世界是什么 | 决策表示 | 代数 | 难度 | 冠军状态 |
|---|---|---|---:|---|---|
| **code** | 13 族算法题 + 程序化 bug 注入 | 10 维基因（读码/改动范围/重试/回滚/测试/重构/文档/并行/回滚速度/脚手架） | 2355 | Lv15（5 维全开，满级） | fit 1.000 —— **打穿了**（触发饱和判据后继续升到满级） |
| **duel** | 两个火柴人徒手对打（纯 numpy 物理） | 9 维基因（武器/攻击性/格挡/踢/距离/反应/待机/捡道具/绕陷阱） | 198 | Elo 制 | fit 0.508 → 1.126，Elo 1573，流派「反击×20」 |
| **sgs** | 三国杀式卡牌单挑 | 4 维基因（攻击性/防守/回血/距离） | 2670 | Lv4（4 维全开） | fit 0.660 —— **零满分，还在爬** |
| **gomoku** | 五子棋 15×15，对手是 Rapfi（Elo 3073） | 双头网络 48 通道（policy 225 + value 1）+ VCF 搜索 | 300+ 局实测 | 固定强敌 | **0 胜 / 4.9%**，存活 25.5 → **48.2 手** |
| **neuro** | 同 duel 场景，双方换成神经网络策略；`--env articulated` 换成肢体胶囊碰撞 | 15/23 维观测 → MLP(64) → 6 动作（PPO） | 按 iteration 计 | 对手池 random / scripted / 冻结快照 | BC 预热后 **vs random 1.0、vs scripted 1.0**，PPO 微调保持；health 报 `saturated` |

> 代数为各域 `evolve.log` / `progress.tsv` 的实时值，截至 2026-10-29 22:10；gomoku 为 2026-10-30 凌晨实测。
> 前三个域当时都在后台跑，数字会继续长。

四域的**结局正好是四种典型状态**，这也是最有价值的部分：

- **code**：冠军 99.6% 满分（3603/3617 条冠军记录 fit=1.000）→ **评估器被打穿，进化退化成随机漫步**
- **sgs**：零满分（最高 0.896，956 条记录无一到顶），fit 长期在 0.57~0.73 波动 → **梯度真实存在，还在选**
- **duel**：fit 从 0.508 单调爬到 1.126，198 代后流派稳定收敛到 **反击 ×20 / 共 25 只**（其余是游走、铁壁、中庸各 1~3）→ **收敛成功，但趋同**
- **gomoku**：存活手数从 25.5 涨到 48.2（近两倍），但**胜率始终 0~4.9%**；且训练好的胜率头接进搜索后手数从 44.6 腰斩到 21.2 → **评估器没被打穿，是被"缺维度"废掉了**

gomoku 这一条给主命题补上了评估器故障的**第二种形态**：code 域是"评估器太松，梯度归零"；gomoku 是"评估器不够松，但它缺了决定胜负的那个维度"。详见 [`domains/gomoku/README.md`](domains/gomoku/README.md)。

---

## 二、机制

### 2.1 基因与个体

遗传域的每个个体是一串实数基因，直接映射成一份可执行的策略。**没有神经网络，没有梯度——只有遗传搜索。**（gomoku / neuro 两个域走梯度路线，见 2.4。）

```
code 域 10 维：read_first edit_scope retry_cap backtrack test_cover
                refactor doc_first parallel_try rollback_speed scaffold

duel 域 9 维： weapon aggr guard kick dist react idle loot avoid

sgs 域 4 维： aggr guard heal dist
```

变异 = 高斯扰动 + 边界裁剪；选择 = 精英保留 + 锦标赛。

### 2.2 世界自迭代（这是"世界模型"的那一半）

不是固定题库。世界有一组难度旋钮，**当最近 10 代冠军的 EMA 波动小于 0.01（平台期）就自动升一级**：

```
code 域 5 维： size（函数体积）depth（依赖深度）hidden（报错隐蔽度）
              scaffold（脚手架约束）resource（时空约束）

sgs 域 5 维：  hand（初始手牌）sha（杀密度）tao（桃密度）hp（起始血量）hero（武将）
```

`hidden` 升到 2 级以上时，解题者**拿不到报错信息**，只能靠读源码定位——这就是"盲改"模式。

升级历史落盘在 `data/*_levels.jsonl`，可回溯。

### 2.3 脑与笔（code 域特有）

编程域里，**本地进化系统和外部 LLM 的分工是刻意分开的**：

```
本地脑 brain_plan()   →  看源码 + 看 bug 描述 + 看报错
                          产出：intent（改什么）、target（改哪个文件）、指令
外部笔 pen_write()    →  只按指令写代码，不参与判断
                          写完由本地拿测试真跑一遍验证
```

**脑负责想，笔负责写。** 这个拆分不是架构洁癖——实测发现，把 LLM 当"全功能脑"用时，它会自作主张改别的地方，改完还说自己对了。拆开之后，`brain_plan` 的 intent 从空泛的"分析并修复"变成了具体的「修正括号配对表」「统计入度每条边加 2 而非加 1」，可核验。

笔是可替换的：默认读 `fleet/compute` 的端点池，也可以直接用环境变量指定任意 OpenAI 兼容端点（见 3.2）。

### 2.4 神经域：PPO + BC 预热 + 世界模型（neuro 域特有）

neuro 域不依赖 torch，整套网络是纯 NumPy 手写的，这样它和遗传域共享"零额外依赖"的前提：

```
selfplay/nn/mlp.py         MLP + Adam + 手写反向传播 + checkpoint
selfplay/nn/ppo.py         离散动作 PPO（GAE / clipped surrogate / entropy / value）
selfplay/nn/world_model.py one-step 世界模型：obs + action → next obs（MSE）
selfplay/core/health.py    评估器健康判据（饱和 / 地板锁 / 缺维 / league overfit / 策略坍缩）
selfplay/core/manifest.py  每次 run 的 manifest（seed / config / git commit / 平台）
selfplay/core/recorder.py  metrics.jsonl + progress.tsv + 可 resume 的 state
```

训练闭环：

1. **BC 预热**：先用一个一维阈值专家（`heuristic_action`：进入踢击距离前一直 FWD，进入后 KICK）跑 8 局，行为克隆 60 epoch。神经策略从 6 选 1 的随机初始点直接探索"靠近再出腿"几乎学不到（稀疏奖励），克隆后立刻就能对 random / scripted 双 100% 胜率。
2. **PPO 微调**：对 random / scripted / 冻结快照混合采样，GAE + clip 更新，不把克隆策略练坏。
3. **league**：每 2 个 iteration 冻结一份当前策略进对手池，防止只过拟合当前自己。
4. **world model**：从真实 transition 学 one-step 预测，MSE 从 0.16 降到 0.0076；它同时是评估器的诊断信号。
5. **world curriculum（新）**：`selfplay/core/curriculum.py`。只有两个信号同时成立才升级：
   - 评估器 `saturated`（冻结 holdout 已区分不出策略）；
   - world model 最近 3 个 iteration 的平均 MSE < `--wm-level-threshold`（世界已经可预测）。

   对手阶梯是 `L0 gene1 → L1 gene2 → L2 hard1 → L3 hard2`，升级后训练池保留全部旧档位（防止遗忘），holdout 切到新档位。`floor_lock` 连续出现时自动降级。这就是遗传域"打满就升级"规则的神经版。
6. **health**：holdout 已 100% 且 entropy 低时直接报 `saturated`——评估器已经区分不出策略，继续调参只是自欺。

实测：L0 在第 5 个 iteration 升级，`holdout_win` 从 1.0 掉到 0.5–0.85，health 回到 `ok`，评估器重新有了梯度。之后 PPO 没有把 L1 打到饱和（稳定在 0.6–0.9），而 world model MSE 继续降到 0.006——**世界"可预测"不等于策略"已掌握"**，这正是课程门控必须要求两个信号同时成立的原因。

**关键修复**：`heuristic_action` 里的距离必须按 `obs[12] * 4.0` 还原成米（`ARENA_HALF = 4.0`），不是 `/8`。旧写法让专家走到贴脸才出手，44% 的对局超时，克隆出来的是"踢不到"的坏策略。

#### 肢体碰撞（`--env articulated`）

默认 neuro 的命中仍是"水平距离 ≤ 射程"。`domains/duel/limbs.py` + `articulated.py` 把它换成肢体几何：

- 2D 矢状面运动学：肩/肘、髋/膝角度 → 手、脚坐标；角度只由状态/招式/计时决定，不额外存状态，replay 仍可从 duel 状态重建
- 命中用胶囊（前臂-拳 / 胫-脚）对躯干、头、双腿做 segment-segment 距离；头 1.35x、躯干 1.0x、腿 0.75x 伤害
- 格挡是真正的前臂护体胶囊：攻击必须够到护体才减伤，而不是"按了格挡就免伤"
- 双方活跃肢体在空档相遇 → **clash**：两招同时取消、无人掉血；若任一方已经够到躯干，则按同时命中（trade）处理
- 三种招式按真实格斗的"速度换伤害"设计：直拳 4/3/7 帧、5.5 伤害；前踢 7/4/11 帧、8.5；头踢 12/4/18 帧、12.5（头 1.35x ≈ 16.9）
- `KICK` 是一个按钮，但引擎按距离自动选招：>0.84 m 是前踢（够到躯干，射程 1.29），≤0.84 m 变成头踢（射程缩到 0.83，打头）
- 头部是独立球体（`HEAD_Y=1.66`），未被格挡的头击额外 +6 帧硬直，并单独计入 `head_hits`
- 观测 15 → 23 维（双方手/脚相对坐标）；`metrics.jsonl` 新增 `clashes`、`head_hits`、`eval_head_hits`

**刻意缩小**：位移仍是 1D。先把肢体接触做对，再谈 2D 平台物理——否则物理 bug 和 RL bug 会混在一起。

实测（`--env articulated`）：

| 招式 | 起手/命中/后摇（帧） | 总时长 | 伤害 | 有效射程 |
|---|---|---|---|---|
| 直拳 / oi-zuki | 4 / 3 / 7 | 14（233 ms） | 5.5（头 7.4） | 1.04 m（打头 ≤0.96） |
| 前踢 / mae geri | 7 / 4 / 11 | 22（367 ms） | 8.5 | 1.29 m（躯干/腿） |
| 头踢 / mawashi geri | 12 / 4 / 18 | 34（567 ms） | 12.5（头 16.9） | 0.83 m（只有贴身才够得到） |

BC（12 局 × 120 epoch，示范对手轮换 random/gene1/gene2/镜像）：`bc_loss 0.0827`，
holdout 0.75–0.92，random 0.67–0.92，每局平均 **16 次头击**；L0 一直没到饱和，课程不升级。
36 个测试里有 20 个是肢体运动学/胶囊/接触规则用例。

---

## 三、怎么跑

### 3.1 依赖

```bash
pip install numpy
```

就这一个。遗传域是纯标准库 + numpy；gomoku 的卷积网络用外部 wheel（见该域 README），neuro 域的 MLP / PPO 是纯 numpy 手写，不需要 torch。

### 3.2 code 域

```bash
cd domains/code

# 纯本地（不调 LLM，走查表兜底）—— 能跑，但没有"笔"，修复质量差
CODE_DUEL_LLM=0 python evolve.py --resume --generations 0 --seed 7 --out-dir evo

# 接外部笔（任意 OpenAI 兼容端点）
export CODE_DUEL_API_BASE=https://your-endpoint/v1
export CODE_DUEL_API_KEY=sk-xxx
export CODE_DUEL_MODEL=your-model
python evolve.py --resume --generations 0 --seed 7 --out-dir evo
```

`--generations 0` = 无限跑。中断后 `--resume` 从 `evo/state.json` 续。

**探针**（评估器体检，不进主循环）：

```bash
python probe.py          # 三层对比：buggy / 查表 / LLM
python probe_hidden.py   # 盲改 vs 喂报错，看梯度有没有真拉开
python probe_new.py      # 只测硬核算法族
```

### 3.3 duel 域

```bash
cd domains/duel
python verify.py                              # 规则自检 200 局
python collect_duel.py --episodes 800 --out duel.csv   # 采数据
python evolve.py --resume --generations 0 --out-dir evo
```

回放：`make_replay.py` 生成 `duel_replay.html`，双击即看。

### 3.4 sgs 域

```bash
cd domains/sgs
python evolve.py --resume --generations 0 --out-dir evo
```

### 3.5 neuro 域（神经自博弈）

```bash
# 默认：BC 预热 + PPO 微调 10 个 iteration
python domains/neuro/train.py --iterations 10 --episodes 4 --max-frames 600 \
    --out-dir runs/neuro

# 中断后从 state.json / state.npz 续跑（与不间断训练逐条一致）
python domains/neuro/train.py --resume --iterations 20 --out-dir runs/neuro

# 只做 BC、不跑 PPO，用来单看克隆策略的底子
python domains/neuro/train.py --iterations 1 --epochs 0 --out-dir runs/neuro_bc

# 关掉世界模型
python domains/neuro/train.py --no-world-model --out-dir runs/neuro_nwm

# 关掉自动课程（评估器饱和后不升级，用于对照）
python domains/neuro/train.py --no-curriculum --out-dir runs/neuro_flat

# 肢体胶囊碰撞版本（23 维观测；示范是"贴身直拳打头"）
python domains/neuro/train.py --env articulated --bc-episodes 12 --bc-epochs 120 \
    --iterations 6 --episodes 6 --out-dir runs/neuro_art

# 调整升级耐心 / 世界模型门控阈值
python domains/neuro/train.py --curriculum-patience 3 --wm-level-threshold 0.015 \
    --out-dir runs/neuro_tuned
```

产物：`manifest.json`（run 元信息）、`metrics.jsonl`（逐 iteration 指标）、`progress.tsv`（表格版）、`state.json` + `state.npz`（可 resume）。

```bash
python -m unittest discover -s tests -v   # 36 个测试：反向传播、PPO、BC、课程、resume 等价、health、肢体几何/接触
```

---

## 四、目录

```
domains/
  code/    evolve.py    主循环（世界升级 / 选择 / 变异）
           env.py       环境：基因定义、个体、解题、跑测试
           tasks.py     题库：13 族 golden 实现 + 24 个 bug 注入器
           gen.py       脑与笔（brain_plan / pen_write）
           probe*.py    评估器体检
  duel/    duel.py      物理与规则（纯 numpy）
           limbs.py     2D 矢状面肢体运动学 + 胶囊碰撞几何
           articulated.py 肢体接触版对局（23 维观测、clash、部位伤害）
           bot.py       基因 → 策略
           evolve.py    进化主循环（含 Elo 与名人堂）
           verify.py    规则自检
  sgs/     env.py       卡牌单挑环境
           evolve.py    进化主循环
  gomoku/  gomoku.py    棋盘与规则（位棋盘 + 7 维规则评估）
           vcf.py       连续冲四搜索（VCF）：找杀 & 破杀，纯规则不吃评估函数
           gpu_selfplay.py  PolicyNet：共享卷积体 + policy 头(225) + value 头(1)
           train_sl.py  双头训练（CE policy + vw·BCE value，按局切分防泄漏）
           selfplay_rapfi.py  造教材：Rapfi 强弱对局，每步记 Eval 当值标签
           dagger.py    DAgger：自己走，每步回头问 Rapfi「这局面你走哪」
           match_rapfi.py 对战 + Piskvork 协议 + 手数裁判 + 热加载
           train_loop.sh   三段式：训练 → 实测手数验收 → 不达标回滚
           probe_value*.py 胜率头体检（饱和度 / 视角自洽性）
           diag.py      单局诊断（逐手看命中哪道闸门）
           results/     成果：逐局记录、训练日志、两个关键权重、A/B 原始记录
           logs/        遗传搜索版（路线 A）的历史运行日志
  neuro/   env.py       把 duel / articulated 包装成 15 / 23 维观测、6 动作的训练环境
           opponents.py random / scripted / 冻结快照 对手池 + BC 专家
           train.py     BC 预热 + PPO + league + 世界模型 + health + resume
selfplay/
  nn/      mlp.py       MLP + Adam + 手写反向传播
           ppo.py       离散动作 PPO（GAE / clip / entropy）
           world_model.py  one-step 观测预测模型
  core/    health.py    评估器健康判据
           curriculum.py 世界课程（health + world model 双门控）
           manifest.py  run 元信息
           recorder.py  metrics.jsonl / progress.tsv / resume 状态
tests/     test_neuro.py 神经栈的单元与回归测试
           test_limbs.py 肢体运动学、胶囊几何与接触规则测试

data/      前三域的进化轨迹（每 N 代抽样）+ 难度晋升历史 + 当前冠军
docs/      FINDINGS.md —— 走通到哪一步、崩在哪、下一步怎么扩
```

> `data/*_progress.tsv` 是从原始轨迹按 10:1 / 5:1 抽样后的版本，原始文件在每个域自己的 `evo/` 下。
> duel 域的 46MB 逐帧对战数据 `duel.csv` 没有入库，用 `collect_duel.py` 可重新生成。
> gomoku 域的 Rapfi 引擎（96M）、torch wheel（194M）、教材数据（188M+42M+17M）都没有入库，`domains/gomoku/README.md` 写了怎么重新生成。

---

## 五、最重要的一条

**自进化的瓶颈不是搜索算法，是评估器。**

搜索再聪明，只要评估器被打穿（冠军轻松满分），选择压力就归零，进化立刻退化成随机漫步——代码在跑、日志在涨、指标好看，但**什么也没学到**。

code 域 2302 代、99.6% 满分，就是这个失败模式的标本。sgs 域零满分、fit 长期在 0.57 上下挣扎，反而是**健康**的。

判断一个自博弈系统是不是真在学，只看一个指标：**冠军的适应度有没有顶到天花板。** 顶住了，就是该升难度或者换题了。

### 5.1 补一句：评估器会以两种方式坏掉

跑完 gomoku 域之后，这句话要扩成两条。评估器的故障不是只有"太松"一种：

| 故障形态 | 表现 | 标本 | 怎么发现 |
|---|---|---|---|
| **饱和型** | 冠军轻松满分，梯度归零 | code 域（99.6% 满分） | 看冠军 fit 有没有顶天花板 |
| **缺维型** | 冠军没法满分、指标一直在爬，但它**缺了决定胜负的那个维度**，梯度指向错误方向 | gomoku 域（胜率头把"活四"和"活三"都判成 0.2） | **只做 A/B：换成它，实战指标是升还是降** |

缺维型更危险，因为所有常规指标都是"健康的"：
- 值损从 0.6849 降到 0.5622（真的在学）
- 均势局面给出 0.755/0.429（真的有梯度、不饱和）
- 方向准确率 88.5%（真的准）

**但它接进搜索后，实战存活从 44.6 手掉到 21.2 手。** 差 17 个标准误。

原因是它学的是"形势好坏"，不是"还有几步死"——活四（下轮必死）和活三（还能救）都给 0.2，搜索分不出该先救哪个。而那 7 个手写数字粗糙，但它数的是威胁等级，排序是对的。

所以补一条判据：

> **评估器不只要"能排序"，还要"排的是对的那个序"。**
> 验证方法只有一个：把它换进决策链，看实战指标。离线指标（损失、准确率、分布形状）全都可能骗人。

这条对前三域同样成立——它们的适应度函数都是人写的，如果写漏了一个决定胜负的维度，同样会出现"指标健康、实战不行"。

---

## License

MIT
