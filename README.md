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
| **neuro** | 同 duel 场景，双方换成神经网络策略；`--env articulated` 换成肢体胶囊碰撞，`--env freeform` 再加竖直轴、跳跃/下蹲、替身术、防御槽、情绪层与 `--physics` 刚体物理 | 15/23/35 维观测（`--emotion` 再 +8，`--mass-obs` 再 +2）→ MLP(64) → 6/9 动作（PPO） | 按 iteration 计 | 对手池 random / scripted / 冻结快照 | BC 预热后 **vs random 1.0、vs scripted 1.0**，PPO 微调保持；health 报 `saturated` |

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
selfplay/nn/latent.py      隐空间世界模型：观测 → z → 预测 → 重建 + 奖励头
selfplay/nn/rnd.py         RND 好奇心：固定随机目标网络，novelty = 预测误差
selfplay/nn/pbt.py         权重空间进化：克隆 / 按张量尺度加噪 / 选择 / 变异
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

#### 自由二维（`--env freeform`）

`domains/duel/freeform.py` 在肢体版上加了竖直轴，动作从 6 个扩到 9 个：

- 新增 **JUMP**（动作 6）：重力 9.8 m/s²，起跳 2.55 m/s，最高约 0.33 m、滞空约 31 帧
- 新增 **DUCK**（动作 7）：头降到 1.19 m，直拳（1.64 m）和头踢（1.70 m）会从头顶掠过；前踢在髋高仍能命中
- **下蹲中踢** → 低扫（脚心 ~0.46 m）：低于所有护体胶囊，**格挡挡不住**，但只能打到腿（0.75x）
- **空中踢** → 飞踢：脚心升到下巴高度，9 帧起手、9 伤害（头 12.2）
- 新增 **SUB**（动作 8，替身术）：4 帧无敌 → 瞬移到对手身后 0.9 m，消耗 40 查克拉、冷却 72 帧
- 观测 23 → 35 维（离地高度/竖直速度、两个下蹲标记、双方查克拉/替身冷却、双方防御槽）

查克拉是一条独立资源：上限 100、每帧回 0.6（约 2.8 秒回满），命中 +12、格挡成功 +4。
所以替身不是免费翻滚，而是和进攻抢资源；被打进硬直时按 SUB 也能脱身，但落地有 6 帧硬直可被反打。

**防御槽（guard meter）**：格挡仍只吃 28% 伤害，但每次挡刀按 2.4× 伤害扣防御槽；
槽空 → 28 帧破防硬直、槽重置到 35，随后是完整的反击窗口。不复读"一直按格挡"。
**Counter-hit**：在对手起手/收招阶段打中，伤害 ×1.35、硬直 +8 帧——惩罚乱挥，
奖励读招（经典 footsies 规则）。

**刚体物理（`domains/duel/physics.py`，`--physics`）**：把"x += vx/60"的 kinematic
假象换成一套最小刚体规则：

| 规则 | 实现 | 可观测效果 |
|---|---|---|
| 质量 | `Fighter.mass`（kg），`--mass0/--mass1`；走路速度乘以 `sqrt(70/m)` | 同一拳打 50 kg 与 120 kg 的位移差约 2.4 倍；100 kg 走路慢 16% |
| 库仑摩擦 | 地面减速度 `μ·g`（μ=0.30 → 2.94 m/s²），低速直接停 | 一次前踢滑行约 0.29 m（旧模型 0.22 m） |
| 重力 | 9.8 m/s²，与跳跃共用；落地非弹性 | 跳跃最高点与滞空帧数不变 |
| 墙面反弹 | 只有硬直中的击退会弹（恢复系数 0.35）；走着撞墙则停 | 20 轮后每局 1.17 次墙撞 |
| 墙角重击 | 击退撞墙按入射速度结算 1.5–7 点伤害，额外 8 帧硬直，攻方 +1.0 形状奖励 | corner slam 每局 1.08 次（第二十一节） |
| 冲量交换 | 命中按伤害给防守方冲量，进攻方吃 22% 反冲 | 重击会把两个人同时推开 |
| Hitstop | 按伤害/爆头给 1–6 帧全体冻结（计时器与情绪都不走） | 打击感来自"停"，不是数字跳 |
| 身位分离 | 按逆质量加权推开并消解接近速度 | 撞进更重的人，被推开的是你 |
| 体重观测 | `--mass-obs`：观测尾部追加 `[自己体重, 对手体重]/140`（+2 维，默认关） | 50 kg vs 100 kg 对局中，轻量席位从 .250 提到 .528（23.3） |
| 随机体重（实验性） | `--rand-mass 60 120`：训练每局重采样双方体重（隐含 `--mass-obs`），评估仍固定 `--mass0/--mass1` | 端点 50 kg 从 .528 掉到 .444（负结果，23.6） |

质量/摩擦/反弹/冲量都是纯函数，单测不依赖环境；`--physics` 默认关闭，
旧的 2D 代码路径逐字不变。实测见 FINDINGS 第二十、二十一节。

**情绪层（`domains/duel/affect.py`，`--emotion`）**：每个角色有 4 个 [0,1] 的漏积分量——
冷静 (`composure`)、恐惧 (`fear`)、愤怒 (`anger`)、气势 (`momentum`)。它们不是奖励，
而是作为观测追加 8 维（双方各 4 维），让只看当前帧的策略拥有"这回合的情绪记忆"：

| 事件 | 效果 |
|---|---|
| 被打中 | 恐惧/愤怒↑、冷静↓、气势清零（越疼幅度越大，爆头额外扣冷静） |
| 打中对手 | 气势↑、冷静↑；格挡成功只小幅加怒 |
| 挥空 | 冷静↓、愤怒↑（越打越急） |
| 破防 | 冷静大降、恐惧/愤怒↑，气势清零 |
| 每帧 | 恐惧/愤怒/气势按 0.995/0.992/0.985 衰减；血量 <35% 持续加恐惧 |
| 情绪传染 | 对手越愤怒你越恐惧；对手气势越高你气势越低 |

低冷静还会让防御槽掉得更快（最高 1.6×）——"手抖"是物理上真的存在，而不是文案。

`metrics.jsonl` 新增 `jumps` / `low_hits` / `flying_hits` / `subs` / `sub_avoided` /
`guard_breaks` / `counters` / `wall_bounces` / `wall_slams` / `sidesteps` /
`sweeps` / `evades` / `affect`（回合末四项均值）/ `act_hist`（策略实际按键分布）/
`eval_act_hist0` / `eval_act_hist1`（eval 时 50 kg 与 100 kg 各自的动作分布，用于检查是否分档），
用来区分"新动作可用"和"策略真的在用"。
实测见 FINDINGS 第十一、十四、十九、二十节。

查克拉是一条独立资源：上限 100、每帧回 0.6（约 2.8 秒回满），命中 +12、格挡成功 +4。
所以替身不是免费翻滚，而是和进攻抢资源；被打进硬直时按 SUB 也能脱身，但落地有 6 帧硬直可被反打。

四条规则不是写死的克制表，而是胶囊几何自然长出来的：

| | 克 | 被克 |
|---|---|---|
| 下蹲 | 直拳、头踢 | 前踢（打躯干） |
| 低扫 | 格挡 | 跳（腾空后腿抬高） |
| 飞踢 | — | 起手慢，落地即空窗 |
| 替身 | 慢招（头踢 12 帧起手） | 40 查克拉 + 72 帧冷却，落地 6 帧可被反打 |

#### 走向 3D 的第一块地基（`domains/duel/space3d.py`）

2D 格斗和 3D 格斗的差别不是"多画一个轴"，而是**攻击线可以被侧移让开**：
Tekken/Soulcalibur 的 sidestep 让同一招在距离不变时挥空。这没法用布尔标记硬塞，
碰撞必须变成体积的。几何核心（`space3d.py`）与 `--arena3d` 已接进自由二维环境
（39 维观测 / 10 动作，旧的 35 维 / 9 动作路径不变）：

| 函数 | 作用 |
|---|---|
| `Vec3` | 位置/速度，含加减、点积、归一化 |
| `point_segment_distance` | 点—线段距离（躯干对肢体） |
| `segment_segment_distance` | 线段—线段距离（肢体对肢体，Ericson RTCD 5.1.9） |
| `capsule_distance` / `capsules_overlap` | 胶囊—胶囊（线段距离减两个半径） |
| `sidestep_clears` | 侧移是否足以离开攻击线（2D 恒为 `False`） |
| `isometric` / `orthographic` | 把 3D 状态投回现有 2D 字符画，调试器不用重写 |

已落地的 z 轴规则：

| 规则 | 实现 | 可观测效果 |
|---|---|---|
| 纵深 | `Fighter.z/vz`，场地半深 1.2 m（`env_cfg["depth"]` 可改） | 撞到深度边界停住，不反弹 |
| 侧移 | `SIDESTEP`（动作 9）是一次 18 帧的侧跳：外 9 帧 + 回 9 帧，回程自动 | 峰值错开 0.45 m；回程结束回到攻击线，可接反击 |
| 命中 | `_hit_info` 先查 `space3d.sidestep_clears`，错开 > 0.32 m 即挥空 | 距离不变、直拳照样打不到 |
| 肢 clash | `_limbs_align()` 让纵深错开的肢体不相撞 | 屏幕上重叠但不同深度不发生 clash |
| 身位分离 | `bodies_separate()` 在错线时关闭 | 侧移不会被碰撞求解器推回来 |
| 观测 | 双方 `z / depth_half`、`vz / 4`，共 +4 维 | 策略能看见自己是否离开攻击线 |
| 兑现（实验性） | `--evade-reward`：active 帧挥空且防守方在侧跳中时付费，**默认关闭** | 40 轮 A/B：开 0.8 是 .375，关是 .500 —— 奖励反而更差，故只保留开关（第二十二节 22.5） |
| 横扫 | 侧跳任意帧按 `KICK` 取消剩余侧跳，起手 `SWEEP`（5/4/12 帧，伤害 6.5），纵深容差 0.55 m | 侧跳第一次有了打回去的手段；训练 0.61 次/局，但确定性 eval 从不选它（22.6） |
| 渲染 | 字符画按 `x + 0.5·z` 投影并显示 `z` | 侧移在终端里可见 |

路线：**几何（已完成）→ z 轴侧移动作与 z 观测（已完成）→ 3D 肢体胶囊姿态 → 摄像机/渲染**。
实测见 FINDINGS 第二十二节。

> 评估预算：8 局 holdout 的单 seed 区间约 ±.37，不足以判定 3D 的优劣。扩到
> 24 局（`--eval-episodes 12`）后 2D .583 vs 3D .569，72 局配对差 −1.4 pp
> （95% CI ±.15，McNemar p=1.00）：**3D 与 2D 不可区分，既非净成本也非增益**。
> 同批对局切成三个 8 局窗口，差值符号会翻转（−16.7 / +16.7 / −4.2 pp）。
> 结论见 FINDINGS 22.7。

#### LLM 训练栈（`selfplay/nn/llmstack.py`）

一条 `--llm-stack` 把所有"大模型默认做法"叠上来，每项也能单独关掉做消融：

| 开关 | 对应的大模型做法 | 在这里做什么 |
|---|---|---|
| AdamW (`--weight-decay`) | 解耦权重衰减 | 只衰减权重矩阵，不碰 bias |
| `--lr-warmup` + `--lr-min-ratio` | warmup + cosine | 前几轮线性升温，再余弦退火到 10% |
| `--kl-coef` | RLHF 的 KL 锚 | 每一步付 `β·KL(π‖π_BC)`，不许为了奖励跑离先验太远 |
| `--ema-decay` | 模型汤 / EMA | 影子权重，单独评测，指标是 `ema_win` |
| `--rft-frac` | 拒绝采样微调（STaR/RFT） | 每轮留收益最高的 25% 轨迹，再做 1 个 epoch 监督学习 |
| `--mirror-bc` | 数据增广 | 左右镜像 BC 示范，免费翻倍（场地对称、动作是"朝对手"的相对语义） |
| `--entropy-final` | 采样温度退火 | 熵系数线性降到终值 |

`--holdout-level N` 把评估对手钉死在 `CURRICULUM_LEVELS[N]`，避免课程在训练中途换敌人导致 A/B 不可比。
实测（freeform，困难 holdout，8 seeds）：**优化器四件套 +7 个百分点，而 KL 锚 + 拒绝采样把胜率打回 -24 个百分点**——
因为这两项会把策略磨尖、把熵压回噪声底，正好杀掉这个环境需要的探索。见 FINDINGS 第十五节。

#### 零数据自进化（`domains/neuro/selfevolve.py`）

`--selfevolve` 把外部数据全部撤掉：没有 BC 示范、没有手写对手阶梯、没有 KL 先验。
剩下的东西由智能体自己长出来：

```bash
python domains/neuro/train.py --env freeform --selfevolve \
    --iterations 45 --episodes 6 --max-frames 420 --out-dir runs/selfevo
```

| 机制 | 做法 |
|---|---|
| 对手池 | `random` + 每轮的自己快照 `leagueN` + 变异基因 `evoN`，**没有一个是对手设计师写的** |
| 胜负统计 | 每个对手一条 EMA 胜率（α=0.25），只有 `record()` 会更新，废弃对手无法污染统计 |
| 选谁打 | `--pfsp-mode pfsp`：权重 ∝ `(1−win)^β`；`progress`：权重 ∝ |Δwin|；`uniform`：对照 |
| 冷启动保护 | `--pfsp-floor`：从没赢过的对手权重 ×0.15，否则新手会被最难的敌人淹没 |
| 自造课程 | 每轮挑"胜率最接近 50%"的那个对手，变异它的基因（`--evolve-sigma`），保底随机基因 |
| 淘汰 | 连打 8 局胜率 ≤0.02 的对手退役；池子不超过 `--pool-size` |
| 指标 | `pool_size` / `pool_frontier` / `pool_frontier_win` / `pool_hardest` / `pool_games` |

实测（freeform，45 轮，每轮 6 局，各 2 seeds）：**20 轮以内 uniform 更稳，45 轮之后 PFSP 反超**
（PFSP 0.71 / 0.92，uniform 0.00 / 0.29）。原因不是采样器谁更好，而是 PFSP 的前提"你至少能赢一部分"
在冷启动时不成立——前 36 轮它把帧全给了打不过的对手，第 40 轮左右才真正开始进步。
**结论：零数据能学，但学得慢、方差大**；这套机制解决的是"不用人给数据"，不是"少算力"。
见 FINDINGS 第十六节。

#### 自己上手打（`domains/duel/play.py`）

同一个 freeform 环境，左边那个小人交给你：

```bash
# 你打算法
python domains/duel/play.py --opponent neural --ckpt runs/neuro_free

# 无键盘环境（CI / 远程）：让一个"有反应延迟的玩家"替你打，并打印每局战报
python domains/duel/play.py --auto --rounds 8 --reaction 12 \
    --opponent neural --ckpt runs/neuro_free
```

按键：`d` 前进、`a` 后退、`j` 直拳、`k` 踢（引擎按距离/姿态自动选前踢、头踢、低扫、飞踢）、
`s` 下蹲、`空格` 跳、`l` 格挡、`e` 替身术（40 查克拉）、`q` 退出。界面是终端字符画：两条血条 + 查克拉条 + 防御槽 + 俯视 12 行场地；加 `--emotion` 会多两行实时心态（冷静/恐惧/愤怒/气势）。

`--auto` 不是"假 AI 陪你演戏"：它只读网络同样的观测（普通 35 维，`--emotion` 为 43 维），每 `--reaction` 帧（默认 12 ≈ 200 ms）
才重新决策一次，不会读取引擎里的 `me.move`；判断对方踢腿只看脚的离地高度，屏幕上你也能看到。

实测（10 局，`--style pressure`）：反应 12 帧时对 random 0/10、脚本 4/10、神经 0/10；
把反应压到 2 帧（33 ms，接近键盘玩家的手感）后，random 10/10、神经 **3/10**。
也就是说算法赢的几乎全部是"零延迟 + 每秒 60 次决策"的帧优势，而不是更聪明的战术。
详见 FINDINGS 第十二节。

#### 算力账本：每 CPU 秒换来多少 Elo

```bash
python tools/bench.py --steps 20000            # 单核微基准：环境 / 观测 / 前向
python tools/bench.py --steps 20000 --json runs/bench.json
```

本机实测（单核，freeform）：

| 环节 | 耗时 | 占一步 |
|---|---|---|
| 环境 `step` | 38.2 µs | 100% |
| 其中观测构造 | 21.3 µs | 56% |
| 策略前向 29→64→8 | 30.7 µs | — |

端到端约 **14.5k 帧/s**（估算），一次 8 iteration、6 局 × 500 帧的训练实测
**3027 帧/s**——差距全部来自评估和 PPO 更新。训练会把每个 iteration 的分段耗时写进
`compute.jsonl`（`collect` / `ppo` / `eval` / `wall_sec` / `frames_per_sec` / `elo_per_kcs`），
`metrics.jsonl` 只保留可复现的技能指标。

实测结论（同 seed、8 iteration、6 局 × 500 帧）：

| 配置 | 总耗时 | collect | eval | 最终 holdout |
|---|---|---|---|---|
| 默认（每轮评估） | 7.9 s | 2.4 s | **5.1 s（65%）** | 0.875 |
| `--eval-every 4` | **4.2 s** | 2.5 s | 1.3 s | 0.875 |

同一份技能，少一半算力。另有一个零风险的微优化：姿态正向运动学按
`(位置, 状态, 招式, 帧桶)` 缓存后，articulated 每步 **−13%**、freeform **−8%**
（A/B 实测，见 `tools/bench.py`）。详见 FINDINGS 第十三节。

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

# 自由二维版本（35 维观测 / 9 动作，含防御槽、破防与 counter-hit）
python domains/neuro/train.py --env freeform --entropy-coef 0.05 \
    --iterations 8 --episodes 6 --out-dir runs/neuro_free

# 打开刚体物理（质量可给双方不同：重的人难推动但走得慢）
python domains/neuro/train.py --env freeform --physics --mass0 50 --mass1 100 \
    --iterations 20 --episodes 4 --out-dir runs/neuro_weight

# 打开第三轴：39 维观测 / 10 动作（SIDESTEP）
python domains/neuro/train.py --env freeform --physics --arena3d \
    --iterations 20 --episodes 4 --out-dir runs/neuro_3d

# 侧移成功奖励是实验开关（默认 0 关闭；开 0.8 实测更差，见 FINDINGS 22.5）
python domains/neuro/train.py --env freeform --physics --arena3d \
    --evade-reward 0.8 --iterations 20 --episodes 4 --out-dir runs/neuro_3d_evade

# 判定机制优劣时用 24 局 holdout：8 局窗口的差值符号会随机翻转（22.7）
python domains/neuro/train.py --env freeform --physics --arena3d \
    --iterations 40 --episodes 4 --eval-episodes 12 --out-dir runs/neuro_3d_24

# 再打开情绪层（冷静/恐惧/愤怒/气势，43 维观测）
python domains/neuro/train.py --env freeform --emotion --entropy-coef 0.05 \
    --iterations 15 --episodes 5 --out-dir runs/neuro_emo

# 再用 8 维隐空间世界模型做预测（观测 -> 8 -> 预测 -> 重建）
python domains/neuro/train.py --env freeform --latent-world --latent-dim 8 \
    --entropy-coef 0.05 --iterations 8 --episodes 6 --out-dir runs/neuro_latent

# 异质量对局：把体重写进观测（+2 维），否则轻量席位在蒙眼开车（23.3）
python domains/neuro/train.py --env freeform --physics \
    --mass0 50 --mass1 100 --mass-obs --iterations 40 --episodes 4 \
    --eval-episodes 12 --out-dir runs/neuro_mass

# 体重课程：训练每局在 60–120 kg 采样，评估仍固定 50 vs 100（负结果，23.6）
python domains/neuro/train.py --env freeform --physics \
    --mass0 50 --mass1 100 --rand-mass 60 120 --iterations 40 --episodes 4 \
    --eval-episodes 12 --out-dir runs/neuro_rand_mass

# 让策略直接吃学到的 8 维隐向量（而不是 35 维手工特征）
python domains/neuro/train.py --env freeform --latent-policy --latent-dim 8 \
    --iterations 15 --episodes 5 --out-dir runs/neuro_latent_policy

# 好奇心（隐空间预测误差）当内在奖励
python domains/neuro/train.py --env freeform --latent-policy --surprise-coef 1.0 \
    --out-dir runs/neuro_curious

# 世界模型想象训练（Dyna）：真实一步 + 模型滚两步
python domains/neuro/train.py --env freeform --latent-world \
    --imagine-steps 2 --imagine-batch 32 --out-dir runs/neuro_imagine

# 权重空间进化（PBT）
python domains/neuro/train.py --env freeform --pbt --pbt-size 4 --pbt-games 8 \
    --pbt-every 3 --pbt-adopt --out-dir runs/neuro_pbt

# 用连续适应度（剩余血量差）而不是二值胜负来驱动 PBT
python domains/neuro/train.py --env freeform --pbt --fitness margin \
    --pbt-size 4 --pbt-games 8 --out-dir runs/neuro_pbt_margin

# RND 好奇心：奖励"没见过的状态"，而不是"世界模型没预测到的跳变"
python domains/neuro/train.py --env freeform --latent-policy --latent-dim 8 \
    --surprise-coef 0.5 --surprise-mode rnd --out-dir runs/neuro_rnd

# 配对 A/B：同一批评估种子，逐 iteration 比较
python tools/ab.py --a runs/exp_a --b runs/exp_b --metric holdout_win

# 调整升级耐心 / 世界模型门控阈值
python domains/neuro/train.py --curriculum-patience 3 --wm-level-threshold 0.015 \
    --out-dir runs/neuro_tuned
```

评估结果的写法也变了：`metrics.jsonl` 现在每轮都带 `holdout_ci_low / holdout_ci_high`
和逐局 `holdout_outcomes`。**8 局的标准误约 ±0.17**，比较两个 run 请用 `tools/ab.py`
的配对区间，而不是比两个小数。

产物：`manifest.json`（run 元信息）、`metrics.jsonl`（逐 iteration 指标，可复现）、`compute.jsonl`（分段耗时与 Elo/CPU 秒，与机器相关所以单独存）、`progress.tsv`（表格版）、`state.json` + `state.npz`（可 resume）。

```bash
python -m unittest discover -s tests -v   # 196 个测试：反向传播、PPO、BC、课程、resume 等价、health、肢体几何/接触、自由二维/替身术/防御槽/counter、情绪层、刚体物理/墙撞/重量级/体重观测/随机体重、3D 几何/纵深侧移/错线挥空/侧跳横扫/evade 兑现、玩家模式、隐空间世界模型、LLM 训练栈、零数据自进化、算力账本、统计区间、PBT、想象训练、RND 好奇心、连续适应度
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
           freeform.py    自由二维版：加竖直轴、跳跃/下蹲/替身、防御槽、情绪（35/43 维观测、9 动作）
           affect.py      四维情绪漏积分器（冷静/恐惧/愤怒/气势）
           space3d.py     3D 几何：Vec3 / 线段-线段 / 胶囊 / 侧移判定 / 投影
           play.py        玩家模式：键盘操控 / 延迟玩家 vs 神经策略
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
  neuro/   env.py       把 duel / articulated / freeform 包装成 15 / 23 / 35 维观测（情绪 +8、质量 +2）、6 / 9 动作的环境
           opponents.py random / scripted / 冻结快照 对手池 + BC 专家
           train.py     BC 预热 + PPO + league + 世界模型 + health + resume
           selfevolve.py 零数据自进化：对手池、胜率 EMA、PFSP / 学习进度采样、基因变异
selfplay/
  nn/      mlp.py       MLP + Adam/AdamW + 手写反向传播
           ppo.py       离散动作 PPO（GAE / clip / entropy / KL 锚）
           world_model.py  one-step 观测预测模型
           latent.py    8 维隐空间世界模型（encoder/predictor/decoder）
           llmstack.py  LR 调度、EMA、拒绝采样、镜像增广
  core/    health.py    评估器健康判据
           curriculum.py 世界课程（health + world model 双门控）
           budget.py    算力账本：分段计时 + Elo/CPU 秒
           stats.py     bootstrap 区间 / 配对差 / Wilson 区间（小样本专用）
           manifest.py  run 元信息
           recorder.py  metrics.jsonl / progress.tsv / resume 状态
tools/     bench.py     单核微基准：环境步进 / 观测 / 前向
           ab.py        配对 A/B：逐 iteration 差值 + 95% 区间
tests/     test_neuro.py 神经栈的单元与回归测试
           test_limbs.py 肢体运动学、胶囊几何与接触规则测试
           test_freeform.py 跳跃/下蹲/低扫/飞踢与二维物理测试
           （含玩家模式的反应延迟与战报对账）
           test_llmstack.py LR 调度、EMA、拒绝采样、镜像增广、AdamW
           test_selfevolve.py 对手池、PFSP/progress 采样、基因变异、淘汰与池容量
           test_stack.py  统计区间、配对差、PBT、隐空间奖励头、想象训练
           test_budget.py  Elo 换算与算力账本

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
