# SelfPlay Lab

**三个域，一套机制：自己跟自己打，用遗传搜索找最优策略，让"世界"在冠军打满时自动变难。**

这个仓库想回答一个问题：

> **自进化 / 世界模型这条路，到底能不能走通？**

结论先说：**机制走通了，瓶颈在评估器。** 详细证据见 [`docs/FINDINGS.md`](docs/FINDINGS.md)。

---

## 一、三个域

同一套代码骨架（基因 → 个体 → 对战/解题 → 适应度 → 选择 → 变异 → 世界升级），换三个"世界"：

| 域 | 世界是什么 | 基因 | 代数 | 难度 | 冠军状态 |
|---|---|---|---:|---|---|
| **code** | 13 族算法题 + 程序化 bug 注入 | 10 维（读码/改动范围/重试/回滚/测试/重构/文档/并行/回滚速度/脚手架） | 2302 | Lv5（5 维全开） | fit 1.000 —— **打穿了** |
| **duel** | 两个火柴人徒手对打（纯 numpy 物理） | 9 维（武器/攻击性/格挡/踢/距离/反应/待机/捡道具/绕陷阱） | 61 | Elo 制 | fit 0.508 → 1.126，Elo 1446.7 |
| **sgs** | 三国杀式卡牌单挑 | 4 维（攻击性/防守/回血/距离） | 949 | Lv4（4 维全开） | fit 0.573 —— **零满分，还在爬** |

> 数据取自各域 `evo/champions.jsonl` 的 unique gen 计数，截至 2026-10-29 21:30。
> 三个域当前都还在后台跑，数字会继续长。

三个域的**难度行为正好是三种典型状态**，这也是最有价值的部分：

- **code**：冠军 99.6% 满分（3603/3617 条记录 fit=1.000）→ **评估器被打穿，进化退化成随机漫步**
- **sgs**：零满分，fit 0.573~0.896 波动，mean 0.479 → **梯度真实存在，还在选**
- **duel**：fit 从 0.508 单调爬到 1.126，流派收敛到「反击×20」→ **收敛成功，但趋同**

---

## 二、机制

### 2.1 基因与个体

每个个体是一串实数基因，直接映射成一份可执行的策略。没有神经网络，没有梯度——**只有遗传搜索**。

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

---

## 三、怎么跑

### 3.1 依赖

```bash
pip install numpy
```

就这一个。三个域都是纯标准库 + numpy。

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
           bot.py       基因 → 策略
           evolve.py    进化主循环（含 Elo 与名人堂）
           verify.py    规则自检
  sgs/     env.py       卡牌单挑环境
           evolve.py    进化主循环

data/      三个域的进化轨迹（每 N 代抽样）+ 难度晋升历史 + 当前冠军
docs/      FINDINGS.md —— 走通到哪一步、崩在哪、下一步怎么扩
```

> `data/*_progress.tsv` 是从原始轨迹按 10:1 / 5:1 抽样后的版本，原始文件在每个域自己的 `evo/` 下。
> duel 域的 46MB 逐帧对战数据 `duel.csv` 没有入库，用 `collect_duel.py` 可重新生成。

---

## 五、最重要的一条

**自进化的瓶颈不是搜索算法，是评估器。**

搜索再聪明，只要评估器被打穿（冠军轻松满分），选择压力就归零，进化立刻退化成随机漫步——代码在跑、日志在涨、指标好看，但**什么也没学到**。

code 域 2302 代、99.6% 满分，就是这个失败模式的标本。sgs 域零满分、fit 长期在 0.57 上下挣扎，反而是**健康**的。

判断一个自博弈系统是不是真在学，只看一个指标：**冠军的适应度有没有顶到天花板。** 顶住了，就是该升难度或者换题了。

---

## License

MIT
