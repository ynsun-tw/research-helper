# 使用场景：从一个想法到一篇可投稿草稿

这是一份贴近真实工作流的中文使用指南。我们用同一个虚构但合理的研究员
**小李** 串起四个场景，覆盖 Research Agent 当前实现的全部能力（M1 ~ M5
里 E5.3 + E5.4 之前的所有特性）。每个场景都包含：

- 起点状态（小李在做什么、文件夹里有什么）。
- 实际敲入的命令（slash 或 CLI）。
- 应当看到的输出结构（不是字面 LLM 文本，而是表格、面板、退出码等
  确定性的部分）。
- 输出落到了本地哪些文件 / 表里，下一步可以怎么消费。
- 常见陷阱与处理方式。

> 适用版本：v0.5.0（Test PyPI 上已发布）。
> 配置文件：`~/.research-agent/config.yaml`。
> 所有本地状态都在 `~/.research-agent/` 下，可以随时整个目录备份或
> 删除重来。

如果你只想看英文版的完整 slash/工具调用清单，参见
[`end-to-end-demo.md`](./end-to-end-demo.md)。

> **2026-05 后**：Research Agent 把全部能力都暴露成 LLM 工具，
> 自然语言可以直接触发。下面的场景给出的是"显式 slash 命令"路径，
> 是因为输出形态最确定、最适合作为参考。如果你想看自然语言怎么走完
> 同一条路径，往下翻到[**附录 A — 全自然语言版**](#附录-a--全自然语言版)；
> 两条路径背后调用的是同一套工具。

---

## 人物：研究员小李

- 方向：长上下文 LLM 的高效注意力（稀疏 / 路由 / KV cache）。
- 工作机：macOS，已经用 pipx 安装了 `paper-research-agent==0.5.0`。
- OpenRouter key 已经写进 `config.yaml`，默认模型
  `deepseek/deepseek-chat`，`language: zh`。
- 手头有两篇自己已发表的论文 PDF（用来训练 Scribe 风格）：
  `~/papers/li-2023-sparse-routing.pdf` 与
  `~/papers/li-2024-attention-cache.pdf`。

小李这一周的目标：

1. 把 "top-k 稀疏注意力 + 学习路由" 这条 idea 调研清楚（场景 A）。
2. 写 introduction 的初稿（场景 B）。
3. 给方法部分准备一张架构图、给实验部分准备一张结果图（场景 C）。
4. 周五晚上做一次复盘 + 环境体检（场景 D）。

---

## 场景 A：从一个想法到一份带凭据的文献清单

> **目标**：把脑子里 "top-k 稀疏注意力" 的模糊想法，变成一份
> Critic 批过、Analyst 总结过、并有具体引用图谱的研究笔记。
> **时长**：30 ~ 45 分钟（取决于 LLM 速度）。
> **覆盖**：M3 全部 + M2 辩论 + M1 论文加载。

### A.1 进入 REPL 并先看一眼现状

```bash
research
```

欢迎面板渲染后看到 `›` 提示符。小李先确认上下文是空的：

```text
› /history
```

输出：`No prior /search queries recorded yet.`（首次会话符合预期）。

### A.2 用 `--mode` 圈定调研方向

先看理论侧：

```text
› /search --mode theoretical efficient attention long context
```

终端先后渲染：

```
Biasing search toward theoretical papers.
Searching arXiv…
Scoring relevance (LLM)…
┌──────────┬─────────────────────────────────────┬──────┬───────┬────────────────────┬──────┐
│ ID       │ Title                               │ Year │ Score │ Why                │ Read │
├──────────┼─────────────────────────────────────┼──────┼───────┼────────────────────┼──────┤
│ 2009.06732 │ Efficient Transformers: A Survey  │ 2020 │ 0.92  │ Direct survey of … │   ·  │
│ 2007.14062 │ Big Bird: Transformers for …      │ 2020 │ 0.86  │ Sparse attention … │   ·  │
│ 2006.04768 │ Linformer: Self-Attention with …  │ 2020 │ 0.83  │ Low-rank approx …  │   ·  │
└──────────┴─────────────────────────────────────┴──────┴───────┴────────────────────┴──────┘
Sorted by relevance (LLM, 0-1).
```

再看实验侧：

```text
› /search --mode applied sparse attention 32k context benchmark
```

`Biasing search toward applied papers.` 横幅出现，候选集会偏向有完整
benchmark 报告的工作（FlashAttention、Longformer 实验对照等）。

### A.3 把候选堆进阅读队列

```text
› /queue add 2009.06732 Efficient Transformers: A Survey
› /queue add 1706.03762 Attention Is All You Need
› /queue add 2007.14062 Big Bird
› /queue list
```

`/queue list` 给出三行 `status=pending`。

也可以让 LLM 工具调用做这件事：

```text
› 把 reformer 那篇（2001.04451）也加进来
```

终端先显示 `→ calling queue_add (arxiv_id=2001.04451, title="Reformer")`，
然后是确认信息。

### A.4 顺着队列读第一篇

```text
› /queue read
```

后台动作：从 arXiv 拉 PDF → 写入 `~/.research-agent/cache/papers/`
→ PyMuPDF 解析 → Analyst + Critic 并行调用 LLM → 把 paper 行存进
`papers` 表 → 把阅读队列那条 mark 成 `done`。

屏幕上你会看到四块面板：

1. **Paper header**：标题、作者、年份、abstract 截断。
2. **Analyst 报告**：贡献点、方法洞察、潜在影响、关联工作、声明 vs
   证据对齐度、置信度 0~1。
3. **Critic 报告**：异议列表、支持分数（0~9，永远不给 10）、打分理由、
   诚实性备注。
4. **Read summary footer**：`Loaded arxiv:1706.03762 · score 0.78/9 ·
   read_at 2026-05-26T10:14:02Z`。

### A.5 与 paper 对话，沉淀 idea

```text
› /discuss 把 dense O(n²) 自注意力换成 top-k 稀疏注意力（每个 query 只
关注得分前 k 的 key），目标 32k 上下文。路由用 query 的一个学习线性
投影。
```

第一轮是 **结构化辩论**：Analyst 列贡献假设和已有相关工作（Reformer、
Routing Transformer 等）、Critic 列 objections（top-k 不可微、质量损失、
KV 带宽是真正瓶颈等）并给一个 0~9 的分数。

后续追问就走 **散文模式**，不再重打分：

```text
› 那如果用 Gumbel-top-k 让路由可微，Critic 的第二条反对还成立吗？
```

输出会引用第一轮里 Critic 提到的具体那条 objection，Analyst 给出技术
建议，Critic 给短反驳。

### A.6 保存 idea，再走一次想法提醒

```text
› /idea save top-k-sparse-attention-with-learned-routing
› /ideas
```

`/ideas` 列出刚保存的那条，状态 `active`，附最近一次 critic 分数。

现在把这条 idea **暂时搁置**，并钉一个 "解锁条件"——意思是 "等什么事情
发生再回来看"：

```text
› /ideas update <id-prefix> --status shelved --condition "FlashAttention-3 release" --condition "可微 top-k 的新进展"
```

然后再去读一篇相关的论文：

```text
› /read 2007.14062
```

读完之后，如果这篇论文的语义和小李那条 shelved idea 高度相关（默认余弦
≥ `alert_threshold=0.80`），屏幕上会自动跳一条提醒横幅：

```
🔔 Parked idea(s) may relate to this paper:
   • <id-prefix> Top-k sparse self-attention with learned routing  (similarity 0.84)
       /idea show <id-prefix>
```

如果新论文的标题 / 摘要里**字面**包含 `FlashAttention-3` 之类的关键词
（活化条件），还会再给一行 "Shelved idea(s) may have an unblock" 横幅。
这条机制平时基本看不到，但一旦命中，说明你确实该回头看那条想法了。

### A.7 引用图谱：找谁引用了、它又引用了谁

```text
› /cites              # 谁引用了当前 anchor paper
› /refs               # 当前 anchor 引用了谁
```

两张表都通过 Semantic Scholar 拉取，附年份和 venue。可以从某条结果直接
`/queue add <id>` 继续往下挖。

### A.8 让 Searcher 给一个 "下一步搜什么"

辩论积累的上下文够多以后，可以把决策权交给 Searcher：

```text
› /refine
```

终端会给出一条建议：

```
Suggested next search: top-k sparse attention KV cache (mode: applied)
Reason: Critic 在 A.5 里把内存带宽列为真正瓶颈，对应的实验工作还
没读过。
                                                  (confidence 78%)

Run this search? [y]es / [e]dit / [s]kip ›
```

`y` 直接派发；`e` 进入编辑模式（preserve mode 标记），允许小李把
`KV cache` 改成 `KV cache budget`。

### A.9 跨会话回忆

下次重启 REPL（哪怕过了三天），可以这样把上文捞回来：

```text
› /recall 稀疏注意力和 KV cache 带宽
```

或者用自然语言：

```text
› 上次我和 Critic 讨论的关于 KV bandwidth 的反对意见，结论是什么？
```

LLM 会调用 `recall_history(query=..., limit=3)`，从
`discussions` 向量索引里语义检索过去会话里的相关消息，再把它们当
上下文给出复盘。

> **本地落点**：截至本场景结束，
> - `papers` 表 ≥ 2 行（含 anchor + 一篇关联）。
> - `ideas` 表至少一行，`activation_conditions` 字段 JSON 数组里
>   有两条字符串。
> - `discussions` 表里 user / analyst / critic / system 多角色记录。
> - `search_queries` ≥ 2 条（理论 + 实验两次 search）。
> - `~/.research-agent/chroma/` 下有 `ideas` 和 `discussions` 两个
>   向量集合。

---

## 场景 B：从笔记到稿子——用 Scribe 起草并审查

> **目标**：把 idea 落成 introduction 段落，让 Analyst + Critic 在
> 写作模式下挑刺，迭代到自己满意，并把改稿沉淀回 fingerprint。
> **时长**：1 ~ 1.5 小时（含人工审稿）。
> **覆盖**：M4 全部（E4.1 ~ E4.4）。

### B.1 训练自己的写作语料

```bash
research style train ~/papers/li-2023-sparse-routing.pdf ~/papers/li-2024-attention-cache.pdf
```

输出片段：

```
Importing 2 source(s)…
  li-2023-sparse-routing.pdf  → 47 paragraphs (after filter)
  li-2024-attention-cache.pdf → 52 paragraphs (after filter)
Stored 99 prose paragraphs in style_samples.
```

进 `~/.research-agent/memory.db`：

```bash
sqlite3 ~/.research-agent/memory.db \
  "SELECT paper_id, COUNT(*) FROM style_samples GROUP BY paper_id;"
```

应该看到两个 paper_id 各对应几十段。

> 过滤规则：丢弃太短 / 太长 / 数学占比过高 / 字母占比 < 60% 的段落，
> 也丢弃 References / Acknowledgements / Appendix 等 section。

### B.2 构建第一版 fingerprint

```bash
research style fingerprint
```

它把刚才那 99 段做统计分析：

- 宏观：abstract opener、intro opener、related work 组织策略、平均段
  落数。
- 微观：句长 mean / median / p10 / p90、过渡词频率、hedging vs
  confidence 比、被动语态率、type-token ratio。
- 个人标记：citation 格式（`\cite{}` / `[N]` / `(Author, year)`）、
  figure ref 写法、em-dash 使用率。

落盘到 `~/.research-agent/style/fingerprint.json`。可以
`research style show` 看摘要：

```
Style training corpus:
  papers: 2
  samples: 99
Fingerprint (v1):
  abstract opener: "We propose a"
  intro opener: "Self-attention has become"
  related-work strategy: thematic
  sentence length: mean 19.4 (p10/p90: 11 / 32)
  hedging per 100 sentences: 2.8
  confidence per 100 sentences: 1.4
  passive voice per 100 sentences: 6.1
  citation format: latex_cite
```

### B.3 让 Scribe 起草

借助场景 A 里沉淀下来的 idea + 讨论上下文：

```bash
research write introduction \
  --context "我们提出 top-k 稀疏注意力 + 学习路由，目标 32k 上下文，
             路由是 query 上的线性投影，主要在带宽受限场景下竞争 FlashAttention" \
  --versions 3 \
  --words 320 \
  --output drafts/intro.md
```

发生了什么：

1. `cli_write` 加载 fingerprint v1。
2. `--context` 触发 MemoryKeeper 自动捞取场景 A 里相关 idea + 最近的
   `/discuss` 片段（语义相似度 ≥ 0.5，各 top 3）。
3. Scribe 用 3 个不同 directive（concise / technical depth /
   narrative arc）**并行**调用 LLM，得到 3 份草稿。
4. 终端先打印一行 `Scribe context: user context, 1 related idea(s),
   3 discussion excerpt(s)`，再渲染三个版本的 Rich panel。
5. 同时把 3 份草稿写进 `drafts/intro.md`，每份 `## Version X — <variant>`
   一节。

如果想再生成一版但保持与既有 method.md 一致：

```bash
research write introduction \
  --context "同上" \
  --check-against drafts/method.md \
  --versions 2 \
  --output drafts/intro-v2.md
```

`--check-against` 会把 method.md 的正文（截到 ~2KB）塞进 prompt，并
明确告诉 Scribe **不要重复也不要矛盾**。

### B.4 审稿：Analyst + Critic 并跑

```bash
research review drafts/intro.md \
  --section introduction \
  --interactive
```

后台：

1. 把 `intro.md` 视作单段草稿。
2. `Orchestrator.collect_writing_reviews_async` 同时（`asyncio.gather`）
   跑两个 agent：
   - **Analyst（写作模式）**：列论据强弱、与相关工作的差异化、证据
     vs 声明对齐、缺失的 context。
   - **Critic（写作模式）**：列过度声明、无依据结论、逻辑断点、
     hedging 不匹配、自相矛盾。
3. 进入 **交互选择**——逐条 y / n / s（skip）地接受或拒绝每条
   issue 和 suggestion。
4. 选完后，把 **只被接受的那部分** 反馈丢给 Scribe，让它出一份修订版。
5. 终端打印 4 个面板：原稿、Analyst 报告、Critic 报告、修订稿，
   并在底部给一个 **带颜色的 unified diff** 高亮变化。

接受 / 拒绝的全部决策 + 原稿 + 改稿都会写进 `draft_revisions` 表
（除非加了 `--no-save`）。这是下一步学习的语料。

### B.5 持续学习：让 fingerprint 跟上你的改稿习惯

```bash
research style update
```

它会：

1. 重新算 fingerprint，输入是 `style_samples`（原始论文段落） +
   `draft_revisions.revised_text`（你接受了 reviewer 反馈之后的改稿）。
2. 把上一版 `fingerprint.json` 重命名成 `fingerprint_v1.json`，新版
   存到 `fingerprint.json`。
3. 终端打印新旧两版的并排对比（句长 / hedging / passive 等关键字段
   diff）。

历史一目了然：

```bash
research style history
```

输出：

```
Fingerprint versions in ~/.research-agent/style/
  fingerprint.json      v2  (current)  · samples=99  · revisions=4
  fingerprint_v1.json   v1               · samples=99  · revisions=0
```

> **意义**：你写得越多、改得越多，fingerprint 就越贴近你**当下**的
> 嗓音，而不是几年前的那篇论文。

### B.6 自查重：别在自己的语料里抄自己

每次 `research write` 完都建议跑一次：

```bash
research check drafts/intro.md --threshold 0.4 --output reports/intro.similarity.md
```

它对每段做 TF-IDF + 余弦相似度，跟 `style_samples` 全量比对，把命中段
按相似度从高到低列出来，每条配一条**按严重程度分级**的改写建议：

- ≥ 0.7 → "rewrite from scratch"
- ≥ 0.5 → "paraphrase and cite"
- ≥ 0.4 → "trim or merge"

退出码：干净 = 0，有命中 = 2。所以可以直接塞进 CI：

```bash
research check drafts/*.md --threshold 0.5 || {
  echo "self-plagiarism detected" >&2
  exit 1
}
```

---

## 场景 C：从草稿到图——三种图，一个命令

> **目标**：一晚上敲完 introduction 后，给 method 和 results 各配一
> 张图，省去手画 TikZ / 调 matplotlib 的时间。
> **时长**：30 分钟。
> **覆盖**：M5 E5.3（S5.3.1 ~ S5.3.3）。

### C.1 架构图（TikZ）

```bash
research figure --type architecture \
  --desc "三层稀疏 Transformer encoder：第一层全连接注意力，第二、三层
         用 top-k 路由稀疏注意力，每层之间有残差和 LayerNorm" \
  --versions 2 \
  --output figs/architecture.md
```

发生了什么：

1. `Illustrator(figure_type="architecture")` 启动，加载
   `prompts/illustrator_architecture.yaml` 这套系统提示词。
2. 并行跑 2 个 variant：`layered horizontal` 和 `hub-and-spoke`。
3. 输出是 JSON，含 `code` / `style_label` / `notes` / `suggested_use`
   四个字段。Rich 渲染时把 `code` 当 LaTeX 高亮。
4. `figs/architecture.md` 里每个 Version 一段，代码用
   ```` ```latex ```` 围栏，方便直接粘进 LaTeX 文档。

终端预览（节选）：

```
┌─ Version A · architecture · layered horizontal ──────────────────┐
│ \usetikzlibrary{positioning, arrows.meta}                        │
│ \begin{tikzpicture}[                                             │
│   block/.style={draw, rounded corners, minimum width=2.4cm,      │
│     minimum height=0.8cm, font=\small},                          │
│   arrow/.style={-Stealth, thick}]                                │
│   \node[block] (in)  {Input Embedding};                          │
│   \node[block, right=of in] (l1) {Layer 1: Dense Attn};          │
│   \node[block, right=of l1] (l2) {Layer 2: Top-k Sparse};        │
│   \node[block, right=of l2] (l3) {Layer 3: Top-k Sparse};        │
│   …                                                              │
│ \end{tikzpicture}                                                │
└──────────────────────────────────────────────────────────────────┘
notes: stacks layers L→R; explicit residual arrows under each block
suggested use: system overview at start of methods section
```

### C.2 结果图（matplotlib，可顺手跑一遍验证）

```bash
research figure --type result \
  --desc "三种方法在 32k 上下文 perplexity 上的比较，三条 baseline + 我们" \
  --data "ours: 8.1; FlashAttn-2: 8.6; Longformer: 9.4; BigBird: 9.9" \
  --versions 2 \
  --verify \
  --output figs/results.md
```

`--verify` 会把每个 variant 的 Python 代码塞进临时目录，用当前解释器
（`MPLBACKEND=Agg`，30 s 超时）真正跑一遍。终端会打印：

```
✓ Version A ran successfully.
✗ Version B failed: ModuleNotFoundError: No module named 'seaborn'
```

`Version B` 的 stderr 末尾几行会写进 `figs/results.md` 的代码块旁边，
方便你知道是装个 seaborn 还是改一下 prompt 再生成一次。

> 注意 `--verify` 不是隔离环境，跑的就是你当前的解释器（真隔离环境
> 是 E5.1 reproduce 的范围，已经 defer 了）。所以它**只能**告诉你
> 代码至少能 import + run；要审美还是要看图。

### C.3 概念图 prompt（用于 DALL·E 3 / Midjourney / SD）

```bash
research figure --type concept \
  --desc "top-k 路由在 query-key 空间里挑出 top-k 邻居的几何示意" \
  --versions 3
```

3 个 variant 会**分别** tune 给三家模型：

- Version A → DALL·E 3：完整的自然语言描述句。
- Version B → Midjourney v6：逗号分隔关键词 + `--ar 16:9 --style raw`。
- Version C → Stable Diffusion：带权重 `(clean line art:1.3)`、有
  `negative_prompt`（`photorealistic, cluttered, text, watermark`）。

Rich 输出在 header 上会标 `target: dalle3` / `midjourney` / `sd`。
负面 prompt 单独以 `[dim]negative:[/dim]` 一行渲染。

> **直接渲染图片**（DALL·E API 调用）是 S5.3.3 的可选子任务，本里程碑
> 没做。一旦做了就只需要再加一个 `--render` 开关。

---

## 场景 D：周复盘 + 环境体检

> **目标**：周五晚上 5 分钟，确认环境健康、产出一份本周的 Markdown
> 复盘报告。
> **时长**：5 分钟。
> **覆盖**：M5 S5.4.2（doctor）+ M3 S3.3.2（insights）。

### D.1 体检

```bash
research doctor
```

输出是一张 Rich 表格，每行一个 check：

```
                                          research doctor
┌──────┬───────────────┬─────────────────────────────────────────────┬────────────────────────┐
│      │ Check         │ Status                                      │ Hint                   │
├──────┼───────────────┼─────────────────────────────────────────────┼────────────────────────┤
│  ✓   │ config file   │ ~/.research-agent/config.yaml (mode 0o600)  │                        │
│  ✓   │ api key       │ set (sk-or-…f3a2)                           │                        │
│  ✓   │ data dir      │ ~/.research-agent                           │                        │
│  ✓   │ database      │ ~/.research-agent/memory.db (integrity OK)  │                        │
│  ✓   │ chroma dir    │ ~/.research-agent/chroma                    │                        │
│  !   │ disk space    │ 312 MB free at ~/.research-agent            │ consider freeing space │
│  ✓   │ package       │ paper-research-agent 0.5.0                  │                        │
│  ✓   │ chromadb      │ imports cleanly                             │                        │
└──────┴───────────────┴─────────────────────────────────────────────┴────────────────────────┘
! 1 warn(s) (non-fatal)
```

退出码 0（全是 ok / warn）或 1（任何一个 fail）。塞进 CI / cron 都行。

常见处理：

| Check 失败 | 处理 |
|---|---|
| `api key  no api_key configured` | `research config set api_key sk-or-...` |
| `config file  mode 0o644` | `chmod 600 ~/.research-agent/config.yaml` |
| `database  unreadable` | 备份后删掉 `memory.db`，下次启动会重建空 schema |
| `chromadb  import failed` | `pip install --force-reinstall chromadb` |
| `disk space  only 67 MB free` | 清理或换 `data_dir`（`research config set data_dir <path>`） |

### D.2 周复盘

```bash
research insights --since 7d --output reports/2026-05-26-weekly.md
```

不调 LLM、纯 SQLite 聚合，毫秒级出报告，写到磁盘上。报告分三段：

```markdown
# Research Insights — last 7 days
_Generated 2026-05-26 18:42:11 UTC_

## Papers
- Total: 5
- By year:
  - 2024: 2
  - 2020: 2
  - 2017: 1
- Top tags: efficient-attention (3), sparse (2), kv-cache (2)
- Top authors: Vaswani (1), Choromanski (1), Zaheer (1), …

## Ideas
- Total: 2
- By status:
  - active: 1
  - shelved: 1
- Average critic score: 6.50/9
- Most-debated ideas (by score history length):
  - `c8e1` Top-k sparse self-attention with learned routing — 2 round(s)
- Highest-scoring ideas:
  - `c8e1` Top-k sparse self-attention with learned routing — 7/9

## Discussions
- Sessions: 4
- Messages: 38
- By role:
  - user: 14
  - analyst: 10
  - critic: 9
  - system: 5
- Most recent sessions:
  - `9a4f` — 2026-05-26 17:38:02
```

把 `reports/` 目录交给 git，几周下来就有一份能折线化的研究脉络。

`--since` 支持 `7d` / `4w` / `6m` / `1y` / `all`，纯整数当天数。

---

## 附录：调优与排错

### 多模型切换

```bash
research config set model anthropic/claude-3.5-sonnet     # Critic 推荐
research config set model openai/gpt-4o                   # 多模态友好
research config set model deepseek/deepseek-chat          # 默认 / 便宜
```

只要是 OpenRouter 列表里 slug，全都支持。所有 agent 走同一个 client。

### 中英语切换

```bash
research config set language zh      # agent 默认用中文回答
research config set language en      # 英文回答
```

切换只影响 agent 输出，不影响内部存储（讨论里的角色名、CLI 文案、
`/help` 文本均保留英文）。

### 自定义 idea 提醒阈值

```bash
research config set alert_threshold 0.7    # 更敏感，提醒变多
research config set alert_threshold 0.9    # 更挑剔，几乎只在近义复读时提醒
```

合法范围 `[0, 1]`，YAML 里写错（如 `1.5`）会被钳到边界、不会崩溃。

### 排除调试时的友好提示

默认情况下，未捕获的异常会被 `cli.main()` 包装成一行 Rich 彩色提示，
看起来像：

```
Unexpected error: ValueError: something
Run with RESEARCH_AGENT_DEBUG=1 to see the full traceback, …
```

真要看堆栈：

```bash
RESEARCH_AGENT_DEBUG=1 research <command>
```

`Ctrl-C` 触发 `KeyboardInterrupt` 的退出码是 130，REPL 会在退出前打
一行 `Interrupted.`。

### 强制不写向量库

调试 / CI 不希望 ChromaDB 写盘时：

```bash
RESEARCH_AGENT_TEST_MODE=1 research <command>
```

这会用关键词回退替代向量检索，相似度结果可能差一些，但不再依赖
`chroma/` 目录。

### 重置整个本地状态

```bash
rm -rf ~/.research-agent/memory.db \
       ~/.research-agent/chroma \
       ~/.research-agent/cache \
       ~/.research-agent/style
```

`config.yaml` 留着，下次启动会自动重建表 + 索引 + 空向量集合。
fingerprint 想保留就别删 `style/`。

### CI 集成片段（GitHub Actions）

```yaml
- name: Smoke test the CLI
  run: |
    pip install paper-research-agent==0.5.0 \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/
    research --version
    research doctor
    research check drafts/intro.md --threshold 0.5 || (echo "self-plagiarism" && exit 1)
```

---

## 结语

这四个场景大致覆盖了 v0.5.0 时点 Research Agent 的**全部已实现功能**：

| 场景 | 主要里程碑 | 命令 |
|---|---|---|
| A · 文献调研 | M1 + M2 + M3 全部 | `/search`, `/queue`, `/read`, `/discuss`, `/idea save`, `/cites`, `/refs`, `/refine`, `/recall` |
| B · 写作流水线 | M4 全部 | `research style train\|fingerprint\|update\|history`, `research write`, `research review`, `research check` |
| C · 配图 | M5 E5.3 | `research figure --type architecture\|result\|concept` |
| D · 体检 + 复盘 | M5 E5.4 + M3 S3.3.2 | `research doctor`, `research insights` |

尚未落地的功能（v0.5.0 显式 defer）：

- **E5.1**：`research reproduce <arxiv-id>`，自动建 venv + 跑作者代码
  + 出复现报告。
- **E5.2**：完整三栏 Textual TUI。

如果你的工作流踩到这两块需求，可以在 issue 里留言；本里程碑的代码框架
（subprocess 验证、隔离环境抽象、Rich 渲染分层）已经为后续接入留好位置。

---

## 附录 A — 全自然语言版

上面 A/B/C/D 四个场景每一步都对应一个 chat 工具调用。如果不想记
slash，把上面四节合并成一段连贯对话，效果完全一样：

```text
$ research

You> 帮我先看下环境正不正常
→ run_doctor()
[诊断表格]
All checks passed.

You> 搜一下高效注意力机制相关的理论论文，最近一年的
→ search_arxiv(query="efficient attention long context", mode="theoretical")
[评分后的论文表]

You> 把这几篇都加进待读队列
→ queue_add(arxiv_id="...", title="...")  (重复多次)

You> 把第一篇读完然后告诉我关键贡献
→ queue_next() → load_paper(source="...")
[Analyst + Critic 报告]

You> 我想讨论一下：用 top-k 稀疏注意力加学习路由替代 dense attention
→ discuss_idea(idea="...")
[结构化首轮辩论]

You> 把这个 idea 存下来，标题就叫 sparse-routing-attention
→ save_current_idea(title="sparse-routing-attention")

You> 看一眼这篇论文的引用图谱，谁基于它做了后续工作？
→ get_citations(arxiv_id="...")

You> 我的写作风格训过了吗？
→ style_show()
"Style corpus is empty. Suggest train_style first."

You> 把我自己的两篇论文当样本：~/papers/li-2023-sparse-routing.pdf,
    ~/papers/li-2024-attention-cache.pdf
"我会把这 2 个 PDF 解析成段落写入 style_samples 表，会覆盖这两篇之前的
样本。可以吗？"
You> 可以
→ train_style(sources=[...], append=false)
[导入统计]

You> 现在算指纹
→ build_fingerprint()

You> 帮我写一段 introduction，主题是稀疏 top-k 注意力支持 32k 上下文，
    大概 300 字
→ draft_section(section="introduction", context="...", target_words=300)
[3 个变体面板]

You> B 版本最好，存到 ~/drafts/intro.md
→ save_draft_to_file(path="~/drafts/intro.md", kind="section",
                     section="introduction", version="B")

You> 这段跟我自己以前发过的论文有没有重复？
→ check_self_plagiarism(target="latest:introduction:B")
[相似度报告]

You> 帮我再改一遍，让它更紧凑、把 Critic 的几个点都顾到
→ revise_draft(target="latest:introduction:B")
[issue 列表 + 修改后版本]

You> 满意，存到 ~/drafts/intro_revised.md
→ save_draft_to_file(path="~/drafts/intro_revised.md", kind="revision")

You> 给方法部分准备一张 TikZ 架构图：三层 encoder + 路由层
→ draft_figure(figure_type="architecture",
               description="three-layer encoder with learned routing")
[2 个 TikZ 草稿]

You> 第一版好，写到 ~/figs/arch.md
→ save_draft_to_file(path="~/figs/arch.md", kind="figure",
                     figure_type="architecture", version="A")

You> 这周做了啥？给个 30 天的回顾
→ research_insights(since_days=30)
[Markdown 报告]

You> 等会重新打开 REPL，把 fingerprint 更新一下，把刚刚的 revision 也学进去
→ update_fingerprint()

You> /exit
```

这里**唯一**写到磁盘的瞬间，是用户明确说 "存到 …" 时的
`save_draft_to_file`。draft 本身一直留在 session 内存里。
任何带 *state-mutating* 注记的工具（`train_style`、`build_fingerprint`、
`update_fingerprint`、`set_config`）调用前 agent 都会先口头确认。
