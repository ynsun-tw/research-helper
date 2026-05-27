# Architecture Design — Research Agent

## 1. 系统愿景

Research Agent 是一个本地优先的多智能体 CLI 工具，定位为研究者的"资深合作研究员"。系统的核心差异化在于：

- **批判性对话**：内置的 Critic Agent 强制产出反对意见，避免 AI 迎合偏差
- **长期记忆**：跨会话的三层记忆架构，构建用户的研究知识图谱
- **风格模仿**：Scribe Agent 从用户已发表论文中提取写作指纹；Illustrator 配套
  生成 TikZ / matplotlib / 概念图
- **自我查重**：Scribe 输出在保存前可自动与用户既有发表语料比对
- **本地隐私**：所有数据存储在 `~/.research-agent/`，无强制网络同步；
  ChromaDB 的句向量模型（166 MB ONNX，all-MiniLM-L6-v2）首次启动后亦完全本地运行

> 形态约束（0.6+）：**chat-first**。所有能力都在 REPL 内通过 slash 命令或
> 自然语言触发；CLI 子命令保留作为脚本化入口（CI / cron / shell pipelines）。
> 详见 ADR-006、ADR-008。

---

## 2. 有界上下文（Bounded Contexts）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Research Agent System                          │
│                                                                         │
│  ┌──────────────────────────  Interaction layer  ──────────────────┐    │
│  │   Chat Router · slash dispatch · LLM tool-call loop (27 tools)  │    │
│  │   prompt_toolkit REPL · streaming output · token budget          │    │
│  └──────┬─────────────────────────┬────────────────────────┬───────┘    │
│         │                         │                        │            │
│  ┌──────▼──────────┐    ┌─────────▼────────┐    ┌──────────▼────────┐  │
│  │  Agent Context  │    │  Memory Context  │    │  Paper Context    │  │
│  │ - Orchestrator  │◄──►│ - WorkingMemory  │◄──►│ - Loader/Fetcher  │  │
│  │ - Analyst       │    │ - MemoryKeeper   │    │ - PyMuPDF parser  │  │
│  │ - Critic        │    │ - MetaMemory     │    │ - PaperResolver   │  │
│  │ - Searcher      │    │ - SQLite (8 tbl) │    │ - PaperContext    │  │
│  │ - Scribe        │    │ - ChromaDB (ideas│    │   (PDF cache)     │  │
│  │ - Illustrator   │    │   + discussions) │    └───────────────────┘  │
│  │ - WritingPipe   │    └──────────────────┘                           │
│  └──────┬──────────┘                                                    │
│         │                                                                │
│  ┌──────▼──────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │  Search Context │    │  Writing Context │    │  Config / Doctor  │  │
│  │ - arXiv API     │    │ - StyleSampler   │    │ - Pydantic config │  │
│  │ - Semantic Sch. │    │ - Fingerprint    │    │ - Health checks   │  │
│  │ - GitHub        │    │ - Plagiarism     │    │ - Migrations      │  │
│  │ - ReadingQueue  │    │ - DraftRevisions │    └───────────────────┘  │
│  └─────────────────┘    └──────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

Interaction layer 是 0.6+ 的核心抽象：所有用户意图都经它路由。CLI 子命令
（`research read` / `research write` / …）共享同一份纯函数实现
（`cli_*.py` + `cli_services.py`），便于脚本化但不再是主入口。

---

## 3. 多智能体架构

### 3.1 Agent 交互图

```
                         ┌─────────────┐
                         │  用户(REPL)  │
                         └──────┬──────┘
                                │ /slash · 自然语言
                    ┌───────────▼───────────────┐
                    │   chat/router.py          │
                    │   - SLASH_COMMANDS 分发   │
                    │   - LLM agent loop        │
                    │     (chat_with_tools_stream)
                    │   - _build_messages       │
                    │     (tool_log 过滤 + cap) │
                    └──┬──────────────────┬─────┘
                       │                  │
                       │       ┌──────────▼─────────────┐
                       │       │ chat/tools.py — 27 个   │
                       │       │ LLM_TOOLS registry      │
                       │       │  ─ search/queue/cites   │
                       │       │  ─ paper/discuss/idea   │
                       │       │  ─ draft/figure/save    │
                       │       │  ─ revise/plagiarism    │
                       │       │  ─ style/fingerprint    │
                       │       │  ─ config/doctor/insights│
                       │       └──────────┬─────────────┘
                       │                  │ 调用对应 agent/service
                       │   ┌──────────────▼──────────────┐
                       │   │  Orchestrator 路由 / 聚合     │
                       │   │  WritingPipeline 串联         │
                       │   └─┬────┬────┬────┬────┬────┬──┘
                       │     │    │    │    │    │    │
                       │  ┌──▼──┐ │ ┌──▼──┐ │ ┌──▼──┐ │
                       │  │Analy│ │ │Crit │ │ │Scrb │ │
                       │  └─────┘ │ └─────┘ │ └─────┘ │
                       │       ┌──▼─┐    ┌──▼─┐  ┌───▼────┐
                       │       │Srch│    │Illu│  │Memory  │
                       │       │er  │    │stra│  │Keeper +│
                       │       │    │    │tor │  │MetaMem │
                       │       └────┘    └────┘  └────────┘
                       │
                  ChatSession (state)
                  ├─ anchor_paper
                  ├─ current_idea_id / debate
                  ├─ recent_drafts / figures / revisions
                  ├─ WorkingMemory（会话 transcript）
                  └─ prompt_session（prompt_toolkit）
```

### 3.2 Agent 职责矩阵

| Agent | 核心职责 | 输入 | 输出 | 约束 |
|-------|---------|------|------|------|
| Orchestrator | 路由、聚合、并行调度 | 用户意图 | 结构化任务包 | 不做价值判断 |
| Analyst | 深度理解、洞察发现 | 论文 / Idea | 贡献分析、关联 | 区分声明 vs 实证 |
| Critic | 反驳、风险评估 | 论文 / Idea / 草稿 | 反对点、支持度评分 | 不允许打 10 分；分 < 7 须给理由 |
| Searcher | 多源文献搜索 + 查询精化 | 查询意图 / 讨论上下文 | 相关度评分的论文列表 / 下一步查询建议 | 承认搜索局限 |
| Scribe | 风格模仿写作 + 自动修订 | 章节需求 / 旧草稿 | N 个变体；revision | 受 fingerprint 约束；只写不存盘 |
| Illustrator | 配图代码生成 | figure_type + 描述 + 可选数据 | TikZ / matplotlib / 文生图 prompt | 可选 ``verify`` 真跑 matplotlib 子进程 |
| WritingPipeline | 把 Analyst+Critic 串到 Scribe 修订上 | 旧草稿 + section | ReviewedDraft（issues + 新文） | 全自动；不假装人类判断 |
| MemoryKeeper | 跨会话语义召回 | 查询 / 触发词 | 关联消息、idea、paper | 异步维护向量；不主动发言 |
| MetaMemory | 活动 rollup（阅读 / 讨论 / idea 趋势） | 时间窗 | InsightsReport（Markdown） | 只读；从 SQLite 聚合 |

### 3.3 协作模式

```python
# 并行分析模式（读新论文）
parallel: [Analyst, Critic] → Orchestrator 聚合 → 持久化

# 辩论模式（讨论 Idea）
sequential: Analyst(建树) ↔ Critic(破坏)  → DebateHistory 追加

# 搜索流水线
Searcher.suggest_refinement(memory) → search_arxiv → 用户审核 →
queue_add / load_paper

# 写作流水线（M4，0.5+）
Scribe.draft_section(context, fingerprint) → N variants（cached）
  ↳ check_self_plagiarism（与用户既往发表语料比对，无 LLM 调用）
  ↳ revise_draft：WritingPipeline = parallel:[Analyst, Critic]
                  → Scribe.rewrite(issues + 原文) → ReviewedDraft（cached）
  ↳ save_draft_to_file（唯一会落盘的工具）

# 配图流水线（M5）
Illustrator.generate(figure_type, description, data?) → N variants
  ↳ optional verify: subprocess(matplotlib, 30s timeout)

# 引文导航
get_citations / get_references → Semantic Scholar
```

---

## 4. 数据架构

### 4.1 存储层设计

```
~/.research-agent/
├── config.yaml              # API key / model / base_url / language / app_*
├── memory.db                # SQLite（schema 见 4.2）
├── chroma/                  # ChromaDB PersistentClient
│   ├── ideas/               # Idea 标题+描述向量
│   └── discussions/         # REPL 消息向量（recall_history）
├── papers/                  # PDF 本地缓存（按 arXiv id / hash）
├── style/
│   ├── samples/             # 训练样本副本（PDF）— 可选保留
│   ├── fingerprint.json     # 当前写作风格指纹
│   └── history/             # 归档的旧 fingerprint
└── repl_history             # prompt_toolkit 跨会话命令历史（0.6+）

# 此外，ChromaDB 首次启动时会下载到
# ~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/  (≈166 MB ONNX 模型，本地推理)
```

### 4.1.1 `config.yaml` 字段

| 字段 | 默认 | 说明 |
|------|------|------|
| `api_key` | — | OpenRouter API key（`sk-or-...`） |
| `model` | `deepseek/deepseek-chat` | OpenRouter 模型 slug |
| `base_url` | `https://openrouter.ai/api/v1` | LLM API 基址 |
| `language` | `en` | Agent 回复语言：`en`（英文）或 `zh`（简体中文） |
| `app_title` / `app_url` | 见代码默认值 | OpenRouter 推荐请求头 |
| `data_dir` | `~/.research-agent` | 数据根目录 |

`language` 通过 system prompt 后缀注入，影响 `read` / `discuss` 的 Analyst 与 Critic 输出语言（不改变 PDF 原文）。

### 4.2 SQLite 核心 Schema

完整定义见 `storage/database.py:SCHEMA`。当前 8 张表：

| 表 | 模块 | 用途 |
|---|---|---|
| `papers` | `storage/database.py:PaperRepository` | 论文元数据 + Analyst/Critic 笔记 |
| `ideas` | `storage/ideas.py:IdeaRepository` | Idea 生命周期 + 评分历史 |
| `discussions` | `storage/discussions.py:DiscussionRepository` | REPL 对话历史；含 `metadata.kind=tool_log` 标记 |
| `search_queries` | `storage/searches.py:SearchRepository` | 用户 /search 查询记录 |
| `search_results` | 同上 | 每次查询命中的论文 hit 列表 |
| `reading_queue` | `storage/reading_queue.py:ReadingQueueRepository` | "稍后读"队列（pending/done/skipped） |
| `style_samples` | `storage/database.py:StyleSampleRepository` | Scribe 训练用的段落级样本 |
| `draft_revisions` | `storage/draft_revisions.py:DraftRevisionRepository` | (original, revised) 配对 + 用户接受/拒绝信号 |

关键 schema（删节，保留语义骨架）：

```sql
CREATE TABLE papers (
    id TEXT PRIMARY KEY,             -- arxiv:2301.12345 | doi:* | local:hash
    title TEXT NOT NULL,
    abstract TEXT,
    sections TEXT,                   -- JSON: [(title, body), ...]
    full_text TEXT,
    analyst_notes TEXT,              -- JSON：Pydantic AnalysisPayload
    critic_notes TEXT,               -- JSON：Pydantic CritiquePayload
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ideas (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT,                     -- active|shelved|waiting|experimenting|abandoned|completed
    critic_score REAL,               -- 1-9（不允许 10；见 ADR）
    score_history TEXT,              -- JSON: [{ts, score, reason}, ...]
    related_papers TEXT,
    ...
);

CREATE TABLE discussions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,              -- user|assistant|system|tool
    content TEXT NOT NULL,
    metadata TEXT,                   -- JSON: {kind?: "tool_log", idea_id?, ...}
    idea_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE style_samples (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    section_title TEXT,
    paragraph TEXT NOT NULL,
    char_count, word_count, sentence_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE draft_revisions (
    id TEXT PRIMARY KEY,
    section TEXT NOT NULL,
    original_text TEXT NOT NULL,
    revised_text TEXT NOT NULL,
    selected_issues, selected_suggestions     TEXT,   -- 用户接受的反馈
    rejected_issues, rejected_suggestions     TEXT,   -- 用户拒绝的反馈
    interactive INTEGER NOT NULL DEFAULT 0,           -- 1 = 来自交互式 review
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`draft_revisions` 是 Scribe 持续学习的语料：未来 M6+ 的 fingerprint 训练会
读 `selected_*` 字段反向调整风格权重。

### 4.3 三层记忆模型

```
工作记忆 (Working Memory)               memory/working_memory.py
  ├─ 范围：当前会话
  ├─ 存储：进程内 Python 对象
  ├─ 截断：~4 chars/token 启发式估计，按 max_context_tokens 反向裁剪
  └─ 落盘：session.close() 时 persist → discussions 表 + Chroma 索引

研究记忆 (Research Memory)             agents/memory_keeper.py +
  ├─ 范围：跨会话、所有 idea / paper      storage/* + ChromaDB
  ├─ 存储：SQLite + 两个 Chroma collection（ideas / discussions）
  └─ 用途：recall_history 工具的语义召回、idea 关联、阅读队列

元认知记忆 (Meta Memory)               agents/meta_memory.py
  ├─ 范围：长期（since_days 可配）
  ├─ 存储：从 SQLite 聚合，不单独建表
  └─ 用途：research_insights 工具——阅读量、idea 状态分布、讨论强度趋势
```

---

## 5. 技术栈决策

| 层级 | 选择 | 备选 | 决策理由 |
|------|------|------|---------|
| 语言 | Python 3.9+ | Go | AI 生态最完善；3.9 兼容（`eval_type_backport` 后向支持 PEP 604） |
| CLI | Typer | Click, argparse | 基于类型注解，代码即文档 |
| REPL | prompt_toolkit ≥ 3.0 | Rich `console.input` | 跨平台行编辑、历史、Tab 补全、Ctrl+R 反搜（ADR-009，0.6） |
| 渲染 | Rich ≥ 13.7 | 自实现 | Markdown、表格、live streaming 兼容 |
| LLM | OpenRouter（默认） | DeepSeek 直连 / OpenAI | OpenAI SDK 通用接口；流式 + tool calling；模型可热换 |
| LLM-output 解析 | Pydantic v2 schemas | 自写 JSON 提取 | 见 ADR-007；运行时类型校验，错误信息可用 |
| 向量库 | ChromaDB（含 ONNX `all-MiniLM-L6-v2`） | Qdrant, FAISS | 零配置，本地推理；首启动下载 166 MB 后无网依赖 |
| 关系库 | SQLite + FTS5（按需） | PostgreSQL | 零依赖，单文件，支持全文搜索 |
| PDF | PyMuPDF (fitz) | pdfplumber, pypdf | 性能最佳；段落保真度足够 Scribe 训练用 |
| 配置 | Pydantic Settings + YAML | .env only | 类型安全 + 可读性 |
| 打包 | pipx（推荐） / pip | — | Python CLI 工具标准分发；详见 README |
| 发布 | GitHub Actions → PyPI | 手工 twine | TestPyPI + PyPI 两条流水线，手动 approval gate |

> Textual TUI（M5 中规划过）已下线：实测 chat REPL + slash + 自然语言能覆盖
> 95% 场景；多维护一份 UI 不划算。代码里 `ui/` 目录现在只剩 Rich 渲染辅助。

---

## 6. CLI 形态（chat-first，0.6+）

`research` 默认进入 prompt_toolkit REPL。CLI 子命令（`research read` /
`research write` / `research figure` / `research check` / `research style` /
`research doctor` / `research config`）继续工作，但定位是**脚本化入口**——
跑在 CI / cron / pipeline 里。日常交互都该走 REPL：

```
research
You> /search transformer attention            # 显式 slash
You> /read arxiv:1706.03762                   # 显式 slash
You> 帮我比较 A 和 B 哪个更适合稀疏注意力        # 自然语言 → LLM 选工具
You(arxiv:1706.03762)> /draft intro 300字     # 状态感知 prompt
You(arxiv:1706.03762 | idea:9a1f0..)> /exit
```

REPL 键位 / 流式输出 / 状态 prompt 见 README "REPL keys" 一节。
所有 27 个 LLM tool 列表见 `chat/tools.py:LLM_TOOLS`，16 个 slash 命令
列表见 `SLASH_COMMANDS`（`/exit`、`?` 在 router 层直接处理）。

### 6.1 Chat Loop 数据流（0.6+）

```
User input
  ├─ "?" / "/?" / "help"  → 等价 /help
  ├─ "/cmd args"          → SLASH_COMMANDS[cmd].handler(session, args)
  └─ 自然语言             → _chat_with_llm(session, text)
                            │
                            ▼
                _build_messages(session)
                  ├─ system: CHAT_SYSTEM_PROMPT（精简版，仅路由规则）
                  ├─ system: anchor paper hint（若有）
                  └─ history: 最近 8 条 memory 消息
                             过滤 metadata.kind == "tool_log"
                            │
                            ▼
                LLMProvider.chat_with_tools_stream(
                    messages, tools=llm_tool_schemas(),
                    on_content_delta=stream_to_console
                )
                  ├─ 无 tool_calls → 流式打印 + memory.append("assistant")
                  └─ 有 tool_calls →
                       for tc in response.tool_calls:
                           result = LLM_TOOLS[tc.name].executor(session, args)
                           messages.append(role="tool", _cap_tool_result(result))
                           # 同时落入 WorkingMemory，但 tag tool_log
                           memory.append("system", result[:400],
                                         metadata={"kind": "tool_log"})
                       重新进入 stream 循环（≤ MAX_TOOL_ITERATIONS=6）
                            │
                            ▼
                每轮结束打印 [~N in → ~M out tokens · K rounds]

ChatSession 状态：
  anchor_paper · current_idea_id · debate · WorkingMemory
  recent_drafts / recent_figures / recent_revisions     # 写作流水线缓存
  prompt_session                                         # prompt_toolkit
```

约束：
- **Tool result cap**（`MAX_TOOL_RESULT_CHARS = 8000`）：单个 `role=tool`
  payload 超长截断 + 提示 LLM 重新调用。
- **Tool log filter**：tool 结果只在当前 agent loop 喂给 LLM；持久化时打
  `kind=tool_log`，下一轮 `_build_messages` 跳过。详见 0.6.1 commit。
- **history_limit = 8**（曾经是 20）：长期上下文走 `recall_history`。

### 6.2 仓库布局（实际状态）

```
research-bot/
├── planning/                # 规划文档（PLAN, milestones, notes, architecture）
├── examples/                # 中文 + 自然语言场景脚本
├── src/research_agent/
│   ├── __init__.py
│   ├── cli.py               # Typer 入口；默认 REPL，子命令为脚本入口
│   ├── cli_services.py      # run_read / run_discuss / run_followup 共享纯函数
│   ├── cli_ideas.py         # run_ideas_list / show / update
│   ├── cli_write.py         # Scribe + 写作上下文（WritingContext, WriteResult）
│   ├── cli_figure.py        # Illustrator
│   ├── cli_review.py        # WritingPipeline = Analyst+Critic → Scribe rewrite
│   ├── cli_check.py         # check_self_plagiarism
│   ├── cli_style.py         # 训练 / fingerprint / 历史
│   ├── cli_doctor.py        # 环境健康检查（无网络、无 LLM）
│   ├── config.py            # Pydantic Settings 配置
│   ├── chat/
│   │   ├── session.py       # ChatSession + WritingPipeline 缓存
│   │   ├── router.py        # slash 分发 + agent loop + 流式 + token telemetry
│   │   ├── tools.py         # 27 LLM_TOOLS + 16 SLASH_COMMANDS
│   │   └── prompt_ui.py     # prompt_toolkit factory + 状态 prompt
│   ├── core/
│   │   ├── llm.py           # LLMProvider, ChatMessage, ChatResponse, ToolCall
│   │   │                    # chat_with_tools_stream(on_content_delta=...)
│   │   ├── language.py      # 回复语言 en/zh
│   │   ├── paper.py · paper_resolver.py · paper_context.py · loader.py
│   │   ├── debate_prompts.py
│   │   └── idea.py
│   ├── agents/
│   │   ├── base.py          # Agent 基类
│   │   ├── schemas.py       # Pydantic v2 全部 LLM-output 模型 + parse_model
│   │   ├── prompts.py       # YAML 模板加载
│   │   ├── orchestrator.py
│   │   ├── analyst.py · critic.py · debate.py
│   │   ├── searcher.py      # arXiv / SS / GitHub + suggest_refinement
│   │   ├── scribe.py        # 写作 + 修订
│   │   ├── illustrator.py   # TikZ / matplotlib / 文生图 prompt
│   │   ├── writing_pipeline.py  # 自动 review-and-rewrite
│   │   ├── memory_keeper.py · meta_memory.py
│   ├── memory/working_memory.py
│   ├── storage/
│   │   ├── database.py             # 8-table schema + 迁移
│   │   ├── discussions.py · ideas.py · searches.py
│   │   ├── reading_queue.py · draft_revisions.py
│   │   ├── vector_store.py         # IdeaVectorStore
│   │   └── discussion_vectors.py   # DiscussionVectorStore
│   ├── search/
│   │   ├── arxiv.py · arxiv_search.py
│   │   ├── semantic_scholar.py     # 引文图（forward/backward refs）
│   │   └── github.py               # repo + papers-with-code
│   ├── style/
│   │   ├── samples.py · extractor.py · analyzer.py
│   │   ├── filters.py · fingerprint.py
│   │   └── plagiarism.py           # 段落级 cos-sim 比对
│   ├── parsers/pdf.py
│   ├── prompts/                    # YAML system prompts（analyst/critic/scribe/…）
│   └── ui/formatting.py
├── tests/                 unit / integration / e2e（724 passing）
├── .github/workflows/     ci / release-testpypi / release-pypi
├── pyproject.toml
└── README.md
```

---

## 7. 关键数据流

### 7.1 论文分析流（`/read` / `load_paper`）

```
用户输入 → chat.router → cmd_read / exec_load_paper → core/loader.py
  │
  ├─ PaperResolver: arxiv_id | title 模糊匹配 | 本地 PDF 路径
  ├─ Fetcher.download → PDF 落到 ~/.research-agent/papers/
  └─ PDFParser.extract → Paper(sections, abstract, full_text)
        │
        ▼
  Orchestrator.analyze_paper(paper)
        │  并行：
        ├─ Analyst.analyze(paper)   → AnalysisPayload   (Pydantic)
        └─ Critic.critique(paper)   → CritiquePayload   (Pydantic)
        │
        ▼
  PaperRepository.save(paper, analyst_notes, critic_notes)
  ChatSession.anchor_paper = paper
  WorkingMemory.append("user"/"assistant")
```

### 7.2 Idea 讨论流（`/discuss` / `discuss_idea`）

```
要求 anchor_paper ≠ None；否则提示先 load_paper

Orchestrator.debate_round(paper, idea, debate_history, memory)
  │
  ├─ MemoryKeeper.recall(idea_seed)  → 跨会话相关讨论 / idea
  ├─ 辩论轮次：
  │   ├─ Analyst.support(idea, paper, transcript)  → IdeaSupportPayload
  │   ├─ Critic.oppose(idea, paper, transcript)    → CritiquePayload
  │   └─ DebateHistory.add_round(...)
  └─ followup_turn(...)  可重复直至用户 /save 或 /exit
        │
        ▼
  save_current_idea → IdeaRepository.create(...)
                   → IdeaVectorStore.upsert(idea)
```

### 7.3 写作流水线（M4，0.5+）

```
draft_section(section, context?, target_words?, versions?, check_against?)
  │
  ├─ Scribe.draft_section
  │     ├─ 读 fingerprint.json（若有）→ 风格特征 prompt 注入
  │     ├─ 读 recall_history(context) → 相关历史 idea / paper
  │     └─ 并行生成 N 个 DraftPayload（Pydantic）
  └─ ChatSession.recent_drafts[section] += drafts   # 不落盘

revise_draft(target=latest:<section>[:<ver>], section?, target_words?)
  │
  ├─ Resolver: 解析 latest ref 或 path → 原文
  ├─ WritingPipeline = Analyst.review + Critic.review（并行）
  │   → WritingReviewPayload（issues, suggestions, support_score）
  ├─ Scribe.rewrite(原文, issues, target_words) → ReviewedDraft
  └─ ChatSession.recent_revisions[section] = ReviewedDraft

check_self_plagiarism(target=path|latest, threshold?)
  │
  ├─ style/plagiarism.py：段落级向量相似度
  ├─ 数据源：style_samples 表里的用户既有发表段落
  └─ 输出：ParagraphMatch 列表（source_id, similarity, draft_paragraph）

save_draft_to_file(path, kind=section|figure|revision, section?, version?)
  │
  └─ 唯一会写盘的工具。draft_revisions 表只在交互 review 时落（不在 stream 模式）
```

### 7.4 搜索与引文导航

```
search_arxiv(query, max_results?, mode?)
  │
  ├─ Searcher 拼接 prompt（mode = theoretical|applied|group:<author>）
  ├─ arXiv API 拉候选 → Pydantic SearcherScoresPayload 评分
  └─ SearchRepository.persist  （search_queries + search_results）

suggest_search_refinement
  │
  ├─ 读最近若干轮 working memory
  └─ Searcher → SearcherRefinementPayload（next query + reason + 可选 mode）

get_citations / get_references
  │
  └─ Semantic Scholar API（forward / backward refs，可缓存）

queue_add / queue_list / queue_next
  │
  └─ reading_queue 表，FIFO；queue_next 不改状态，由 load_paper 之后再更新
```

### 7.5 风格训练与指纹

```
train_style(sources | directory, append?)
  │
  ├─ style/extractor.py：PDF → 段落级样本
  ├─ style/filters.py：去引文 / 去公式 / 长度过滤
  └─ StyleSampleRepository.save → style_samples 表

build_fingerprint / update_fingerprint
  │
  ├─ style/analyzer.py：统计指标（句长、被动比、依存深度…）+ 关键短语
  ├─ 写入 ~/.research-agent/style/fingerprint.json
  └─ update_fingerprint 同时归档旧版本到 style/history/
```

---

## 8. Token 预算与上下文管理（0.6.1+）

LLM 调用是这套系统里唯一的真实美元成本，所以 chat loop 的设计明确按
"每轮最小固定开销 + 上限化可变开销"做了优化。当前预算曲线：

| 组成 | 旧 (≤0.6.0) | 新 (0.6.1+) | 备注 |
|---|---|---|---|
| System prompt | ~2616 tok | ~629 tok | 删掉与 schema description 重复的部分，只留 chaining rules |
| Tool schemas（27 个） | ~4440 tok | ~3824 tok | top-5 fattest 全部精简 |
| **每轮固定 overhead** | **~7056** | **~4453** | -37% |
| Tool result（单条） | 无上限 | ≤ 2000 tok | `MAX_TOOL_RESULT_CHARS = 8000` |
| History 窗口 | last 20 msg | last 8 msg | 长记忆走 recall_history |
| Tool 结果回灌 | 永久滚动 | 当前 loop 仅一次 | metadata `kind=tool_log` 在下轮被过滤 |

### 8.1 设计原则

- **静态信息只发一次（同一 round 之内）**：系统提示词不嵌入 tool 描述（OpenAI
  schema 字段已经有了，重复 = 双倍 token）。
- **Tool 结果是一次性的**：调用、用完、写日志；下一轮起对 LLM 不可见
  （但 `/history` 与 audit 仍可见）。需要再用的让 LLM 重新调一次工具。
- **预算可观测**：每轮末尾打 `[~N in → ~M out tokens · K rounds]`，用户能
  直观看到一次"问 → 答"花了多少（基于 4 chars/token 启发式，无 tiktoken 依赖）。

### 8.2 未来工作（候选 P1）

- **Prompt caching**（Anthropic / OpenRouter 的 `cache_control` 标记）：
  把 system prompt + tool schemas 标 ephemeral cache，相同会话内第二轮起
  90% 折扣。需要兼容不同 provider 的 header 格式。
- **真分词器**：tiktoken / cl100k 替代估算，对中文 / 多语种更准。代价是
  +~800 KB 依赖且 OpenRouter 模型分词器各异，目前不划算。
- **响应级缓存**：相同 user prompt + 相同 tool 状态 → 跳过 LLM 直接复读。
  路径正交，未排。

---

## 9. 安全与隐私

- 所有数据本地存储；网络出口只有 LLM API + arXiv / Semantic Scholar / GitHub
  公共 API。
- API Key 在 `~/.research-agent/config.yaml`，建议 `chmod 600`。
- ChromaDB 默认 ONNX 模型（166 MB）首次启动后离线运行；语义召回、查重、
  写作流水线都不再额外联网。
- 论文 PDF / 风格样本 / 草稿都在 `~/.research-agent/` 内，可随时整目录
  备份或删除。
- prompt_toolkit REPL 历史保存在 `~/.research-agent/repl_history`，
  纯文本——不要把敏感 token 直接输入。
- 发布流水线（GitHub Actions）禁止 `git tag` 在 twine upload 之前生成，
  避免出现"打了 tag 但没真发布"的悬挂状态（见 `.github/workflows/
  release-pypi.yml` 内的注释）。
