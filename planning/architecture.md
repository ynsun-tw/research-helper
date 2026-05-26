# Architecture Design — Research Agent

## 1. 系统愿景

Research Agent 是一个本地优先的多智能体 CLI 工具，定位为研究者的"资深合作研究员"。系统的核心差异化在于：

- **批判性对话**：内置的 Critic Agent 强制产出反对意见，避免 AI 迎合偏差
- **长期记忆**：跨会话的三层记忆架构，构建用户的研究知识图谱
- **风格模仿**：Scribe Agent 从用户已发表论文中提取写作指纹
- **本地隐私**：所有数据存储在 `~/.research-agent/`，无强制云依赖

---

## 2. 有界上下文（Bounded Contexts）

```
┌──────────────────────────────────────────────────────────────────┐
│                       Research Agent System                       │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Agent Context  │    │  Memory Context │    │ Paper       │  │
│  │                 │    │                 │    │ Context     │  │
│  │ - Orchestrator  │◄──►│ - WorkingMem    │◄──►│             │  │
│  │ - Analyst       │    │ - ResearchMem   │    │ - Parser    │  │
│  │ - Critic        │    │ - MetaMem       │    │ - Fetcher   │  │
│  │ - Searcher      │    │ - VectorStore   │    │ - Cache     │  │
│  │ - Scribe        │    │ - SQLite DB     │    │             │  │
│  │ - MemoryKeeper  │    └─────────────────┘    └─────────────┘  │
│  └────────┬────────┘                                            │
│           │                                                      │
│  ┌────────▼────────┐    ┌─────────────────┐                    │
│  │  Search Context │    │   UI Context    │                    │
│  │                 │    │                 │                    │
│  │ - arXiv API     │    │ - CLI (Typer)   │                    │
│  │ - Semantic Sch. │    │ - TUI (Textual) │                    │
│  │ - Google Scholar│    │ - Output Format │                    │
│  │ - GitHub/PWC    │    └─────────────────┘                    │
│  └─────────────────┘                                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 多智能体架构

### 3.1 Agent 交互图

```
                         ┌─────────────┐
                         │   用户(REPL)  │
                         └──────┬───────┘
                                │ /slash 或 自然语言
                    ┌───────────▼───────────┐
                    │   Chat Router         │
                    │   - slash 分发         │
                    │   - LLM agent loop    │
                    │     (tool_calls)      │
                    │   - ChatSession 状态   │
                    └──┬───────────┬────────┘
                       │           │ 工具调用
                       │  ┌────────▼─────────┐
                       │  │  Tool Registry   │
                       │  │  search_arxiv    │
                       │  │  load_paper      │
                       │  │  discuss_idea    │
                       │  │  save_current_idea│
                       │  │  list_ideas      │
                       │  └────────┬─────────┘
                                   │
                    ┌──────────────▼─────────┐
                    │   Orchestrator        │
                    │   - 任务路由           │
                    │   - 输出聚合           │
                    │   - 冲突标注           │
                    └──┬────┬────┬────┬────┘
                       │    │    │    │
              ┌────────┘    │    │    └────────┐
              │             │    │             │
         ┌────▼────┐   ┌───▼────▼───┐   ┌────▼────┐
         │Analyst  │   │   Critic   │   │Searcher │
         │         │◄─►│            │   │         │
         │好奇心驱动│   │ 魔鬼代言人  │   │文献雷达  │
         └────┬────┘   └─────┬──────┘   └────┬────┘
              │              │               │
              └──────┬───────┘               │
                     │                       │
              ┌──────▼──────┐         ┌─────▼──────┐
              │    Scribe   │         │MemoryKeeper│
              │             │         │            │
              │ 写作风格影子  │         │  外脑/记忆  │
              └─────────────┘         └────────────┘
```

### 3.2 Agent 职责矩阵

| Agent | 核心职责 | 输入 | 输出 | 约束 |
|-------|---------|------|------|------|
| Orchestrator | 路由、聚合、协调 | 用户意图 | 结构化任务包 | 不做价值判断 |
| Analyst | 深度理解、洞察发现 | 论文/Idea | 贡献分析、关联 | 区分声明 vs 实证 |
| Critic | 反驳、风险评估 | 论文/Idea | 1+反对点、支持度评分 | 不允许打10分 |
| Searcher | 多源文献搜索 | 查询意图 | 相关度评分的论文列表 | 承认搜索局限 |
| Scribe | 风格模仿写作 | 章节需求 | 多版本草稿 | 持续学习修改 |
| MemoryKeeper | 记忆检索、关联发现 | 查询/触发词 | 历史记录、模式 | 不主动发言 |

### 3.3 协作模式

```python
# 并行分析模式 (读新论文)
parallel: [Analyst, Critic] → Orchestrator 聚合

# 辩论模式 (讨论 Idea)
sequential: Analyst(建树) ↔ Critic(破坏) → 多轮迭代

# 搜索流水线
Searcher → 用户审核 → parallel: [Analyst, Critic] 批量读

# 写作审查流水线
Scribe(生成) → parallel: [Analyst, Critic](审查) → Scribe(修改)
```

---

## 4. 数据架构

### 4.1 存储层设计

```
~/.research-agent/
├── config.yaml              # 配置文件（API key、model、language、OpenRouter 等）
├── memory.db                # SQLite 主数据库
│   ├── papers               # 论文元数据 + 分析结果
│   ├── ideas                # Idea 生命周期记录
│   ├── discussions          # 对话历史
│   ├── style_samples        # 用户写作样本
│   └── search_history       # 搜索历史
├── chroma/                  # ChromaDB 向量存储
│   ├── papers_embeddings    # 论文摘要/全文向量
│   ├── ideas_embeddings     # Idea 向量
│   └── discussions_embeddings
├── cache/
│   ├── papers/              # PDF 本地缓存（按 arXiv ID）
│   └── search/              # 搜索结果缓存（TTL: 24h）
├── style/
│   └── fingerprint.json     # 写作风格指纹
└── exports/                 # 导出报告
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

```sql
-- 论文表
CREATE TABLE papers (
    id TEXT PRIMARY KEY,           -- arxiv:2301.12345 | doi:xxx | local:hash
    title TEXT NOT NULL,
    authors TEXT,                  -- JSON array
    abstract TEXT,
    year INTEGER,
    venue TEXT,
    pdf_path TEXT,                 -- 本地缓存路径
    analyst_notes TEXT,            -- JSON: Analyst 分析结果
    critic_notes TEXT,             -- JSON: Critic 批判结果
    relevance_score REAL,
    read_at TIMESTAMP,
    tags TEXT,                     -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Idea 表
CREATE TABLE ideas (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT,                   -- active|shelved|waiting|experimenting|abandoned|completed
    analyst_score REAL,
    critic_score REAL,             -- 1-10
    critic_objections TEXT,        -- JSON array
    related_papers TEXT,           -- JSON array of paper IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 对话历史表
CREATE TABLE discussions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,            -- user|orchestrator|analyst|critic|searcher|scribe|memory
    content TEXT NOT NULL,
    metadata TEXT,                 -- JSON: 关联的 paper_id/idea_id 等
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.3 三层记忆模型

```
工作记忆 (Working Memory)
  ├── 范围：当前会话
  ├── 存储：进程内 Python 对象
  └── 用途：上下文窗口管理、会话状态

研究记忆 (Research Memory)
  ├── 范围：跨会话、所有项目
  ├── 存储：SQLite + ChromaDB
  └── 用途：论文评价、Idea 记录、讨论历史

元认知记忆 (Meta Memory)
  ├── 范围：长期
  ├── 存储：SQLite aggregation views
  └── 用途：用户决策模式、研究兴趣趋势
```

---

## 5. 技术栈决策

| 层级 | 选择 | 备选 | 决策理由 |
|------|------|------|---------|
| 语言 | Python 3.11+ | Go | AI 生态最完善，科研社区主力语言 |
| CLI | Typer | Click, argparse | 基于类型注解，代码即文档 |
| TUI | Textual | Rich+自定义 | 现代化 TUI 框架，组件丰富 |
| LLM | OpenRouter API | DeepSeek 直连, OpenAI | 默认经 OpenRouter；OpenAI SDK；可换任意兼容模型 |
| 向量库 | ChromaDB | Qdrant, FAISS | 零配置，Python 原生，本地部署 |
| 关系库 | SQLite + FTS5 | PostgreSQL | 零依赖，单文件，支持全文搜索 |
| 图计算 | NetworkX | Neo4j | 引用图规模小（<10万节点），无需图DB |
| PDF | PyMuPDF | pdfplumber, pypdf | 性能最佳，提取质量高 |
| 配置 | Pydantic Settings + YAML | .env only | 类型安全 + 可读性 |
| 打包 | pipx | pip global | Python CLI 工具标准分发方式 |

---

## 6. CLI 形态（M2.5 起：单入口 REPL）

`research` 命令默认进入对话式 shell。只保留 `research config` 作为非交互子命令；
原先的 `read` / `discuss` / `ideas` 改为 REPL 内的 slash 命令，并通过 LLM
function/tool calling 暴露给自然语言输入（见 ADR-006）。

```
research                          # 进入 REPL
You> /search transformer          # 显式工具：arXiv 搜索
You> /read arxiv:1706.03762       # 显式工具：下载 + Analyst/Critic
You> /discuss apply to molecules  # 显式工具：辩论（基于已锚定论文）
You> 帮我比较 A 和 B 哪个更好用      # 自然语言 → LLM 自主决定调用哪些工具
You> /exit
```

### 6.1 Chat Loop 数据流

```
User input
  ├─ "/cmd args"    → SLASH_COMMANDS[cmd](session, args)
  └─ 自然语言        → LLMProvider.chat_with_tools(messages, tools=llm_tool_schemas())
                       ├─ 返回 content（无 tool_calls）→ 渲染给用户
                       └─ 返回 tool_calls → 逐个执行 LLM_TOOLS[name].executor(session, args)
                          → 把结果作为 role="tool" 消息回灌 → 继续 chat_with_tools
                          → 受 MAX_TOOL_ITERATIONS=6 限制
ChatSession 持久状态：anchor_paper / debate_history / current_idea_id / WorkingMemory
持久化：退出时 WorkingMemory.persist → DiscussionRepository（含 idea_id 关联）
```

### 6.2 仓库布局

```
research-bot/
├── planning/                    # 规划文档（本目录）
├── src/
│   └── research_agent/
│       ├── __init__.py
│       ├── cli.py               # Typer 入口：默认 REPL + research config
│       ├── chat/                # 对话 shell（M2.5）
│       │   ├── __init__.py
│       │   ├── session.py       # ChatSession（状态与后端句柄）
│       │   ├── router.py        # slash 分发 + LLM agent loop
│       │   └── tools.py         # SLASH_COMMANDS + LLM_TOOLS 注册
│       ├── cli_services.py      # run_read / run_discuss 等可复用纯函数
│       ├── cli_ideas.py         # run_ideas_list/show/update 可复用纯函数
│       ├── config.py            # Pydantic Settings 配置
│       ├── core/
│       │   ├── llm.py           # LLMProvider, ChatMessage, ToolCall, ChatResponse
│       │   ├── language.py      # 回复语言 en/zh 与 prompt 注入
│       │   ├── paper.py         # 论文实体
│       │   ├── paper_resolver.py# 搜索 + 加载 + 交互式选择
│       │   ├── debate_prompts.py# 辩论用户提示模板
│       │   └── idea.py          # Idea 实体
│       ├── agents/
│       │   ├── base.py          # Agent 基类（支持 prompt_stem 切换）
│       │   ├── orchestrator.py  # analyze_paper / debate_round / followup_turn
│       │   ├── analyst.py
│       │   ├── critic.py
│       │   ├── debate.py        # DebateResult, FollowUpResult, DebateHistory
│       │   ├── memory_keeper.py
│       │   ├── searcher.py      # M3+ 占位
│       │   └── scribe.py        # M4+ 占位
│       ├── memory/
│       │   └── working_memory.py
│       ├── storage/
│       │   ├── database.py      # SQLite + migrations
│       │   ├── discussions.py
│       │   ├── ideas.py
│       │   └── vector_store.py  # ChromaDB（含 keyword fallback）
│       ├── search/
│       │   └── arxiv_search.py
│       ├── parsers/
│       │   └── pdf.py
│       └── ui/
│           └── formatting.py    # Rich 渲染（read_report / debate / paper_header / followup）
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── pyproject.toml
└── README.md
```

---

## 7. 关键数据流

### 7.1 论文分析流

```
用户: research → You> /read arxiv:2301.12345        (或自然语言触发 load_paper 工具)
  │
  ▼
chat.router → tools.cmd_read / exec_load_paper → Orchestrator
  │
  ├── Paper Context
  │     ├── Fetcher.download(arxiv_id)  → PDF
  │     └── PDFParser.extract()        → 结构化文本
  │
  ├── 并行
  │     ├── Analyst.analyze(paper)     → 贡献 + 洞察
  │     └── Critic.critique(paper)     → 缺陷 + 评分
  │
  ├── MemoryKeeper.store(paper, analysis)
  │
  └── Orchestrator.aggregate()         → 格式化输出
```

### 7.2 Idea 讨论流

```
用户: research → /read 锚定论文 → You> /discuss "把 A 和 B 结合起来"
                                (或自然语言触发 discuss_idea 工具)
  │
  ▼
chat.router → tools.cmd_discuss / exec_discuss_idea → Orchestrator
  │
  ├── MemoryKeeper.recall(related_history)  → 历史关联
  │
  ├── 辩论轮次 (最多 N 轮):
  │     ├── Analyst.support(idea)           → 支持论点
  │     ├── Critic.oppose(idea)             → 反对 + 评分
  │     └── Orchestrator.check_convergence()
  │
  └── 输出: 共识 + 分歧 + Memory 提醒
```

---

## 8. 安全与隐私

- 所有数据本地存储，无强制网络同步
- API Key 存储在 `~/.research-agent/config.yaml`，文件权限 `600`
- 可选加密备份（M5 阶段实现）
- 论文 PDF 缓存有大小限制（默认 5GB），超限时按 LRU 淘汰
