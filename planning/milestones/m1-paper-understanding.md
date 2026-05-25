> Status: PENDING
> Index: [../../PLAN.md](../../PLAN.md)

# M1 — MVP: 论文深度理解

**目标用户**: 独立研究者、博士生
**交付物**: 可用的命令行 MVP，能深度分析论文并进行批判性讨论
**时间**: 2 周
**验收标准**:
- `research read <paper>` 能产出 Analyst + Critic 双视角分析
- `research discuss` 能进行基础批判性对话
- `research config` 能配置 API key
- 基础会话内记忆正常工作

---

## Epic E1.1 — 项目基础搭建

> Status: **DONE**
> 价值假设：建立可维护的工程基础，使后续迭代顺畅

### Story S1.1.1 — Python 包结构与工具链初始化

**验收条件**:
- `pyproject.toml` 配置完成，支持 `pip install -e .`
- `research --help` 命令可运行
- `pytest` 能发现并运行测试
- `ruff` + `mypy` 检查通过

**Tasks**:
- [x] T1.1.1.1 初始化 `pyproject.toml`（依赖：typer, textual, pydantic-settings, pyMuPDF, chromadb, openai, pyyaml）
- [x] T1.1.1.2 创建 `src/research_agent/` 包结构（参见 architecture.md §6）
- [x] T1.1.1.3 配置 ruff、mypy、pytest
- [x] T1.1.1.4 创建 `cli.py` Typer 入口，注册顶层命令占位符

### Story S1.1.2 — 配置系统

**验收条件**:
- `research config set api_key sk-xxx` 保存到 `~/.research-agent/config.yaml`
- `research config show` 展示当前配置（脱敏显示 API key）
- 配置文件权限自动设为 `600`
- 缺少必要配置时给出友好错误提示

**Tasks**:
- [x] T1.1.2.1 实现 `Config` Pydantic Settings 模型（api_key、model、data_dir 等字段）
- [x] T1.1.2.2 实现 `config set/get/show` CLI 子命令
- [x] T1.1.2.3 实现启动时配置校验，缺少 api_key 时引导用户配置
- [x] T1.1.2.4 单元测试：配置读写、脱敏展示、校验逻辑

---

## Epic E1.2 — 论文解析上下文

> Status: **DONE**
> 价值假设：高质量的论文解析是所有 Agent 分析的基础

### Story S1.2.1 — PDF 下载与解析

**验收条件**:
- 支持 `arxiv:2301.12345` 格式自动下载 PDF
- 支持本地 PDF 文件路径
- 解析输出结构化内容：标题、摘要、章节、正文、参考文献
- PDF 缓存到 `~/.research-agent/cache/papers/`

**Tasks**:
- [x] T1.2.1.1 实现 `ArxivFetcher`：根据 arXiv ID 下载 PDF
- [x] T1.2.1.2 实现 `PDFParser`（PyMuPDF）：提取结构化文本
- [x] T1.2.1.3 实现论文实体 `Paper` dataclass（id、title、authors、abstract、sections、full_text）
- [x] T1.2.1.4 实现本地缓存：已下载的 PDF 不重复下载
- [x] T1.2.1.5 单元测试：解析样本论文，验证关键字段提取

### Story S1.2.2 — 数据库基础层

**验收条件**:
- SQLite 数据库自动初始化（首次运行时建表）
- `Paper` 可存储和检索
- 支持按 ID、标题、标签查询

**Tasks**:
- [x] T1.2.2.1 实现 `Database` 类：初始化连接、建表（papers/ideas/discussions）
- [x] T1.2.2.2 实现 `PaperRepository`：CRUD 操作
- [x] T1.2.2.3 集成测试：存储 + 检索 Paper，验证数据完整性

---

## Epic E1.3 — 核心 Agent 系统

> 价值假设：Analyst + Critic 双视角是产品的核心差异化能力

### Story S1.3.1 — LLM 客户端封装

**验收条件**:
- 统一的 `LLMClient` 接口，底层调用 DeepSeek API
- 支持流式输出（streaming）
- API 错误时友好提示（而非堆栈）
- 测试时可注入 mock LLM

**Tasks**:
- [ ] T1.3.1.1 实现 `LLMClient`（openai SDK，base_url 指向 DeepSeek）
- [ ] T1.3.1.2 实现流式输出支持
- [ ] T1.3.1.3 实现重试逻辑（指数退避，最多 3 次）
- [ ] T1.3.1.4 定义 `LLMProvider` 抽象接口，便于后续切换模型

### Story S1.3.2 — Agent 基类

**验收条件**:
- `BaseAgent` 提供统一的调用接口：`agent.run(context) → AgentResponse`
- `AgentResponse` 包含：内容、Agent 角色标识、置信度/评分（可选）
- 所有 Agent 都有独立的 System Prompt

**Tasks**:
- [ ] T1.3.2.1 定义 `AgentResponse` dataclass
- [ ] T1.3.2.2 实现 `BaseAgent`（持有 LLMClient，定义 `run` 抽象方法）
- [ ] T1.3.2.3 实现 Agent System Prompt 加载机制（从 `prompts/` 目录读取 YAML）

### Story S1.3.3 — Analyst Agent

**验收条件**:
- 输入论文，输出：核心贡献、方法洞察、潜在影响、关联作品
- 明确区分"作者声称"和"实际有效"
- 单次分析 < 60s（含 LLM 调用）

**Tasks**:
- [ ] T1.3.3.1 设计并写入 Analyst System Prompt（角色：好奇心驱动，天然发现价值）
- [ ] T1.3.3.2 实现 `Analyst.analyze_paper(paper) → AnalysisResult`
- [ ] T1.3.3.3 定义 `AnalysisResult` schema（贡献列表、洞察、关联、置信度）
- [ ] T1.3.3.4 集成测试：对真实论文运行分析，人工验证输出质量

### Story S1.3.4 — Critic Agent

**验收条件**:
- 输入论文或 Idea，输出：≥1 个有依据的反对观点、支持度评分（1-10）
- 支持度 < 7 时必须说明原因
- 不允许打 10 分
- 即使找不到反对点也要诚实说明（而非强行反对）

**Tasks**:
- [ ] T1.3.4.1 设计并写入 Critic System Prompt（角色：魔鬼代言人，评分规则约束）
- [ ] T1.3.4.2 实现 `Critic.critique_paper(paper) → CritiqueResult`
- [ ] T1.3.4.3 定义 `CritiqueResult` schema（反对点列表、支持度评分、理由）
- [ ] T1.3.4.4 实现评分解析：从 LLM 输出中稳定提取数字评分
- [ ] T1.3.4.5 集成测试：验证评分范围、反对点非空

### Story S1.3.5 — Orchestrator 基础版

**验收条件**:
- 接受用户命令，路由到对应 Agent
- 并行调用 Analyst + Critic 分析论文
- 聚合输出，明确标注共识点和冲突点

**Tasks**:
- [ ] T1.3.5.1 实现 `Orchestrator.route(command, context) → Task`
- [ ] T1.3.5.2 实现并行 Agent 调用（asyncio.gather）
- [ ] T1.3.5.3 实现结果聚合：提取 Analyst/Critic 共识 + 分歧
- [ ] T1.3.5.4 集成测试：端到端论文分析流程

---

## Epic E1.4 — 基础记忆系统

> 价值假设：会话内记忆使对话连贯，是 Idea 讨论的前提

### Story S1.4.1 — 工作记忆（会话内）

**验收条件**:
- 同一会话内，Agent 能访问本次对话的完整历史
- 支持将历史注入 Agent 的 context window
- 会话结束后自动持久化到 SQLite

**Tasks**:
- [ ] T1.4.1.1 实现 `WorkingMemory`：持有当前会话消息列表
- [ ] T1.4.1.2 实现 `WorkingMemory.to_context(max_tokens)`：截断到指定 token 限制
- [ ] T1.4.1.3 实现会话自动保存到 `discussions` 表
- [ ] T1.4.1.4 单元测试：token 截断逻辑、持久化

---

## Epic E1.5 — CLI 命令接口

> 价值假设：完整的 CLI 接口让用户可以立即使用 MVP

### Story S1.5.1 — `research read` 命令

**验收条件**:
- `research read arxiv:2301.12345` 输出格式化的双视角分析
- `research read ./paper.pdf` 支持本地文件
- 输出包含：📄 Analyst 分析 + 🔴 Critic 批判 + 📊 综合评估
- 进度指示器显示下载/分析状态

**Tasks**:
- [ ] T1.5.1.1 实现 `read` CLI 命令（支持 arxiv ID 和文件路径）
- [ ] T1.5.1.2 实现格式化输出（Rich 渲染，带颜色分区）
- [ ] T1.5.1.3 实现进度条（下载 → 解析 → 分析 → 输出）
- [ ] T1.5.1.4 E2E 测试：`research read arxiv:xxx` 完整流程

### Story S1.5.2 — `research discuss` 命令

**验收条件**:
- `research discuss` 进入交互式对话循环
- 每次 AI 回复同时包含 Analyst 和 Critic 视角
- 支持 `exit` / `quit` / Ctrl+C 退出
- 退出时自动保存对话记录

**Tasks**:
- [ ] T1.5.2.1 实现 `discuss` CLI 命令（REPL 循环）
- [ ] T1.5.2.2 实现对话轮次：用户输入 → Orchestrator 路由 → 双视角回复
- [ ] T1.5.2.3 实现退出处理和对话持久化
