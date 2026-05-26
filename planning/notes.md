# Notes — Assumptions, Risks & Open Questions

## 假设

### 技术假设
- DeepSeek API 稳定可用，延迟 < 5s（P95）
- 本地机器有足够存储（推荐 10GB+）用于 PDF 缓存和向量库
- 用户有至少 2-3 篇已发表论文供 Scribe 学习风格（M4 前提条件）
- arXiv 和 Semantic Scholar 的公开 API 在目标地区可访问

### 产品假设
- 目标用户（博士生/独立研究者）习惯命令行工具
- 用户愿意在初次使用时花 10-15 分钟完成风格指纹训练
- 批判性反馈不会让用户感到沮丧，而是视为有价值的对话

---

## 风险

### 高风险
| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| DeepSeek API 上下文限制导致长论文分析截断 | 高 | 高 | 分块处理 + 摘要链策略 |
| Critic Agent 产出流于形式（为反对而反对） | 中 | 高 | Prompt 工程 + 定期校准测试 |
| ChromaDB 在大量论文后检索质量下降 | 中 | 中 | 定期 re-indexing + 分集合管理 |
| arXiv/Semantic Scholar API 限流 | 中 | 中 | 本地缓存 + 指数退避重试 |

### 中风险
| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| PDF 解析质量对双栏/数学公式论文不稳定 | 高 | 中 | PyMuPDF 优先 + 降级处理 |
| 风格指纹学习样本不足时 Scribe 输出质量差 | 中 | 中 | 最低样本量检查 + 明确提示 |
| TUI 在不同终端的兼容性问题 | 低 | 中 | 优先保证 CLI 模式，TUI 为增强 |

---

## 开放问题

### 产品方向
- [ ] **P1** Critic 支持度评分的校准机制如何设计？是否需要用户反馈来调整基线？
- [ ] **P2** Memory Keeper 的主动提醒频率如何平衡？过多提醒会打断思路，过少则失去价值
- [ ] **P3** 多用户场景是否需要支持？（研究组共享一个实例）

### 技术方向
- [ ] **T1** Agent 间通信采用 Python 函数调用还是消息队列？当前倾向函数调用（同步，简单）
- [ ] **T2** 向量嵌入模型选择：本地（sentence-transformers）vs API（DeepSeek/OpenAI embedding）？
  - 本地：离线可用，无 API 成本，但首次下载大
  - API：质量更高，但增加网络依赖
- [ ] **T3** 论文全文 vs 摘要+关键段落的向量化策略对检索质量的影响待评估
- [ ] **T4** TUI 是否需要支持多窗格同时展示多 Agent 输出？

### 未来特性（超出当前 Roadmap）
- 与文献管理工具集成（Zotero、Mendeley）
- 邮件摘要（每日 arXiv digest）
- 多模态支持（图表理解）
- 协作模式（研究组共享记忆库）

---

## 决策记录

### ADR-001: 选择 SQLite 而非 PostgreSQL
- **状态**: 已决定
- **原因**: 目标用户是个人研究者，零配置是关键；SQLite FTS5 满足全文搜索需求；数据量（<100万条记录）在 SQLite 能力范围内

### ADR-002: 选择 DeepSeek 而非 OpenAI 作为默认 LLM
- **状态**: 已演进 → 见 ADR-005
- **原因**: 兼容 OpenAI SDK（迁移成本低）；长上下文窗口（适合论文分析）；成本显著低于 GPT-4
- **风险**: DeepSeek 可用性/政策风险，代码中通过 `LLMProvider` 抽象层隔离

### ADR-003: 优先 CLI 模式，TUI 为增强
- **状态**: 已决定
- **原因**: CLI 模式可脚本化、可管道化；TUI 需要更多开发资源；MVP 不依赖 TUI

### ADR-004: Agent 回复语言可配置（`language`）
- **状态**: 已决定（M1 后增强，2026-05）
- **原因**: 中英文用户均需可读的双视角分析；配置项优于每次在 prompt 里说明
- **实现**: `config.yaml` 的 `language`（`en` | `zh`，默认 `en`）；`core/language.py` 向 Analyst/Critic system prompt 追加语言指令
- **范围**: 仅 Agent 生成内容；CLI 界面与 PDF 原文语言不变

### ADR-005: 默认 LLM 网关改为 OpenRouter
- **状态**: 已决定（M1 后增强，2026-05）
- **原因**: 单 key 切换多模型；与 OpenAI SDK 兼容；`sk-or-` key 与 `provider/model` slug 对齐
- **实现**: `base_url` 默认 `https://openrouter.ai/api/v1`；`api_key` 与 `base_url` 不一致时自动迁移

### ADR-007: Agent JSON 解析迁移到 Pydantic v2 schemas（Post-M5）
- **状态**: 已决定（2026-05）
- **触发**: 全部 6 个 agent（Analyst / Critic / Scribe / Searcher / Illustrator / writing_pipeline）都靠 `agents/base.py::extract_json` + `data.get(key, default)` 解析 LLM JSON，分散在 11 个 `_parse_*` 函数里，每个都自带一份 `_as_str_list` + 区间 clamp 的样板代码。类型上拿到的是裸 `dict[str, Any]`，下游无法约束。
- **决策**: 引入 `research_agent/agents/schemas.py`，所有 LLM JSON 回复用 Pydantic v2 模型 + 单一入口 `parse_model(raw, Model)` 解析。
- **选型权衡**: 评估了 (A) `pydantic-ai` 全框架 / (B) 仅 Pydantic v2 校验 / (C) Pydantic + 原生 structured outputs 三条路线，选 B：零新框架、零和 `LLMProvider` 抢地盘、不动 `chat/router.py`、`MockLLMProvider` 测试模式不变、CLI 启动时间不退化（`pydantic` 通过 `pydantic-settings` 早已是传递依赖）。C 留作后续可选启用路径，A 因违反 local-first + 最小依赖约束被否决。
- **实现要点**:
  - `schemas.py` 提供 10 个模型：`AnalysisPayload` / `IdeaSupportPayload` / `ConclusionPayload` / `CritiquePayload` / `WritingReviewPayload` / `DraftPayload` / `SearcherScoreItem` + `SearcherScoresPayload` / `SearcherRefinementPayload` / `FigurePayload` / `ClaimEvidencePayload`
  - 全部 schema 统一约定：`extra="ignore"` + 全字段默认 + `mode="before"` `field_validator` 做容错 coerce（错型列表 → `list[str]`、`confidence` clamp 到 [0,1]、Critic `support_score` clamp 到 [1,9] 不允许 10）
  - `parse_model` 把 `ValidationError`（Pydantic v2 中本身就是 `ValueError` 子类）和 JSON decode 错误统一抛 `ValueError`，agent 里既有的 `except ValueError` 优雅降级路径零改动
  - Critic 保留 free-text "Rating: 7/10" 正则兜底：用 `_score_key_present()` 探测 JSON 信封里是否带 score 字段，没有才走老的 `parse_support_score` 正则路径
  - `extract_json` 从 `agents/base.py` 删除；唯一的"裸 JSON 信封"访问点是新公开的 `schemas.strip_to_json()`
  - `pydantic>=2.7` 提升为 `pyproject.toml` 直接依赖
- **影响**:
  - 下游拿到的是带类型的 Pydantic 实例而不是 `dict[str, Any]`，IDE 补全 / mypy / 字段约束全部生效
  - 字段范围校验（`support_score ∈ [1,9]`、`confidence ∈ [0,1]`）从每个 agent 收敛到一个 schema 文件，DRY
  - 11 个 `_as_str_list` / 内联 clamp 样板被消掉
  - 新增 21 个 contract 测试（`tests/unit/test_agent_schemas.py`）锁定容错语义（垃圾 JSON → ValueError、缺字段默认、错型 coerce、范围 clamp、`assumption/basis` ↔ `claim/evidence` 别名）
- **代价/风险**:
  - 增加一个直接依赖 `pydantic>=2.7`（实际零成本：`pydantic-settings` 早已传递引入）
  - Pydantic v3 升级时需要回看 `field_validator(mode="before")` API 是否保持兼容
- **未做**: LangGraph 编排写作流水线、OpenTelemetry 可观测性 — 这两条建议作为后续可选改进保留，不在本次范围。
- **后续可选演进**: 当主用模型（DeepSeek 等）稳定支持 OpenAI 风格 `response_format={"type": "json_schema"}` 时，可在 `LLMProvider.chat` 加 opt-in 参数走原生 structured outputs，schema 本身不需要改 — 这是 ADR-007 设计时刻意留出的演进口子。

### ADR-006: CLI 改为对话式 REPL（M2.5）
- **状态**: 已决定（2026-05）
- **触发**: 多子命令工具（`research read` / `discuss -p ...` / `ideas ...`）每条命令独立完成一次任务后退出，与"研究合作研究员"的对话定位不符；论文锚定/idea 状态/记忆等会话状态在子命令之间没有自然承载。
- **决策**: 收敛到单一入口 `research` 直接进入对话 shell。仅保留 `research config` 作为非交互子命令。其余功能改由 REPL 内提供：
  - 显式 slash 命令：`/search`、`/read`、`/discuss`、`/paper`、`/idea`、`/ideas`、`/help`、`/exit`
  - 自然语言输入由 LLM 通过 OpenAI 风格 function/tool calling 自主调度同一套工具（`search_arxiv` / `load_paper` / `discuss_idea` / `save_current_idea` / `list_ideas`）
- **实现要点**:
  - 新增 `src/research_agent/chat/`：`session.py`（`ChatSession` 持 anchor_paper、debate、memory、repos）、`router.py`（slash 分发 + LLM agent loop）、`tools.py`（slash + LLM 工具注册表）
  - 扩展 `core/llm.py`：`ChatMessage` 新增 `tool_call_id`/`tool_calls`/`name`；新增 `ToolCall`/`ChatResponse`；`LLMProvider.chat_with_tools` + `LLMClient` 走非流式 + tool 参数；`MockLLMProvider` 支持入队结构化 tool_call 响应
  - 业务逻辑复用：`run_read` / `run_discuss` / `run_ideas_*` / `_handle_debate_turn` / `_load_anchor_paper` 维持原状，被 chat 工具薄包装
  - Agent loop 上限 `MAX_TOOL_ITERATIONS=6`，工具错误以文本形式回灌给 LLM 而不抛异常，未知工具同样回灌错误信息
- **影响**:
  - `research read` / `research discuss` / `research ideas ...` 三组 typer 子命令移除（service 函数仍可被 import）
  - `tests/unit/test_cli.py` / `tests/e2e/test_read_cli.py` 改为驱动 REPL（输入 `/read ...\n/discuss ...\n/exit`）
  - 新增测试：`tests/unit/test_chat_router.py`（slash 分发 + LLM fallback）、`tests/unit/test_chat_tool_loop.py`（多轮 tool_call + 错误 + 循环上限）
- **代价/风险**:
  - REPL 单一入口让脚本/管道场景失能；如有 batch 需求，需要在后续里程碑提供 `--prompt`/`--once` 模式
  - LLM tool calling 需要模型支持 OpenAI function-calling；mock provider 已覆盖单元测试，但部分 OpenRouter 模型行为差异需要 e2e 验证
