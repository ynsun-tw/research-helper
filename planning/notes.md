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

### ADR-008: 全功能迁移到 agent 工具层（Post-M5）
- **状态**: 已决定（2026-05）
- **触发**: ADR-006 已把交互入口从多 Typer 子命令收敛到单一 REPL，但 M4/M5 后续添加的写作流水线（`research write` / `figure` / `check` / `review`）、`style` 子命令、`config` 读写、`doctor` 诊断仍然只挂在 Typer CLI 下。在 REPL 内"我现在装的语料够吗？""帮我改一下 intro"等自然语言意图无法触发对应能力——用户必须退出 REPL → 跑 CLI 命令 → 再回到 REPL，对话定位被打断。
- **决策**: 在保留 Typer 子命令作为 scripting 入口的前提下（D1），把现有 CLI 的每一个动词在 `chat/tools.py` 注册为 LLM 工具，让自然语言可以触达全部能力。分三批落地（Phase 1/2/3），按副作用风险从低到高排队。
- **选型权衡**:
  - 是否删掉 Typer 子命令：否（D1）。`research insights --since 30d -o report.md` 等 cron / CI 场景仍有价值，破坏性变更收益不足。
  - 状态变更工具是否在工具层加二段确认（write a confirmation token，二次调用才执行）：否（D3）。改用 prompt 指令要求 LLM 先口头确认。代码省 ~200 行、可逆破坏面有限（样本可重训、config 可重写、fingerprint 有 archive 备份），承担"模型偶尔不遵守"的尾部风险。
  - `revise_draft` 是否在 REPL 里复刻 CLI 的 `--interactive` y/N 流程：否（T-2.5）。chat 里没有合适的 prompt-toolkit UI；改为自动 revise + 把 issue 列表 plain-print 由用户后续追问。
  - draft 缓存位置：session 内存而非 DB。生命周期与 REPL 一致；落盘是 `save_draft_to_file` 的唯一职责，杜绝 LLM 自作主张写文件。
- **实现要点**:
  - 三个 session 缓存槽：`recent_drafts: dict[section, list[Draft]]` / `recent_figures: dict[type, list[FigureDraft]]` / `recent_revisions: dict[section, Draft]`，由 `draft_section` / `draft_figure` / `revise_draft` 填充。
  - 统一引用语法 `latest` / `latest:<section>` / `latest:<section>:<version>`，供 `check_against` / `target` 等参数复用；解析集中在 `_parse_latest_ref` + `_resolve_latest_draft` + `_resolve_check_targets`。需要 path 接口（如 `cli_write.check_against`）时把 latest 内容 spill 到 per-call 的 `tempfile.mkdtemp` 再清理。
  - **Phase 1（只读）**：`run_doctor` / `style_show` / `style_history`（外加既有的 `research_insights`），共 11 个测试。
  - **Phase 2（写作 pipeline）**：`draft_section` / `draft_figure` / `save_draft_to_file` / `check_self_plagiarism` / `revise_draft`，共 25 个测试。`save_draft_to_file` 是磁盘写入的唯一入口；其他工具一律只缓存。
  - **Phase 3（状态变更）**：`train_style` / `build_fingerprint` / `update_fingerprint` / `get_config` / `set_config`，共 18 个测试。`set_config` 在返回字符串里强制 mask `api_key`，避免模型回显 secret。
  - `CHAT_SYSTEM_PROMPT` 同步增加 13 个工具描述 + 显式规则：no auto-save、style_show 用作 fingerprint 存在性检查、`revise_draft` 不假装人工判断、所有 **STATE-MUTATING** 工具调用前必须先 paraphrase + 询问 + 等待用户首肯。
- **影响**:
  - REPL 用户可以全程不退出做完"装样本 → 算指纹 → 写 intro → 查重 → 修订 → 保存 → 巡检"全链路。
  - Typer 子命令零行为变化，scripting 场景不受影响。
  - 工具层共 13 个新工具、+54 个单元测试、+0 个新依赖；`chat/tools.py` 从 ~1900 行涨到 ~2900 行，仍可单文件维护。
- **代价/风险**:
  - LLM 不遵守"先确认再调用 STATE-MUTATING 工具"的提示时，可能直接覆盖 fingerprint 或 config。已通过 prompt 反复强调 + `update_fingerprint` 会归档旧版本兜底，但仍存在用户被打断的可能。需要在 Phase 4 集成测试 + 人肉 C 阶段验证抽样确认 prompt 的命中率。
  - 长任务（draft_section 多并行 LLM 调用）目前没有流式进度反馈，REPL 里只能看到 `console.status("Thinking…")`；用户体验上"卡 30 秒没反应"。后续可引入 per-variant 阶段性 print。
  - `set_config` 改 api_key / model / base_url 不会重建 session 内的 LLMClient，本 session 仍走旧客户端；返回串里已明示需要重启 REPL，但用户可能忽略。
- **未做**: slash 命令在 Phase 2/3 没补全（写作类只通过自然语言路径触发）；写作流水线在 REPL 里没有 cancel / 中断机制；多个 figure type 共存时的 save 默认值选择策略只做了"二选一报错"，没做更智能的优先级。
- **后续可选演进**: 加上 `draft_section` 的流式进度、slash 快捷路径（`/write` `/figure` `/check` `/review` `/save-draft`）、`set_config` 后热重载 LLMClient、把 draft 缓存可选落盘到 SQLite 让跨 session 持续可用。

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

### ADR-009: REPL 输入层改用 prompt_toolkit + LLM 输出流式化（0.6）
- **状态**: 已决定（2026-05）
- **触发**: 0.5 系列 REPL 直接用 `rich.console.input`（其本质是 `input()` 加 Rich 渲染），导致：(a) 在 readline 没自动接入的终端（部分 zsh + brew Python、Windows ConPTY、tmux/screen 嵌套）回退键 / 方向键打印的是字面量字符，无法行编辑；(b) 没有跨会话历史 / Tab 补全 / Ctrl+R 反搜，"我上周 search 过的那个" 只能重新打字；(c) 用户问"画一张稀疏注意力的架构图，并用 200 字介绍它"这类大问题时，模型可能 30 秒不返回任何字符，体感像卡死。
- **决策**: 用 `prompt_toolkit ≥ 3.0` 接管 REPL 输入；同时给 `LLMProvider` 加一条流式带工具调用的接口，让最终文本 token-by-token 打到 console。
- **选型权衡**:
  - 输入层：`prompt_toolkit` vs 仅 `readline`/`pyreadline3`。前者跨平台一致（macOS libedit / Linux readline / Windows ConPTY 全覆盖）且自带 history / completer / keybinding API；后者在 Windows 上需要单独打包。选 `prompt_toolkit`。
  - 流式接口：(A) 新方法 `chat_with_tools_stream` 走 callback 注入文本 chunk、最终返回完整 `ChatResponse`；(B) 把 `chat_with_tools` 改成可选 `stream=True` 参数；(C) 让 `chat_with_tools` 始终走流式内部累积。选 A——保持向后兼容、callback 签名简单、Mock provider 可以零开销地一次回调全文。
  - 渲染：要不要在流式过程中保留 Rich Markdown 渲染？否。Rich Markdown 必须拿到完整字符串才能渲染（headers、表格、code fence 边界），与 token-by-token 输出不兼容。让步：流式期间渲染纯文本，必要时让用户在 `/history` 里看 Markdown 渲染过的版本（slash 输出仍走 Rich）。
  - tty 检测：CliRunner / 管道（`echo … | research`）没法做光标控制，prompt_toolkit 会立刻 EOF。决策：`sys.stdin.isatty() and sys.stdout.isatty()` 才构造 `PromptSession`；测试与脚本场景走老 `input_fn` 路径，完全不引入 prompt_toolkit 依赖。
- **实现要点**:
  - 新增 `chat/prompt_ui.py`：`build_prompt_session(cfg, command_names) -> PromptSession`（`FileHistory(~/.research-agent/repl_history)` + `_SlashCompleter`（仅 `/...` 前缀触发） + Ctrl+L 清屏 keybinding）；`render_state_prompt(session) -> FormattedText`，从 `anchor_paper` 与 `current_idea_id` 渲染 `You(arxiv:... | idea:...)> `。
  - `core/llm.py`：`LLMProvider.chat_with_tools_stream(messages, *, on_content_delta, tools, tool_choice, temperature, max_tokens) -> ChatResponse`。ABC 默认实现是非流式调用 + 一次性把整段 content 喂给 callback；`LLMClient` 真跑 `stream=True`，按 chunk.index 累积 tool_call 的 partial JSON args；`MockLLMProvider` 沿用非流式行为以保留测试语义。
  - `chat/router.py`：`run_chat` 只在 tty + 用户没传 `input_fn` 时构造 `PromptSession`；`_read_user_input` 优先 prompt_toolkit、回退 `input_fn`。Ctrl+C 处理升级为两段式：空 buffer 第一次按 → 仅警告 `Press Ctrl+C again to exit`，第二次 → 真退；中间任何成功输入都重置武装位。`?` / `/?` / `help` 在 router 层 rewrite 成 `/help`。
  - `_chat_with_llm` 拆出 `_stream_one_response`：先打 `…thinking…` 占位，第一个 token 到达时用 `\r` 抹掉、其后所有 token 直接走 `console.file.write`（绕过 Rich 的 line buffering），结束时补一个换行。
- **影响**:
  - 行编辑、`↑/↓` 历史、Tab 补 `/-cmd`、Ctrl+R 反搜在所有终端可用；REPL 历史跨会话持久化在 `~/.research-agent/repl_history`。
  - 用户从输入回车到第一个 token 显示的等待时间从"整段回答 latency"降到"首 token latency"，体感卡顿消失。
  - 流式期间没有 Markdown 渲染（headers / table / code fence 都按纯文本打），是知情代价。
  - 测试零改动：所有现有 `input_fn=lambda _: ...` 测试因为 tty 检测而走老路径；`CliRunner.invoke(input=...)` 同理。新加 13 个 unit test 覆盖 `_SlashCompleter` 三种语境、Ctrl+C 武装/重置、流式 callback、`?` 别名、状态 prompt 渲染。
- **代价/风险**:
  - 新增运行时依赖 `prompt-toolkit>=3.0`（约 600KB）。已是 IPython / pgcli 的传递依赖，安装面够广。
  - prompt_toolkit 与 Rich 共用 stdout 时偶有冲突；目前规避方式是流式期间不开 Rich live 渲染。如果未来要恢复 Markdown，需要在流结束后清屏重绘——目前不值得这个复杂度。
  - 当模型只返回 tool_call 不返回 content 时，"…thinking…" 占位会在 `_cap_tool_result` 介入前消失，用户会看到短暂空白后才出现 `→ calling ...`。可接受。
- **未做**: 多行输入（Alt+Enter）、命令树补全（`/queue add ?` → arxiv ids）、`undo` 命令撤回最近一对消息。三者都不阻断主路径，留作后续可选。

### ADR-010: Chat 路径 token 预算治理（0.6.1）
- **状态**: 已决定（2026-05）
- **触发**: 0.6.0 投产后用实测脚本量化 LLM 调用成本，发现每轮 round-trip 固定 overhead ≈ 7056 tokens（`CHAT_SYSTEM_PROMPT` 2616 + 27 个 tool schemas 4440），而单次"4-tool-call"自然语言对话整体 ≈ 48K tokens。在 DeepSeek $0.27/M input tier 下不致命，但若换 OpenAI / Anthropic 模型立刻不可忍。更关键的是 tool result 没有上限：搜索 20 条 + 论文全文 + 草稿全文等场景下，单条 `role=tool` 消息可达几千 tokens 并随着 agent loop 多轮回灌。
- **决策**: 六个改动一并落地，按 ROI 排序：
  1. **O1 系统提示词重写**: 砍掉每个工具的散文式说明（OpenAI schema 字段已经有），只保留 schema 表达不了的 chaining rules / state-mutation 合约 / 输出风格。
  2. **O2 top-5 fattest schema 精简**: `draft_section` / `draft_figure` / `save_draft_to_file` / `revise_draft` / `train_style` 五个 description 从多句长说明压成单行 + 参数级 hint。
  3. **O3 tool result 上限截断**: `MAX_TOOL_RESULT_CHARS = 8000`（~2000 tokens）；超长部分截断后附 "N chars elided; re-call with narrower args if needed"，让 LLM 知道可以再要更精细的输入。
  4. **O4 tool-log 过滤**: tool 结果仍要写入 `WorkingMemory` 留 audit / `/history`，但打 `metadata={"kind": "tool_log"}`；`_build_messages` 跳过这类条目，避免下一轮上下文里再次发送一份 400 字符摘要。
  5. **O5 history_limit 收紧**: 从 20 降到 8。需要长上下文的查询走 `recall_history` 工具按需召回。
  6. **O6 token telemetry**: 每轮末尾打印 `[~N in → ~M out tokens · K rounds]`（沿用 `memory/working_memory.py::estimate_tokens` 的 4 chars/token 启发式，无 tiktoken 依赖），让成本可见。
- **选型权衡**:
  - 是否引入 tiktoken 做精确计数：否。+800KB 依赖；OpenRouter 跨模型分词器各异，精度没法统一。4 chars/token 估值对趋势已经够用，用户主要关心相对量。
  - 是否上 prompt caching（Anthropic `cache_control` / OpenRouter 透传）：否，留作 ADR-011 候选。需要按 provider 分支处理 header / 消息分段，工程量大；当前优化先把"无脑节省"的部分吃干。
  - tool result 超长时是否做 LLM 自动摘要：否。会再开一个 LLM 调用反而花更多钱；让原工具在源头返回更紧凑的结果（如 `search_arxiv(max_results=5)`）才是正解。LLM 看到 elided 标记后会自然倾向更窄的下一次调用。
  - tool_log 是否完全不写 memory：否，留下做 `/history` 与未来分析。只是排除出 LLM 上下文。
- **实现要点**:
  - `chat/router.py` 新增 `MAX_TOOL_RESULT_CHARS` 常量 + `_cap_tool_result(text)` 帮手；agent loop 写 `messages.append(role="tool", content=_cap_tool_result(result_text))`。
  - `_build_messages` 新增 `eligible = [m for m in messages if m.metadata.get("kind") != "tool_log"]`，按 `history_limit=8` 截取。
  - 新增 `_estimate_message_tokens` / `_estimate_tools_tokens` / `_print_token_footer`，复用 `memory/working_memory.py::estimate_tokens`。
  - `CHAT_SYSTEM_PROMPT` 从 ~150 行散文压到 ~50 行，结构化为四段：routing rules / state-mutation rules / output rules / slash 提示。
- **影响**:
  - 每轮固定 overhead 7056 → 4453 tokens（-37%）。
  - 一次典型 4-tool-call 对话从 ~48K tokens → ~23K tokens（-52%）。
  - tool result 超长时模型不再被海量上下文淹没，更倾向于"重新调一次更精确的工具"。
  - 用户在 REPL 里能直接看到每轮成本，能感性判断是否值得换更便宜模型。
- **代价/风险**:
  - 系统提示词压缩可能让 LLM 偶尔遗漏某个 chaining rule（如先 `style_show` 再 `draft_section`）。已在新增 5 个测试 + 既有 700+ 测试中验证未回归。生产监控点：用户反馈"模型直接乱写没看 fingerprint"。
  - tool_log 不入下一轮上下文意味着 LLM 不再"记得"之前 tool 调过什么。这是预期行为（要回忆走 `recall_history`），但提示词需要明确告诉模型"如果忘了之前 tool 结果，重新调一次"。当前 prompt 没有显式这一句，依赖 LLM 自然行为；如果实测有问题再加。
  - 4 chars/token 在中文（每汉字 ≈ 1.5–2 tokens）下偏低估约 40%。footer 里数字是粗略指引，不是计费值。
- **未做**: prompt caching（ADR-011 候选）、tiktoken 精确计数、tool result 自动摘要、流式期间 cancel LLM 调用。
- **后续可选演进**: 把 ADR-010 的 `MAX_TOOL_RESULT_CHARS` / `history_limit` 改成 `config.yaml` 字段，让重度用户能在精度 vs 成本之间手动平衡；加 prompt caching 后预计同样 4-tool 对话能再降到 ~10K input tokens。
