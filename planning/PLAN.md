# Research Agent — Plan Dashboard

## 里程碑状态

| ID | 标题 | 时间 | 状态 | 详情 |
|----|------|------|------|------|
| M1 | MVP: 论文深度理解 | 2周 | DONE | [m1-paper-understanding.md](milestones/m1-paper-understanding.md) |
| M2 | Idea 讨论工坊 | 1周 | DONE | [m2-idea-workshop.md](milestones/m2-idea-workshop.md) |
| M2.5 | Conversational Shell（单入口 REPL + LLM tool calling） | 0.5周 | DONE | 见 [notes.md](notes.md) ADR-006 |
| M3 | 文献智能搜索 | 3周 | DONE | [m3-literature-intelligence.md](milestones/m3-literature-intelligence.md) |
| M4 | 论文写作助手 | 2周 | DONE | [m4-writing-assistant.md](milestones/m4-writing-assistant.md) |
| M5 | 研究自动化与完善 | 4周 | DONE (scope trimmed: E5.3 + E5.4 shipped; E5.1 reproduce + E5.2 full TUI deferred) | [m5-research-automation.md](milestones/m5-research-automation.md) |
| M6 | 从用户反馈学习（Self-Evolution） | 1.5周 | PLANNED | [m6-self-evolution.md](milestones/m6-self-evolution.md) |

**总计**: ~13.5周（M1–M5 已完成 ~12 周；M6 待开始）

---

## Epic 状态

| Epic | 标题 | 里程碑 | 状态 |
|------|------|--------|------|
| E1.1 | 项目基础搭建 | M1 | DONE |
| E1.2 | 论文解析上下文 | M1 | DONE |
| E1.3 | 核心 Agent 系统（Orchestrator + Analyst + Critic） | M1 | DONE |
| E1.4 | 基础记忆系统 | M1 | DONE |
| E1.5 | CLI 命令接口 | M1 | DONE |
| E2.1 | 辩论模式 | M2 | DONE |
| E2.2 | Critic 评分系统 | M2 | DONE |
| E2.3 | Idea 生命周期管理 | M2 | DONE |
| E2.5.1 | REPL + slash 工具入口 | M2.5 | DONE |
| E2.5.2 | LLM tool calling 与 agent loop | M2.5 | DONE |
| E3.1 | Searcher Agent | M3 | DONE |
| E3.2 | 搜索流水线与用户审核 | M3 | DONE |
| E3.3 | Memory Keeper 完整版 | M3 | DONE |
| E3.4 | 主动提醒机制 | M3 | DONE |
| E4.1 | 写作风格指纹提取 | M4 | DONE |
| E4.2 | Scribe Agent | M4 | DONE |
| E4.3 | 写作审查流水线 | M4 | DONE |
| E4.4 | Self-Plagiarism 检测 | M4 | DONE |
| E5.1 | 代码复现系统 | M5 | DEFERRED |
| E5.2 | TUI 完整界面 | M5 | DEFERRED |
| E5.3 | 图表生成 | M5 | DONE |
| E5.4 | 全系统优化与打磨 | M5 | DONE |
| E6.1 | Reviewer-Feedback Aware Critic & Scribe | M6 | PLANNED |
| E6.2 | Critic 评分校准 | M6 | PLANNED |
| E6.3 | Searcher 偏好学习 | M6 | PLANNED |
| E6.4 | 自我进化观测与治理 | M6 | PLANNED |

---

## Post-M5 跨切重构

完成所有里程碑后落地的、不归属任何 epic 的横向改进。每条都附 ADR。

| ID | 标题 | 日期 | 状态 | 详情 |
|----|------|------|------|------|
| R-001 | Agent JSON 解析迁移到 Pydantic v2 schemas | 2026-05 | DONE | [notes.md](notes.md) ADR-007 |
| R-002 | 全功能迁移到 agent 工具层（自然语言入口） | 2026-05 | DONE | [notes.md](notes.md) ADR-008 |
| R-003 | Chat 路径 token 预算治理（第一轮） | 2026-05 | DONE | [notes.md](notes.md) ADR-010 |
| R-004 | Chat 路径上下文管理（第二轮） | 2026-05 | DONE | [notes.md](notes.md) ADR-011 |

R-001 摘要：6 个 agent 的 11 个 `extract_json` 调用点全部迁移到 `research_agent/agents/schemas.py` 的 Pydantic v2 模型 + 单一入口 `parse_model(raw, Model)`。`agents/base.py::extract_json` 删除；`pydantic>=2.7` 升为直接依赖；新增 21 个 contract 测试锁定容错语义；CLI 启动时间与覆盖率不变。

R-002 摘要：在 `chat/tools.py` 注册 13 个新 LLM 工具，把 `research write/figure/check/review` 全套写作流水线 + `style train/fingerprint/update/show/history` + `config get/set` + `doctor` 全部以自然语言入口暴露给 REPL agent；Typer 子命令保留作 scripting 入口（D1）。session 引入 `recent_drafts` / `recent_figures` / `recent_revisions` 三个内存槽缓存中间产物，`save_draft_to_file` 是唯一磁盘写入入口（D6）。状态变更类工具不在工具层硬控（D3），靠 system prompt 让 LLM 先 paraphrase + 等用户确认。+54 单元测试，零新依赖。

R-003 摘要（ADR-010 / 0.6.1）：6 条改动一次性落地——`CHAT_SYSTEM_PROMPT` 散文重写 + 5 个最胖 schema 节食 + `MAX_TOOL_RESULT_CHARS=8000` tool result 上限 + `kind=tool_log` 标签让 tool 输出不进下一轮 history + `history_limit` 从 20 收紧到 8 + 每轮末尾打印 `[~N in → ~M out tokens · K rounds]` 成本 footer。固定 overhead 7056 → 4453 tokens（−37%），典型 4-tool 对话 ~48K → ~23K tokens（−52%）。

R-004 摘要（ADR-011，跟在 R-003 后）：T1 三项（tool schema 节食、token-budget 选 history、slash 内存写入标 `tool_log`），T2 三项（滚动摘要 compactor、model-aware context budget 查表、idempotent 工具 scratchpad 跨轮缓存），T3 四项 opt-in（tiktoken、loop 内同调用去重、per-tool result cap、provider usage 真实 token 计数）。28 个 schema 序列化从 ~4022 → ~2832 tokens（−29.6%），6 轮 loop 累计省 ~7K tokens；Claude/GPT-4o 上 context 预算从 8K → 196K/124K；长会话 (>16 turns) 自动滚动摘要。+44 单元测试，788 全绿。

---

## 参考文档

- [架构设计](architecture.md) — 系统架构、技术栈、数据流
- [注记与风险](notes.md) — 假设、风险、ADR 决策记录
