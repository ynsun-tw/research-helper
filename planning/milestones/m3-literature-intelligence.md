> Status: IN PROGRESS
> Index: [../../PLAN.md](../../PLAN.md)
>
> **M2.5 适配注记（2026-05）**：CLI 已收敛为单一 `research` REPL（详见
> [../notes.md](../notes.md) ADR-006）。本里程碑里所有写作 `research search` /
> `research read --batch` 的子命令形态，统一转译为 REPL 内的 slash 命令
> （`/search` / `/history` / `/read` …）以及 LLM tool calling 自动调度。
> 子任务文案仍保留原表述，但实现路径走 chat 工具注册表 + `SearchRepository`。

# M3 — 文献智能搜索

**目标用户**: 需要系统追踪相关文献的研究者
**交付物**: Searcher Agent（多源搜索、相关度评分）+ Memory Keeper 完整版（跨会话三层记忆）
**时间**: 3 周（Sprint 3: 2周 + Sprint 4: 1周）
**前置条件**: M2 完成（辩论模式、Idea 管理可用）
**验收标准**:
- 在 `research` REPL 中通过 `/search "efficient attention"`（或自然语言触发 `search_arxiv` 工具）返回多源搜索结果，含相关度评分
- 用户审核后批量阅读：REPL 内 `/queue add` → `/read --queue` 流程（或对应自然语言）
- Memory Keeper 能跨会话检索历史讨论和关联 Idea
- 主动提醒：新论文与历史搁置 Idea 相关时自动提示

---

## Epic E3.1 — Searcher Agent

> 价值假设：文献雷达减少研究者的信息盲区，多源覆盖避免单一数据源偏差

### Story S3.1.1 — arXiv 搜索集成

**验收条件**:
- 支持关键词、作者、类别搜索
- 返回结果包含：标题、作者、摘要、发表时间、arXiv ID
- 每次搜索最多返回 20 条（可配置）
- 结果按相关度排序

**Tasks**:
- [x] T3.1.1.1 实现 `ArxivSearcher`（已在 M2 用 arXiv Atom API 直接实现，见 `search/arxiv_search.py`）
- [x] T3.1.1.2 搜索结果持久化（M3.1 切片：`search_queries` + `search_results` 表 + `SearchRepository`；`/history [N]` 跨会话查看；缓存 TTL 等高级策略并入完整 Searcher Agent）
- [x] T3.1.1.3 结果去重标记：`SearchRepository.already_read` 与 `papers` 表 join，`/search` 输出表格里用 ✓ 标已读
- [x] T3.1.1.4 单元测试：`tests/unit/test_searches.py`（repo 顺序/限额/读标记/legacy DB 迁移 + `cmd_search`/`cmd_history` 集成）
- [x] T3.1.1.5 LLM-callable 历史（M3.1 v2 切片）：注册 `recent_searches(limit?)` 工具，让自然语言（"打开我昨天搜过的 BERT"）能先查历史拿 `arxiv_id` 再 `load_paper`；测试见 `tests/unit/test_chat_recent_searches.py`

### Story S3.1.2 — Semantic Scholar 集成

**验收条件**:
- 支持关键词搜索 + 引用图查询（向前/向后引用）
- 返回额外信息：引用数、影响力分数、开放访问链接
- 与 arXiv 结果合并去重

**Tasks**:
- [x] T3.1.2.1 实现 `SemanticScholarSearcher`（Graph API `/paper/search`，复用 `ArxivSearchHit` 类型，hits 标 `source="semantic_scholar"`；带 3s 客户端节流 + 礼貌 User-Agent + 429/timeout 重试 + 可选 `x-api-key`）。`paper_resolver.search_arxiv_papers` 自动 arXiv 主源 → S2 fallback，UI 在 fallback 时给出黄色提示；`/read` 仍能用 S2 暴露的 `externalIds.ArXiv` 加载。测试见 `tests/unit/test_semantic_scholar.py` + `tests/unit/test_paper_resolver.py` 的 fallback 用例。
- [x] T3.1.2.2 实现引用图查询：`SemanticScholarSearcher.get_citations(arxiv_id)` / `.get_references(arxiv_id)` 复用 `_request_bytes` 重试/节流；`/cites` `/refs` slash + `get_citations` `get_references` LLM 工具，默认从 anchor paper 推断 id，本地 PDF 没有 arXiv 映射时给出明确提示；详见 `tests/unit/test_chat_citations.py` + `tests/unit/test_semantic_scholar.py` 的 citation graph 用例。
- [ ] T3.1.2.3 实现跨源去重：基于标题相似度合并结果
- [ ] T3.1.2.4 集成测试：搜索已知论文，验证引用图数据完整性

### Story S3.1.3 — Searcher Agent 封装

**验收条件**:
- `Searcher.search(query, strategy) → SearchResult[]`
- 每条结果包含 Searcher 给出的相关度评分（0-1）和理由
- 诚实承认搜索局限性（如："未覆盖 ICLR 2024 年以后的论文"）
- 支持搜索策略调整：更理论/更应用/特定作者组

**Tasks**:
- [x] T3.1.3.1 Searcher system prompt（`prompts/searcher.yaml`，含 0-1 分数分档 + 诚实规则）
- [x] T3.1.3.2 `Searcher.score_hits(query, hits) → list[ArxivSearchHit]` 批量给所有候选 LLM 评分，写到 `relevance_score` / `relevance_reason` 字段并按分数降序持久化（`/search` / `/history` / `recent_searches` 全部按相关度排序）
- [ ] T3.1.3.3 搜索策略参数（`--mode theoretical/applied/group:<author>`）
- [ ] T3.1.3.4 集成测试：对比 LLM 相关度评分与人工评分的一致性（单元层面已覆盖 clamp / parse / fallback / sort，见 `tests/unit/test_searcher.py`）

---

## Epic E3.2 — 搜索流水线与用户审核

> 价值假设：用户审核关卡确保用户保持对研究方向的主导权

### Story S3.2.1 — `research search` 命令

**验收条件**:
- `research search "query"` 展示搜索结果列表（含相关度评分）
- 交互式审核：`[a]dd` / `[s]kip` / `[q]uit` 三键操作
- 审核通过的论文加入"待读队列"
- 支持 `--sources arxiv,semantic` 指定数据源

**Tasks**:
- [x] T3.2.1.1 `/search` slash + `search_arxiv` LLM 工具（M2.5 已完成）
- [x] T3.2.1.2 Rich 表格 + LLM 相关度分（task 1）
- [ ] T3.2.1.3 交互式单键审核（chat 模式下用 `/queue add <id>` + LLM 工具 `queue_add` 替代，仍可在后续做"批量审核"流程）
- [x] T3.2.1.4 待读队列持久化：`reading_queue` 表 + `ReadingQueueRepository`（`storage/reading_queue.py`），支持 pending / in_progress / done / skipped 四态

### Story S3.2.2 — 批量阅读

**验收条件**:
- `/queue read` 按顺序加载待读队列的下一篇并自动 Analyst + Critic
- 加载成功后自动把队列状态推进到 done（在 `_load_and_analyze` 成功路径里挂钩）
- 支持手动 `done` / `skip`，下一次 `/queue read` 自动取最早的 pending
- 队列状态持久化在 SQLite，重启 REPL 仍能继续

**Tasks**:
- [x] T3.2.2.1 `/queue read` 取下一篇 pending 并复用 `_load_and_analyze` 流程
- [x] T3.2.2.2 手动 `/queue done|skip <arxiv-id>` + 自动 mark done on success
- [x] T3.2.2.3 队列状态持久化（SQLite，FIFO by `added_at`）；LLM 可调 `queue_next` 拿下一篇 → 链入 `load_paper`
- [ ] T3.2.2.4 E2E 测试：批量读取 3 篇论文的完整流程（单元层面已覆盖 `/queue read` + auto-mark-done + FIFO，见 `tests/unit/test_reading_queue.py`）

### Story S3.2.3 — 动态搜索策略调整

**验收条件**:
- 基于 Analyst/Critic 的讨论内容，Searcher 自动调整后续搜索建议
- 用户可接受建议（`[y]es`）或自定义调整
- 搜索策略变化有说明理由

**Tasks**:
- [ ] T3.2.3.1 实现 Orchestrator 的搜索策略推导：从讨论历史中提取搜索意图
- [ ] T3.2.3.2 实现 `Searcher.suggest_refinement(discussion_context) → SearchSuggestion`
- [ ] T3.2.3.3 实现策略调整的 CLI 交互

---

## Epic E3.3 — Memory Keeper 完整版

> 价值假设：跨会话记忆让 Research Agent 真正成为"外脑"，而非每次重新开始

### Story S3.3.1 — 研究记忆（跨会话持久化）

**验收条件**:
- 所有论文分析、Idea 讨论自动持久化到 SQLite
- 重启后 `research discuss "xxx"` 能自动召回相关历史
- 向量化检索：基于语义相似度找到最相关的历史记录（Top-K）
- 检索延迟 < 2s（在 1000 条记录规模下）

**Tasks**:
- [x] T3.3.1.1 跨会话讨论持久化：`WorkingMemory.persist` 在 SQLite 写入的同时调用可选 indexer，把 user / analyst / critic 三类消息推入 ChromaDB（test/fallback 走 in-memory Jaccard），论文分析结果已在 M1/M2 落到 `papers` 表
- [x] T3.3.1.2 向量化流水线：新增 `storage/discussion_vectors.py::DiscussionVectorStore`（chroma 持久化目录与 idea 共用 `chroma_dir`），`ChatSession.close()` 自动调用 indexer
- [x] T3.3.1.3 语义检索：`MemoryKeeper.recall_history(query, limit, exclude_session_id)` + LLM 工具 `recall_history` + slash `/recall`；返回带 role / session_id / 片段的命中并回写控制台
- [ ] T3.3.1.4 性能测试：1000 条记录下检索延迟验证

### Story S3.3.2 — 元认知记忆

**验收条件**:
- 系统能分析用户的研究兴趣趋势（读了哪些 topic 的论文）
- 系统能识别用户的决策模式（哪类 Idea 最终被采纳/放弃）
- `research insights` 输出每月研究回顾报告

**Tasks**:
- [ ] T3.3.2.1 实现 `MetaMemory`：基于 SQLite 聚合分析
- [ ] T3.3.2.2 实现研究兴趣追踪（论文标签频率统计、时间维度分析）
- [ ] T3.3.2.3 实现决策模式识别（Idea 状态转换统计）
- [ ] T3.3.2.4 实现 `research insights` 命令，输出 Markdown 报告

---

## Epic E3.4 — 主动提醒机制

> 价值假设：主动提醒将被动工具升级为主动合作者

### Story S3.4.1 — 上下文触发提醒

**验收条件**:
- 当前讨论的论文与历史搁置 Idea 语义相似时，自动提醒（相似度 > 0.8）
- 提醒格式：简洁且不打断主流程
- 用户可一键查看关联 Idea 详情
- 误报率 < 20%（通过阈值调优）

**Tasks**:
- [ ] T3.4.1.1 实现 `MemoryKeeper.check_associations(current_context) → Association[]`
- [ ] T3.4.1.2 实现提醒注入：Orchestrator 在输出前检查并附加提醒
- [ ] T3.4.1.3 实现相似度阈值配置（`config set memory.alert_threshold 0.8`）
- [ ] T3.4.1.4 集成测试：构造历史 Idea + 相关新讨论，验证提醒触发

### Story S3.4.2 — 搁置 Idea 激活提醒

**验收条件**:
- 搁置 Idea 关联的"所需条件"（如：需要某数据集/代码）满足时自动提醒
- `research ideas shelve <id> --condition "需要 xxx 数据集"` 设置条件
- 搜索到相关论文时对比已有条件列表

**Tasks**:
- [ ] T3.4.2.1 在 Idea 表新增 `activation_conditions` 字段
- [ ] T3.4.2.2 实现条件设置 CLI：`ideas shelve --condition`
- [ ] T3.4.2.3 实现条件匹配：在搜索结果中检查是否满足任何 shelved Idea 的条件
