> Status: COMPLETE (only DEFERRED items remain — see T3.1.3.4 and T3.2.1.3)
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
- [x] T3.1.2.3 实现跨源去重：`core/paper_resolver.dedupe_hits` 用 (arxiv_id_无版本) 主键 + `difflib.SequenceMatcher` 归一化标题相似度 ≥ 0.9 作为防御性兜底；早出现的来源（arXiv）在冲突时胜出，源标签保留。新增 `search_arxiv_papers(merge_sources=True)` 同时查双源、合并、去重、单源失败容错（仅当两源同时失败/空才抛 `PaperLoadError`）。默认行为不变（仍是 fallback 模式）。13 个单测覆盖版本号剥离、标题归一化、相似度阈值、合并顺序、单源容错、cap，见 `tests/unit/test_paper_resolver.py`。
- [x] T3.1.2.4 集成测试：`tests/integration/test_citation_graph_live.py` 用 arXiv:1706.03762 (Attention Is All You Need) 验证 `search` / `get_citations` / `get_references` / pagination / search→references 往返一致性。所有 hit 必须有有效 arxiv id 格式 + 非空 title + `source="semantic_scholar"` 且无重复。默认 skip，靠 `RUN_NETWORK_TESTS=1` 开启；S2 临时 429/timeout 自动转 skip 而非 fail。已知限制：`get_citations` 对热门论文常返回空（S2 第一页全是无 arxiv 映射的期刊/会议论文，被现有 filter 砍掉），列为后续 UX 改进点（M3 之外）。

### Story S3.1.3 — Searcher Agent 封装

**验收条件**:
- `Searcher.search(query, strategy) → SearchResult[]`
- 每条结果包含 Searcher 给出的相关度评分（0-1）和理由
- 诚实承认搜索局限性（如："未覆盖 ICLR 2024 年以后的论文"）
- 支持搜索策略调整：更理论/更应用/特定作者组

**Tasks**:
- [x] T3.1.3.1 Searcher system prompt（`prompts/searcher.yaml`，含 0-1 分数分档 + 诚实规则）
- [x] T3.1.3.2 `Searcher.score_hits(query, hits) → list[ArxivSearchHit]` 批量给所有候选 LLM 评分，写到 `relevance_score` / `relevance_reason` 字段并按分数降序持久化（`/search` / `/history` / `recent_searches` 全部按相关度排序）
- [x] T3.1.3.3 搜索策略参数：`paper_resolver.parse_search_mode` + `apply_search_mode` 把 mode 翻译为查询前缀偏置（theoretical → "theoretical analysis convergence ..."，applied → "empirical evaluation benchmark ..."，group:<author> → "by <author> ..."）。`search_arxiv_papers` 接受 `mode` 参数；`/search [--mode <m>] <keywords>` slash 支持 shlex 引号解析（多词作者名 `--mode "group:Andrej Karpathy"`）；`search_arxiv` LLM 工具 schema 新增 `mode` 字段，未知 mode 返回 error 但不破坏现有调用。15 个单测覆盖 parse / apply / 不同 mode / 边界条件 / 引号解析 / unbalance fallback / LLM 工具 schema，见 `tests/unit/test_paper_resolver.py` + `tests/unit/test_search_modes_chat.py`。
- [~] T3.1.3.4 **DEFERRED**：LLM 相关度评分 vs 人工评分一致性需要人工标注数据集，等积累 50+ 真实 `/search` 会话再做。当前单元层已覆盖 clamp / parse / fallback / sort 等机械保证，见 `tests/unit/test_searcher.py`。

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
- [~] T3.2.1.3 **DEFERRED**：chat 模式下用 `/queue add <id>` + LLM 工具 `queue_add` 已覆盖单点入队需求；后续若引入 typer-prompt 风格批量审核（一次性翻完 N 条新搜索结果）再补单独 task。
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
- [x] T3.2.2.4 E2E 测试：`tests/e2e/test_queue_batch_read.py` 用 typer `CliRunner` 真跑 `/queue add` × 3 → `/queue read` × 3 → `/queue list all` 完整脚本，断言 console 输出 + SQLite 落地（3 个 queue entry 全 done，3 篇论文都进 `PaperRepository`）。**顺带修了一个真实 bug**：`_load_and_analyze` 用 `paper.id` (`arxiv:1706.03762`) 查队列，但 queue 存的是 raw arxiv id (`1706.03762`)，导致 auto-mark-done 在真实链路上从来没生效；之前的单测把 `paper.id` 直接写成 `"1706.03762"` 把这个 bug 盖住了。修复 + 单测对齐真实 prefix。还覆盖了「队列清空后再 `/queue read` 给出 empty 提示而非 crash」边界。

### Story S3.2.3 — 动态搜索策略调整

**验收条件**:
- 基于 Analyst/Critic 的讨论内容，Searcher 自动调整后续搜索建议
- 用户可接受建议（`[y]es`）或自定义调整
- 搜索策略变化有说明理由

**Tasks**:
- [x] T3.2.3.1 `Orchestrator.extract_search_context(memory, *, max_messages=12, max_chars=3000)` 从 working memory 取最近 N 条消息（任意 role）格式化成 `role: content` 串，每条单消息超过 600 字符自动截断，总长度超过 max_chars 时保留尾部并加 `…\n` 标记前缀。空 memory / 全空内容返回 `""`，让 `/refine` 短路。6 个单测见 `tests/unit/test_searcher_refinement.py` 的 orchestrator 部分。
- [x] T3.2.3.2 `Searcher.suggest_refinement(discussion_context, *, previous_query=None) → SearchSuggestion`：把上一次查询 + 讨论摘要塞进一个独立的 system prompt（要求 LLM 输出 `{query, mode, reason, confidence}` JSON）。`_parse_refinement` 校验 mode（仅放行 `theoretical` / `applied` / `group:<author>`；author 为空 → `mode=None`），把 confidence clamp 到 [0,1]，partial / garbage / 空 context 都返回 degenerate `SearchSuggestion(query="", ...)` 让调用方安全短路。8 个单测覆盖空 context / 完整 payload / 未知 mode / group mode 规范化 / 空作者拒绝 / confidence 越界 clamp / 非 JSON 输出 / 部分字段。
- [x] T3.2.3.3 `/refine` slash + `suggest_search_refinement` LLM 工具：slash 先调 orchestrator helper 取 context，再调 Searcher，渲染建议 banner（`Suggested next search: ... (mode: ...) Reason: ... (confidence X%)`），随后通过 `session.input_fn` 走 accept (`y` 默认) / edit (`e <new>`) / skip (`s`) 三档输入循环；接受时调 `cmd_search`（自动拼出 `--mode <m>` 前缀，多词 group mode 自动加引号），EOF / Ctrl-C 静默 skip。`ChatSession.last_search_query` 字段记录上一次查询，让 refinement 能反思"我刚搜了 X，下一步搜什么"。CHAT_SYSTEM_PROMPT 同步增加 `suggest_search_refinement` 工具说明 + `/refine` slash 列表。14 个单测见 `tests/unit/test_chat_refine_slash.py`（空 context 静默 / 空建议 hint / accept 派发 / skip 不派发 / edit 走自定义 / edit 空回退 / EOF 视为 skip / system memory 写入 / LLM tool 无 context 报错 / LLM tool 返回 payload / LLM tool 空建议 / `_format_search_args` 单元）。

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
- [x] T3.3.1.4 性能测试：`tests/integration/test_recall_perf.py` 索引 1000 条消息（10 个模拟会话 × 100 条，5 个主题轮换），断言 `DiscussionVectorStore.query_similar` < 500ms，`MemoryKeeper.recall_history`（含 SQLite by-id 取回）< 2s（spec 目标），并验证 top-3 命中确实包含查询主题。跑 in-memory Jaccard fallback（最坏情况），实际跑时 1000 条消息 + 3 测试共 ~1.0s。

### Story S3.3.2 — 元认知记忆

**验收条件**:
- 系统能分析用户的研究兴趣趋势（读了哪些 topic 的论文）
- 系统能识别用户的决策模式（哪类 Idea 最终被采纳/放弃）
- `research insights` 输出每月研究回顾报告

**Tasks**:
- [x] T3.3.2.1 `agents/meta_memory.py::MetaMemory(db).compute(since_days?) → InsightsReport`：纯 SQL 聚合 (no LLM)，遍历 `papers` / `ideas` / `discussions` 三张表，吐出一个 dataclass，含 paper_count / papers_by_year / top_tags / top_authors / top_venues / idea_count / ideas_by_status / avg_critic_score / most_engaged_ideas / top_scored_ideas / session_count / message_count / role_counts / recent_session_ids。`since_days` 把窗口收敛到 `WHERE created_at >= ?`（all-time / last N days 两种 period 标签）。`_parse_json_list` 对 NULL / 非 JSON / 非 list 列容错为 `[]`，老库不会炸。
- [x] T3.3.2.2 研究兴趣追踪：`_fill_papers` 用 Counter 累加 tags / authors / venues / 年份，Counter.most_common(top_n=5) 切片。`papers_by_year` 按年份倒序，方便看到"今年读了多少篇"。时间维度通过 `since_days` 实现（30d/7d/6m/1y/all），CLI 都支持。
- [x] T3.3.2.3 决策模式：`_fill_ideas` 用 status Counter 直接得到现状分布；`avg_critic_score` 只对有 `critic_score` 的 idea 求均值（None 排除）；`most_engaged_ideas` 按 `len(score_history)` 倒序（>0 才进榜），代表用户最反复辩论的；`top_scored_ideas` 按 critic_score 倒序。完整状态变迁 audit log 暂不存（需要后续 schema 工作），现状已覆盖"哪些想法被升迁到 experimenting / completed、哪些被 abandoned"这一基础视图。
- [x] T3.3.2.4 `research insights [--since 7d|30d|6m|1y|all] [--output report.md]` Typer 子命令 + `/insights` slash + `research_insights` LLM 工具，都共用 `InsightsReport.to_markdown()` 渲染。Markdown 报告分四块 (Header / Papers / Ideas / Discussions)，结构稳定可直接 commit 进 repo。`/insights` 同步把 markdown 写进 working memory，让 LLM 后续能回答"上周读得最多的 venue 是？"。Rich Markdown 渲染走 console。13 个 chat-slash 单测 + 3 个 typer CLI 单测 + 9 个 MetaMemory 行为单测，覆盖 empty DB / paper rollup / idea rollup / no-score 边界 / discussion rollup / since_days 过滤 / 模板存在性 / 坏 JSON 容错。

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
- [x] T3.4.1.1 实现 `MemoryKeeper.check_associations(context, threshold=0.8, limit=5, statuses=("shelved","waiting"))` 返回 `Association(idea, similarity)`；over-fetch 3× 然后按 threshold + status 过滤；orphan vector 行（SQL 行已删）自动跳过；空 context 返回 []。配套 `IdeaVectorStore.query_with_scores` 把 Chroma cosine distance / Jaccard 都统一映射到 `[0,1]` 区间。`format_associations` 生成简洁 Rich 提示行（一条 idea 一行，含相似度 % + 状态 + `/idea show <prefix>` 跳转）。15 个单测 + 3 个 vector store 契约测试，见 `tests/unit/test_memory_keeper.py`。
- [x] T3.4.1.2 实现提醒注入：`chat/tools._surface_parked_idea_alerts(session, paper)` 在 `/read` 完成 Analyst+Critic 渲染后立即调用 `MemoryKeeper.check_associations(paper.title+abstract)`，把命中的 shelved/waiting idea 渲染成简洁 Rich banner（一行一条，含相似度% + 状态 + `/idea show <prefix>` 快捷方式），同时把摘要塞进 working memory 让 LLM 后续也能看见。任何失败 swallowed（不打断 /read）；空 title+abstract 直接 short-circuit。8 个单测见 `tests/unit/test_chat_alert_injection.py`。
- [x] T3.4.1.3 阈值配置：`Config.alert_threshold: float = 0.8`（pydantic 校验 + clamp 到 [0,1]），存 YAML，`research config set alert_threshold 0.85` CLI 入口，`config show` 显示当前值。非数字/越界拒绝并给出可读 error。4 个 CLI 单测 + 集成里实际 round-trip 阈值。
- [x] T3.4.1.4 集成测试：`tests/integration/test_parked_idea_alerts.py` 真跑 `/read` (`_load_and_analyze`) 流程：构造一篇与 shelved idea 主题重叠的论文 + 一个不相关的 idea（gardening），验证 banner 命中前者、忽略后者；阈值调高时不触发；空库 silent；banner 包含 `/idea show <prefix>` 快捷方式。Jaccard fallback (`use_chroma=False`)，跑时 ~0.3s。

### Story S3.4.2 — 搁置 Idea 激活提醒

**验收条件**:
- 搁置 Idea 关联的"所需条件"（如：需要某数据集/代码）满足时自动提醒
- `research ideas shelve <id> --condition "需要 xxx 数据集"` 设置条件
- 搜索到相关论文时对比已有条件列表

**Tasks**:
- [x] T3.4.2.1 `Idea.activation_conditions: list[str]`（dataclass + sqlite 列 + idempotent migration `ALTER TABLE ideas ADD COLUMN activation_conditions TEXT`）。`IdeaRepository.add_activation_condition` 做 case-insensitive 去重 + 空字符串短路，`clear_activation_conditions` 一键清空。row hydration 容忍老库 NULL 值（fall back 到 `[]`，不破坏从 M2 升级上来的数据库）。4 个单测见 `tests/unit/test_ideas.py`（持久化、去重、清空、legacy NULL）。
- [x] T3.4.2.2 `/ideas update <id> --condition <phrase>` slash（贪婪吞到下一个 `--` flag 之前，所以多词条件无需引号；可在一行里链式 `--condition A --condition B`）+ `--clear-conditions`。底层 `cli_ideas.run_ideas_update` 把 `conditions: list[str]` 和 `clear_conditions: bool` 串到 `IdeaRepository`；`/ideas show` 输出多一段 "Activation conditions" block。7 个 slash 单测见 `tests/unit/test_chat_ideas_update_conditions.py`（单条/多条/清空/清空再设/与 `--status` 组合/未知 id 报错/无 flag 报 usage）。
- [x] T3.4.2.3 `chat/tools._surface_activation_alerts(session, hits)` 在 `/search` 和 LLM `search_arxiv` 工具的渲染表格之后跑一遍：拿出所有 `shelved`/`waiting` 且有 `activation_conditions` 的 idea，逐 hit 做 `condition.lower() in (title + abstract).lower()` 的子串匹配；命中后一行 `- <arxiv_id> "<title>" matches condition "<phrase>" on <idea title> — /idea show <prefix>`，最多渲染 3 条。同时把摘要塞进 working memory 让 LLM 续聊也看得到。任何异常 swallowed（搜索本身不能挂）。设计理由：用户给的条件几乎都是字面术语（"FineWeb-Edu dataset"、"Llama-3.5 release"），子串匹配召回足够，省掉一次 LLM round-trip。10 个单测见 `tests/unit/test_chat_activation_alerts.py`（title/abstract 命中 / 大小写无关 / inactive 状态过滤 / 无条件 idea 无 banner / 空 hits 短路 / 不匹配 silent / 上限 3 条 / 同 (idea, hit) 不重复 / 异常 swallowed / system memory 写入）。
