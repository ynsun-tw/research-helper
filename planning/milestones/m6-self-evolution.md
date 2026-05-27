> Status: PLANNED
> Index: [../../PLAN.md](../../PLAN.md)

# M6 — 从用户反馈学习（Self-Evolution）

**目标用户**: 已经用了一段时间、`draft_revisions` / `score_history` /
`research_insights` 里有积累数据的研究者
**交付物**: 让 Critic / Scribe / Searcher 三个核心 agent 的行为随用户使用而漂移；
配套 `evolution status` / `evolution reset` 观测与回退入口
**时间**: 1.5 周（Sprint 8）
**前置条件**: M4 完成（draft_revisions 表正在产出数据）；M5 完成（research_insights
能聚合数据）；用户已经积累至少 ~10 条 draft revision 或 ~5 个 idea 的多轮评分
（少于此则学习信号噪声 > 信号，工具默认跳过）

**验收标准**:
- 用户反复拒绝某类 Critic 反馈（例如"被动语态太多"）后，下一次
  `revise_draft` 时 Critic 的同类反馈出现频率显著下降
- 用户对若干 idea 给出 `user_score_feedback`（例如"Critic 判得太低"）后，
  后续 `discuss_idea` 时 Critic 评分基线偏移可被观测且可重置
- 用户最近 30 天阅读偏向 applied 论文时，`search_arxiv` 不显式 mode 的查询
  默认偏 applied
- `research evolution status` 一屏展示当前所有正在生效的学习信号 + 数据来源
  + 上次更新时间；`research evolution reset <component>` 能清空某一类
- 自然语言入口可触达：`evolution_status` / `evolution_reset` 两个 LLM tools

## 设计原则

1. **数据已采集**：M4/M5 的所有 schema 字段都早就在写。M6 不引入新表，只
   读现有数据并把它转成 agent prompt 上下文。
2. **可解释**：任何学习产生的行为漂移都能被 `evolution status` 列出，
   附"基于哪些数据点"的引用。模型不该悄悄变。
3. **可回退**：每个学习组件都有独立的 reset；用户怀疑 agent "学坏了"时能
   一键回到出厂行为。
4. **保守优于激进**：单点反馈不足时不学（默认阈值见各 story）。
5. **不引入训练**：学习信号通过 **system prompt 注入** 实现，不微调权重、
   不调用外部训练 API。和 R-001 / R-002 的"凡能 prompt 解决就不动架构"原则一致。

---

## Epic E6.1 — Reviewer-Feedback Aware Critic & Scribe

> 价值假设：用户对 Scribe 草稿的修订过程（接受/拒绝 Critic 提出的 issue）
> 是 agent 行为最强的反馈信号。让 Critic 不再反复提同类被拒反馈、让
> Scribe 在初稿阶段就规避高拒绝率模式。

### Story S6.1.1 — Reviewer-feedback 聚合与归类 [ ] PENDING

**验收条件**:
- 新增 `agents/feedback_analyzer.py`：从 `draft_revisions` 表读所有
  `selected_issues` / `rejected_issues` / `selected_suggestions` /
  `rejected_suggestions`
- 按主题归类（heuristic：关键词桶 — 语态 / 句长 / 引用 / 术语 / 结构 / 数据 / 其他）
- 输出 `FeedbackProfile`：每类的 accept_count / reject_count / 最近 N 条原文样例
- 阈值：单类 < 3 条数据时不进 profile（噪声过滤）

**Tasks**:
- [ ] T6.1.1.1 实现关键词分类（不调 LLM，纯本地 heuristic）—
  `feedback_analyzer.classify(issue: str) -> Topic`，初版手工词典，
  覆盖 7 个 topic + `unknown` fallback
- [ ] T6.1.1.2 实现 `aggregate(revisions: list[DraftRevision]) -> FeedbackProfile`，
  返回 `dict[Topic, FeedbackStats]` 含 accept/reject 计数 + samples
- [ ] T6.1.1.3 实现 `FeedbackProfile.to_prompt_hint() -> str`，生成
  ≤200 字的 prompt 片段，例如 "User has rejected 4 of 5 'passive voice'
  suggestions in past sessions; avoid raising it unless severe."
- [ ] T6.1.1.4 单元测试：阈值过滤、关键词分类边界、prompt hint 渲染、
  空数据 / 单类数据稳健

### Story S6.1.2 — Critic 读 FeedbackProfile [ ] PENDING

**验收条件**:
- `Critic.critique(...)` 在构建 system prompt 时附上当前 `FeedbackProfile` 的
  prompt hint（如果有 ≥3 条数据的主题）
- Critic 不会"自我审查到完全沉默"——hint 是软约束（`avoid unless severe`），
  不是禁令
- `research style show` 输出底部新增一行 "Active reviewer-feedback profile:
  N topics, M data points"

**Tasks**:
- [ ] T6.1.2.1 在 `agents/critic.py` 的 review prompt 流程里加可选
  `feedback_profile_hint` 参数（None 时行为不变）
- [ ] T6.1.2.2 `cli_review.run_review` / `WritingPipeline` 调用前从 DB
  读 profile 并注入
- [ ] T6.1.2.3 集成测试：构造 5 条全拒"被动语态"的 revisions → Critic
  的 next review 在同主题上的 issue 输出概率显著下降（用 MockLLMProvider
  + prompt 检查，验证 hint 出现在 system 里）
- [ ] T6.1.2.4 更新 `style_show` 服务，输出 profile summary

### Story S6.1.3 — Scribe 规避高拒绝率模式 [ ] PENDING

**验收条件**:
- `Scribe.draft_section(...)` 同样在 prompt 里附 profile hint，但措辞反向：
  "User strongly prefers active voice in past revisions; default to active."
- Scribe 不会因 profile 缺数据而行为退化（fallback 到原 fingerprint-only）

**Tasks**:
- [ ] T6.1.3.1 实现 `FeedbackProfile.to_scribe_hint() -> str`，与 Critic 方向相反
- [ ] T6.1.3.2 修改 `Scribe._build_user_prompt` / `_build_system_prompt`
  接受 profile hint
- [ ] T6.1.3.3 测试：相同 input + 不同 profile → prompt 内容差异，且差异
  落在 hint 段而非 fingerprint 段

---

## Epic E6.2 — Critic 评分校准

> 价值假设：用户的 `user_score_feedback`（"Critic 把这一点判低了"）+
> `score_history` 的纵向轨迹，是个人评分基线的良好代理。Critic 用统一基线
> 给所有用户打分会让重度用户感觉"它从来不打超过 6 分"。

### Story S6.2.1 — Score baseline 聚合 [ ] PENDING

**验收条件**:
- 新增 `agents/calibration.py::compute_baseline(ideas: list[Idea]) -> ScoreBaseline`
- baseline 包含：用户 idea 的 `critic_score` 分布（mean / median / p25 / p75），
  以及 `user_score_feedback` 里出现"too low" / "too high" 关键词的次数
- 阈值：少于 5 个有评分的 idea 时返回 `ScoreBaseline.empty()`

**Tasks**:
- [ ] T6.2.1.1 实现纯统计聚合，不调 LLM
- [ ] T6.2.1.2 `ScoreBaseline.to_prompt_hint() -> str`，例如
  "Your scoring distribution: median 6, p75 7.5. User flagged 3 reviews as
  'too low'; bias slightly more generous when evidence supports it."
- [ ] T6.2.1.3 测试：empty 阈值、关键词识别（中英文）、分布稳健

### Story S6.2.2 — Critic 读 baseline [ ] PENDING

**验收条件**:
- `Critic.critique_idea` 在 system prompt 末尾追加 baseline hint（如果非
  empty）
- baseline hint 不能压过"不允许打 10 分"等核心约束 — system prompt 注入
  顺序：核心规则 → baseline hint → 任务

**Tasks**:
- [ ] T6.2.2.1 在 `agents/critic.py` 的 idea critique prompt 里集成
- [ ] T6.2.2.2 由 `Orchestrator.debate_round` 提供 baseline（从 IdeaRepository
  统计）
- [ ] T6.2.2.3 端到端 MockLLM 测试验证 baseline 出现且顺序正确

---

## Epic E6.3 — Searcher 偏好学习

> 价值假设：用户 30 天阅读模式是下一次默认 search mode 的最佳预测器。
> 不显式 mode 时的 `search_arxiv` 调用应该向偏好倾斜。

### Story S6.3.1 — 从 reading history 推断 mode 偏好 [ ] PENDING

**验收条件**:
- 复用 `agents/meta_memory.py` 的 30 天活动统计
- 新增 `infer_search_mode_bias(window_days: int) -> SearchBias`
- SearchBias 字段：`theoretical_ratio`、`applied_ratio`、最常合作的几个
  作者名（如果 `group:` mode 数据足够）
- 阈值：少于 5 篇已读论文时返回 `SearchBias.neutral()`

**Tasks**:
- [ ] T6.3.1.1 用 Pydantic 给 paper 的 abstract 跑一次轻量分类
  （theoretical / applied / mixed）— 只在 `infer_search_mode_bias` 第一次
  被调用 + cache miss 时进行；结果 cache 到 papers 表新增列 `paper_kind`
- [ ] T6.3.1.2 SearchBias.to_default_mode() → Optional[str]，返回当 ratio
  差 > 0.6 时的推荐 mode

### Story S6.3.2 — Searcher 用 default mode [ ] PENDING

**验收条件**:
- `search_arxiv(query, mode=None)` 在 mode 缺省时调 `infer_search_mode_bias`
- LLM 看到的工具响应顶部包含 "ℹ️ defaulted to mode=applied based on
  your recent reading (8/12 papers in last 30 days)"，告诉用户为什么
- 用户显式给 mode 时此逻辑完全跳过

**Tasks**:
- [ ] T6.3.2.1 修改 `chat/tools.py::exec_search_arxiv` 在 args 里没 mode 时
  调 bias 推断
- [ ] T6.3.2.2 在工具结果开头加 disclosure 行（透明性硬要求，见设计原则 2）
- [ ] T6.3.2.3 测试：bias 工作 / 用户显式 mode 覆盖 / SearchBias.neutral
  跳过

---

## Epic E6.4 — 自我进化观测与治理

> 价值假设：用户必须能看到"agent 因为我学了什么"且能一键回退。否则
> "感觉它最近不太好用了"会变成黑盒债务。

### Story S6.4.1 — `research evolution status` CLI [ ] PENDING

**验收条件**:
- `research evolution status` 输出 Markdown 表格，三列：
  组件 / 数据来源 / 当前活动状态
- 涵盖：FeedbackProfile / ScoreBaseline / SearchBias / Fingerprint history
  count
- 每行包含"已收集 N 条 / 阈值 M / active=Y/N"

**Tasks**:
- [ ] T6.4.1.1 新增 `cli_evolution.py::run_evolution_status`
- [ ] T6.4.1.2 Typer 子命令 `research evolution status`（在 cli.py 新增
  evolution app group）
- [ ] T6.4.1.3 输出格式化（Rich Table）

### Story S6.4.2 — `research evolution reset` CLI [ ] PENDING

**验收条件**:
- `research evolution reset reviewer-feedback`：清除 `draft_revisions` 表中
  `selected_*` / `rejected_*` 列（保留 original/revised，不丢素材）
- `research evolution reset score-baseline`：清除所有 idea 的
  `user_score_feedback`（保留 score_history）
- `research evolution reset search-bias`：清除 `paper_kind` 列、强制下次
  重算
- `research evolution reset all`：以上全部
- 每个 reset 都先 print "About to clear X. Continue? [y/N]"，需要 `--yes`
  跳过

**Tasks**:
- [ ] T6.4.2.1 实现 4 个 reset action
- [ ] T6.4.2.2 Typer 子命令 + 确认 prompt
- [ ] T6.4.2.3 测试：reset 不影响原始数据（reviewer-feedback 保留 original/revised）

### Story S6.4.3 — 自然语言入口 [ ] PENDING

**验收条件**:
- 新 LLM tools：`evolution_status()`（read-only）+ `evolution_reset(component)`
  （**STATE-MUTATING**，按 ADR-008 约定要求 LLM 先 paraphrase + 用户确认）
- "agent 最近是不是有点偏？" / "重置一下你学的东西" 类自然语言能命中

**Tasks**:
- [ ] T6.4.3.1 在 `chat/tools.py` 注册两个工具，executor 直接调
  `run_evolution_status` / `run_evolution_reset`
- [ ] T6.4.3.2 更新 `CHAT_SYSTEM_PROMPT`：把 `evolution_reset` 加入
  STATE-MUTATING 工具列表
- [ ] T6.4.3.3 测试：natural language → tool dispatch + 确认流程

---

## 风险与未决

- **过早学习**：阈值（3 条 reviewer feedback / 5 个 idea / 5 篇 paper）是
  拍脑袋值。可能要观察实际用户数据分布再调整。先发后调。
- **关键词分类局限**：S6.1.1 用纯 heuristic 分类 issue 主题，对中文 / 非
  常见措辞会归到 `unknown`。如果命中率太低，下一轮可以引入小模型分类，但这
  样会破坏"不调 LLM 做学习"的原则。
- **隐私视角**：FeedbackProfile / ScoreBaseline / SearchBias 都是本地数据，
  不上传；但 `evolution status` 在分享 session 截图时可能暴露用户偏好。文档
  里需要提醒。
- **与 ADR-010 的协调**：M6 加的 prompt hint 会增加 `CHAT_SYSTEM_PROMPT` 长度
  ~200-300 tokens（最坏 4 个 hint 同时存在）。这部分应该测算进 `[~N in]`
  telemetry，让用户能直接看到 self-evolution 的 token 代价。

## 完成后启动

- ADR-011 in `notes.md`：把 M6 的设计取舍（不微调权重、prompt 注入而非 RL、
  保守阈值、可重置）记录下来
- README "Self-Evolution" 章节：1 段说明 + `evolution status` 截图 + reset
  方法
- `examples/usage-scenarios-zh.md` 加附录 B：演示一段"agent 因为我反复拒绝
  被动语态建议，下次不再提"的完整 transcript
