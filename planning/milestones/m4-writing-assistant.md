> Status: IN PROGRESS
> Index: [../../PLAN.md](../../PLAN.md)

# M4 — 论文写作助手

**目标用户**: 有已发表论文、需要在写作阶段获得支持的研究者
**交付物**: Scribe Agent（风格模仿 + 多版本生成）+ 写作审查流水线
**时间**: 2 周
**前置条件**: M3 完成；用户需提供至少 2 篇已发表论文用于风格学习
**验收标准**:
- `research style train --dir ~/papers/` 能成功提取写作风格指纹
- `research write abstract` 生成 3 个风格不同的摘要版本供选择
- 写作审查流水线：Scribe 生成 → Analyst+Critic 审查 → Scribe 修改
- 从用户修改中持续学习，风格指纹随使用更新

---

## Epic E4.1 — 写作风格指纹提取

> 价值假设：精准的风格指纹是 Scribe 产出"读起来像用户写的"内容的关键

### Story S4.1.1 — 论文样本导入与解析 [x] COMPLETED

**验收条件**:
- `research style train --dir ~/papers/` 批量处理目录下所有 PDF
- `research style train arxiv:xxx arxiv:yyy` 指定 arXiv 论文
- 解析后提取：正文段落、句子列表、章节结构
- 过滤非用户撰写内容（参考文献、公式密集段）

**Tasks**:
- [x] T4.1.1.1 实现 `style train` CLI 命令（支持目录和 arXiv ID）—
  `research style train [sources...] [--dir <path>] [--append]`，加上
  `research style show` 查看现有语料。
- [x] T4.1.1.2 实现样本预处理：提取可用于风格分析的文本段落 —
  `style/extractor.py` 处理 PyMuPDF 软换行 / 连字符换行，按空行切段。
- [x] T4.1.1.3 实现内容过滤：识别并排除参考文献段、公式密集段 —
  `style/filters.py` 黑名单 section + 数学密度 + 字母密度 + 长度/句数。
- [x] T4.1.1.4 将样本存入 `style_samples` 表，关联来源论文 —
  新增 SQLite schema + `StyleSampleRepository`（bulk_add / 按 paper 计数 /
  按 paper 替换）。单元测试覆盖过滤、抽取、CRUD、CLI dispatch。

### Story S4.1.2 — 风格指纹分析 [x] COMPLETED

**验收条件**:
- 提取宏观层面指纹：摘要结构模式、Introduction 叙事弧线、Related Work 组织方式
- 提取微观层面指纹：平均句长、过渡词偏好（however/yet/while...）、置信度表达频率
- 提取个人标识：章节命名习惯、图表引用格式
- 指纹存储为 `~/.research-agent/style/fingerprint.json`

**Tasks**:
- [x] T4.1.2.1 实现 `StyleAnalyzer.analyze_macro(samples) → MacroFingerprint` —
  按 section（Abstract / Introduction / Related Work / Other）分桶后，
  统计 abstract & intro 最常见开头、相关工作组织策略
  （chronological / thematic / comparison）、每篇平均 section 数。
  当前为纯启发式实现；LLM augmentation 留给后续 story。
- [x] T4.1.2.2 实现 `StyleAnalyzer.analyze_micro(samples) → MicroFingerprint` —
  句长分布（mean / median / p10 / p90）、段落句数、过渡词
  per-100-sentences rate、hedging / confidence / passive 频率、
  type-token ratio。
- [x] T4.1.2.3 实现 `StyleAnalyzer.extract_markers(samples) → PersonalMarkers` —
  最常用 section title、引用格式（latex_cite / bracket_num /
  author_year / mixed）、Figure vs Fig. / Table vs Tab.、em-dash 频率。
- [x] T4.1.2.4 实现指纹序列化：`Fingerprint.save_to / load_from`，
  默认路径 `~/.research-agent/style/fingerprint.json`，宽松
  `from_dict` 容忍前后向不兼容字段。
- [x] T4.1.2.5 单元测试：`tests/unit/test_style_fingerprint.py`、
  `tests/unit/test_style_analyzer.py` 共 19 用例，覆盖 round-trip、
  empty corpus、宏观/微观/markers 抽取、CLI dispatch。

### Story S4.1.3 — 风格指纹持续学习

**验收条件**:
- 用户修改 Scribe 生成的内容后，差异自动记录为学习样本
- `research style update` 触发指纹重新计算（增量更新）
- 指纹版本管理：每次更新保留历史版本

**Tasks**:
- [ ] T4.1.3.1 实现用户修改追踪：保存 `(original, revised)` 对
- [ ] T4.1.3.2 实现增量指纹更新：从修改差异中提取新的风格信号
- [ ] T4.1.3.3 实现指纹版本历史（fingerprint_v1.json, v2.json...）

---

## Epic E4.2 — Scribe Agent

> 价值假设：风格模仿写作让用户感觉是自己在写，而非 AI 替代

### Story S4.2.1 — 章节草稿生成 [x] COMPLETED

**验收条件**:
- `research write abstract` / `write introduction` / `write related_work` 等命令
- 每次生成 3 个版本，标注各版本的风格差异（如："版本A更简洁，版本B更技术性"）
- 每个版本约 300-500 词（可配置）
- 生成速度：< 30s / 版本

**Tasks**:
- [x] T4.2.1.1 设计 Scribe System Prompt（基于风格指纹注入写作约束）—
  `prompts/scribe.yaml`：硬约束（模仿指纹微统计 / 标记、不编造数字
  /citation key、不要 self-reference），JSON 输出 `{draft, style_note}`。
- [x] T4.2.1.2 实现 `Scribe.generate(section, context, fingerprint) → Draft[]` —
  `agents/scribe.py`，新增 `Draft` dataclass、`normalize_section`、
  `_build_user_prompt`、`_format_fingerprint`、`_parse_draft`。
- [x] T4.2.1.3 实现 `write` CLI 命令，支持所有标准章节类型 —
  `research write <section> [--context] [--words] [--versions]
  [--output] [--sequential]`，支持别名（intro / methods /
  experiments…）。
- [x] T4.2.1.4 实现多版本并行生成（ThreadPoolExecutor，3 个 LLM 并发调用）—
  默认 parallel=True，tests 通过 `parallel=False` 走顺序路径以
  绕开 MockLLMProvider 的非线程安全。
- [x] T4.2.1.5 实现版本差异标注：每个 variant 由不同的 directive
  驱动（concise / technical depth / narrative arc），LLM 在
  `style_note` 字段里自报。22 unit tests 覆盖 dispatch、解析、
  fingerprint 注入、降级、CLI 输出文件。

### Story S4.2.2 — 上下文感知写作 [x] COMPLETED

**验收条件**:
- 写作时可提供研究上下文：`research write introduction --context "这篇论文研究..."`
- 自动从记忆库中调取相关论文和 Idea 作为写作素材
- 生成内容与用户已有内容保持一致（避免重复或矛盾）

**Tasks**:
- [x] T4.2.2.1 实现上下文构建：从 MemoryKeeper 检索相关材料 —
  `cli_write._build_writing_context` 在 `--context` 文本非空时
  调用 `MemoryKeeper.check_associations`（threshold=0.5、无状态
  过滤，取前 3 个 idea）+ `recall_history`（取前 3 条跨 session
  discussion），并将所有错误降级为空槽位以保护写作流程。
- [x] T4.2.2.2 实现一致性检查：检测与已有草稿的矛盾 —
  `--check-against PATH`（可重复）参数读取已有草稿正文，
  在 prompt 里以 `[EXISTING DRAFTS TO STAY CONSISTENT WITH —
  do not contradict, do not duplicate verbatim]` 段落注入；
  单文件截断 2 KB，避免 LLM 上下文溢出。
- [x] T4.2.2.3 实现 `--context` 参数：接受自然语言描述 —
  CLI 已在 S4.2.1 接入；本 story 把它从「dump 给 LLM」
  升级为「先用作 memory recall 的 query、再合成结构化
  context blob」。`WritingContext.render` 用清晰段头
  （USER CONTEXT / RELATED IDEAS / RECENT DISCUSSION EXCERPTS /
  EXISTING DRAFTS）让 LLM 容易解析。10 unit tests 覆盖 render
  empty / full、truncation、memory 错误吞咽、`run_write` 端到端。

---

## Epic E4.3 — 写作审查流水线

> 价值假设：Scribe → Analyst+Critic → Scribe 的循环让写作质量迭代提升

### Story S4.3.1 — 自动审查循环

**验收条件**:
- Scribe 生成草稿后，自动触发 Analyst 和 Critic 并行审查
- Analyst 检查：论点是否充分、与 related work 的区分是否清晰
- Critic 检查：是否有过度声明（overclaim）、实验是否支撑结论
- Scribe 根据审查意见生成修订版本

**Tasks**:
- [ ] T4.3.1.1 实现 `Orchestrator.writing_review_pipeline(draft) → ReviewedDraft`
- [ ] T4.3.1.2 实现 Analyst 写作审查模式（`analyze_writing` vs `analyze_paper`，Prompt 不同）
- [ ] T4.3.1.3 实现 Critic 写作审查模式（专注于 overclaim 和逻辑漏洞）
- [ ] T4.3.1.4 实现 Scribe 修订：基于审查意见生成改进版本
- [ ] T4.3.1.5 集成测试：端到端写作审查流程

### Story S4.3.2 — 用户选择与反馈

**验收条件**:
- 展示原版 + 修订版，用户选择采纳哪些修改
- 支持部分采纳（接受 Critic 的修改但拒绝 Analyst 的建议）
- 选择结果作为风格学习样本（E4.1.3 的输入）

**Tasks**:
- [ ] T4.3.2.1 实现 diff 视图：展示修订前后的差异
- [ ] T4.3.2.2 实现分项选择：用户逐条决定是否采纳审查意见
- [ ] T4.3.2.3 将用户选择同步到风格学习系统

---

## Epic E4.4 — Self-Plagiarism 检测

> 价值假设：自动检测避免研究者无意中复用过多已发表内容

### Story S4.4.1 — 相似度检测

**验收条件**:
- 生成内容与用户已发表论文进行相似度比对
- 相似度 > 40% 时给出警告
- 标注具体的重叠段落，建议改写方向

**Tasks**:
- [ ] T4.4.1.1 实现 `PlagiarismDetector.check(draft, published_papers) → SimilarityReport`
- [ ] T4.4.1.2 实现段落级相似度（TF-IDF + 向量余弦相似度）
- [ ] T4.4.1.3 实现警告展示：高亮重叠段落 + 改写建议
- [ ] T4.4.1.4 集成测试：用户论文的某段落故意复用，验证能被检测
