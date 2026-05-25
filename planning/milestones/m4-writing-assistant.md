> Status: PENDING
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

### Story S4.1.1 — 论文样本导入与解析

**验收条件**:
- `research style train --dir ~/papers/` 批量处理目录下所有 PDF
- `research style train arxiv:xxx arxiv:yyy` 指定 arXiv 论文
- 解析后提取：正文段落、句子列表、章节结构
- 过滤非用户撰写内容（参考文献、公式密集段）

**Tasks**:
- [ ] T4.1.1.1 实现 `style train` CLI 命令（支持目录和 arXiv ID）
- [ ] T4.1.1.2 实现样本预处理：提取可用于风格分析的文本段落
- [ ] T4.1.1.3 实现内容过滤：识别并排除参考文献段、公式密集段
- [ ] T4.1.1.4 将样本存入 `style_samples` 表，关联来源论文

### Story S4.1.2 — 风格指纹分析

**验收条件**:
- 提取宏观层面指纹：摘要结构模式、Introduction 叙事弧线、Related Work 组织方式
- 提取微观层面指纹：平均句长、过渡词偏好（however/yet/while...）、置信度表达频率
- 提取个人标识：章节命名习惯、图表引用格式
- 指纹存储为 `~/.research-agent/style/fingerprint.json`

**Tasks**:
- [ ] T4.1.2.1 实现 `StyleAnalyzer.analyze_macro(papers) → MacroFingerprint`（LLM 分析宏观模式）
- [ ] T4.1.2.2 实现 `StyleAnalyzer.analyze_micro(papers) → MicroFingerprint`（统计分析：句长分布、词频）
- [ ] T4.1.2.3 实现 `StyleAnalyzer.extract_markers(papers) → PersonalMarkers`（章节命名、引用格式）
- [ ] T4.1.2.4 实现指纹序列化：写入 `fingerprint.json`
- [ ] T4.1.2.5 单元测试：解析样本论文，验证指纹字段非空

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

### Story S4.2.1 — 章节草稿生成

**验收条件**:
- `research write abstract` / `write introduction` / `write related_work` 等命令
- 每次生成 3 个版本，标注各版本的风格差异（如："版本A更简洁，版本B更技术性"）
- 每个版本约 300-500 词（可配置）
- 生成速度：< 30s / 版本

**Tasks**:
- [ ] T4.2.1.1 设计 Scribe System Prompt（基于风格指纹注入写作约束）
- [ ] T4.2.1.2 实现 `Scribe.generate(section, context, fingerprint) → Draft[]`
- [ ] T4.2.1.3 实现 `write` CLI 命令，支持所有标准章节类型
- [ ] T4.2.1.4 实现多版本并行生成（asyncio.gather，3个LLM并发调用）
- [ ] T4.2.1.5 实现版本差异标注：调用 LLM 概括各版本特点

### Story S4.2.2 — 上下文感知写作

**验收条件**:
- 写作时可提供研究上下文：`research write introduction --context "这篇论文研究..."`
- 自动从记忆库中调取相关论文和 Idea 作为写作素材
- 生成内容与用户已有内容保持一致（避免重复或矛盾）

**Tasks**:
- [ ] T4.2.2.1 实现上下文构建：从 MemoryKeeper 检索相关材料
- [ ] T4.2.2.2 实现一致性检查：检测与已有草稿的矛盾
- [ ] T4.2.2.3 实现 `--context` 参数：接受自然语言描述

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
