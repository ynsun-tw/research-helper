> Status: PENDING
> Index: [../../PLAN.md](../../PLAN.md)

# M2 — Idea 讨论工坊

**目标用户**: 需要深度批判性对话的研究者
**交付物**: 完整的辩论模式、Critic 评分系统、Idea 生命周期管理
**时间**: 1 周
**前置条件**: M1 完成（Analyst、Critic、基础 CLI 可用）
**验收标准**:
- `research discuss "我的 idea"` 触发辩论模式，输出支持/反对双视角
- Critic 评分稳定输出 1-10 分，低于 7 分有理由说明
- Idea 可保存、查看、更新状态
- 对话中能看到与历史讨论的关联提醒

---

## Epic E2.1 — 辩论模式

> 价值假设：强制双视角辩论防止 AI 迎合，是产品核心护城河

### Story S2.1.1 — 结构化辩论输出格式

**验收条件**:
- 每次 Idea 讨论输出固定结构：支持论点 + 反对论点 + 建议 + 评分
- Analyst 不允许只说"good idea"，必须给出具体依据
- Critic 必须给出至少 1 条有依据的反对观点
- 输出格式可通过 Rich 渲染为彩色分区

**Tasks**:
- [ ] T2.1.1.1 设计辩论输出 schema（DebateResult: supports, objections, suggestions, score, confidence）
- [ ] T2.1.1.2 更新 Analyst System Prompt：Idea 讨论时要求具体论据
- [ ] T2.1.1.3 更新 Critic System Prompt：禁止"我同意你的观点"式回复
- [ ] T2.1.1.4 实现 `Orchestrator.debate_idea(idea_text) → DebateResult`
- [ ] T2.1.1.5 单元测试：验证输出 schema 字段完整性

### Story S2.1.2 — 多轮辩论迭代

**验收条件**:
- 用户可以针对某个反对点继续追问
- Analyst 可以回应 Critic 的反对（调整支持立场）
- Critic 可以坚守或调整评分（需说明理由）
- 每轮保留完整辩论上下文

**Tasks**:
- [ ] T2.1.2.1 实现辩论会话状态：持有 DebateHistory（每轮的 DebateResult）
- [ ] T2.1.2.2 实现追问模式：用户输入针对特定 Agent 的追问
- [ ] T2.1.2.3 实现评分变化追踪：记录 Critic 评分跨轮次的变化
- [ ] T2.1.2.4 集成测试：3 轮辩论流程，验证上下文连贯性

---

## Epic E2.2 — Critic 评分系统

> 价值假设：可量化的评分给用户直观的质量信号

### Story S2.2.1 — 评分校准机制

**验收条件**:
- 评分严格在 1-10 范围内（10 分被 Prompt 约束禁止）
- 7 分以下必须包含具体反对理由
- 评分不随用户情绪变化（防止 AI 迎合）
- 用户可给评分打标记（"我觉得你高估了"），系统记录但不立即改变

**Tasks**:
- [ ] T2.2.1.1 在 Critic System Prompt 中实现评分量表（1-3/4-6/7-8/9/禁10）
- [ ] T2.2.1.2 实现评分解析器：稳定从 LLM 输出提取数字（处理 "7/10", "7 out of 10" 等格式）
- [ ] T2.2.1.3 实现用户评分反馈记录（不影响当前评分，写入 metadata）
- [ ] T2.2.1.4 单元测试：评分解析边界情况

### Story S2.2.2 — 评分历史与趋势

**验收条件**:
- 同一 Idea 的多次评分可查看历史
- `research ideas show <id>` 显示评分变化趋势
- Memory Keeper 能检测到某 Idea 评分持续低于 5 时主动提示

**Tasks**:
- [ ] T2.2.2.1 在 `ideas` 表新增 `score_history` 字段（JSON array）
- [ ] T2.2.2.2 实现评分历史更新逻辑（每次讨论追加）
- [ ] T2.2.2.3 实现评分趋势查询接口

---

## Epic E2.3 — Idea 生命周期管理

> 价值假设：系统化管理 Idea 状态让研究者保持清晰的研究全局视图

### Story S2.3.1 — Idea CRUD 与状态机

**验收条件**:
- 讨论结束后系统提示是否保存为 Idea
- Idea 状态：`active → shelved/experimenting/abandoned/completed`
- `research ideas list` 按状态分组展示
- `research ideas update <id> --status shelved` 手动更新状态

**Tasks**:
- [ ] T2.3.1.1 实现 `IdeaRepository`：CRUD 操作
- [ ] T2.3.1.2 实现状态机：合法状态转换（active → shelved 允许，completed → active 不允许）
- [ ] T2.3.1.3 实现 `ideas` CLI 子命令（list/show/update/archive）
- [ ] T2.3.1.4 在 `discuss` 命令结束时提示保存 Idea
- [ ] T2.3.1.5 集成测试：完整状态转换流程

### Story S2.3.2 — 讨论脉络追踪

**验收条件**:
- 每条 Idea 关联其全部历史讨论记录
- `research ideas show <id>` 展示 Idea 的完整讨论脉络
- Memory Keeper 在新讨论开始时主动提醒关联的历史 Idea

**Tasks**:
- [ ] T2.3.2.1 在讨论表新增 `idea_id` 外键
- [ ] T2.3.2.2 实现讨论与 Idea 的关联逻辑（创建/关联已有 Idea）
- [ ] T2.3.2.3 实现 Memory Keeper 基础版：在会话开始时检索相似历史（向量相似度）
- [ ] T2.3.2.4 实现 ChromaDB 向量存储：Idea 描述向量化存储
- [ ] T2.3.2.5 集成测试：相似 Idea 检测（给出一个类似的 Idea，验证能找到历史记录）
