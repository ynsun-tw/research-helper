# M1 验收清单 — MVP: 论文深度理解

> 对应：[milestones/m1-paper-understanding.md](milestones/m1-paper-understanding.md)  
> Epic 范围：E1.1 – E1.5（全部 DONE）  
> **M1 验收状态：已通过（2026-05-25）**  
> 建议验收人：开发者自测 + 可选 1 名目标用户（博士生/独立研究者）

---

## 一、M1 总体验收标准（必须全部通过）

| # | 标准 | 验收命令 / 方法 | 通过 □ |
|---|------|-----------------|--------|
| M1-1 | `research read <paper>` 产出 Analyst + Critic 双视角分析 | 见 [§三.1](#31-research-read) | |
| M1-2 | `research discuss` 可进行基础批判性对话 | 见 [§三.2](#32-research-discuss) | |
| M1-3 | `research config` 可配置 API key | 见 [§三.0](#30-环境与配置) | |
| M1-4 | 基础会话内记忆正常工作 | 见 [§三.2](#32-research-discuss) + [§四](#四数据与持久化) | |

**M1 通过条件**：上表 4 项均为 ☑，且 [§二](#二自动化门禁) 全绿。

---

## 二、自动化门禁

在项目根目录执行（需已 `pip install -e ".[dev]"`）：

```bash
cd /path/to/research-bot
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src
.venv/bin/pytest -q
```

| 检查项 | 预期 | 通过 □ |
|--------|------|--------|
| Ruff lint | 0 errors | |
| Ruff format | 已格式化 | |
| Mypy | Success on `src/` | |
| Pytest | 全部通过（当前基线 ≥ 87） | |

---

## 三、手动冒烟测试

### 3.0 环境与配置

**前置**

- [ ] Python ≥ 3.11
- [ ] 已安装：`pip install -e ".[dev]"`
- [ ] OpenRouter API key（`sk-or-...`）已就绪

**配置**

```bash
research --help
research config set api_key <your-openrouter-key>
research config set model deepseek/deepseek-chat   # 或任意 OpenRouter model slug
research config show
```

| # | 检查点 | 预期 | 通过 □ |
|---|--------|------|--------|
| C-1 | `research --help` | 显示 `read`、`discuss`、`config` | |
| C-2 | `config show` 中 `api_key` | 脱敏显示（非完整 key） | |
| C-3 | `config show` 中 `base_url` | `https://openrouter.ai/api/v1`（OpenRouter key 时应自动对齐） | |
| C-4 | 配置文件权限 | `ls -l ~/.research-agent/config.yaml` → `-rw-------` (600) | |
| C-5 | 无 key 时 `research read` | 友好错误 + exit 1，提示配置 key | |

---

### 3.1 `research read`

**arXiv 论文（需网络）**

```bash
research read arxiv:1706.03762
```

| # | 检查点 | 预期 | 通过 □ |
|---|--------|------|--------|
| R-1 | 进度提示 | 出现 Loading → Parse → Analyze → Save 阶段 | |
| R-2 | 输出分区 | 含 **Analyst**、**Critic**、**Synthesis** 三块（Rich 面板） | |
| R-3 | Analyst 内容 | 有贡献/方法/影响等结构化要点，非空 | |
| R-4 | Critic 内容 | 有支持度评分（1–9，不为 10）及反对点或诚实说明 | |
| R-5 | Synthesis | 有综合摘要；可有 Consensus / Conflicts | |
| R-6 | 耗时 | 单次分析在合理范围（规划目标 &lt; 60s，视模型与网络而定） | |
| R-7 | PDF 缓存 | `~/.research-agent/cache/papers/1706.03762.pdf` 存在 | |
| R-8 | 再次读取同 ID | 不重复下载（或明显更快） | |

**本地 PDF**

```bash
research read /path/to/your.pdf
```

| # | 检查点 | 预期 | 通过 □ |
|---|--------|------|--------|
| R-9 | 本地路径 | 成功解析并输出双视角分析 | |
| R-10 | 无效路径 | 友好错误，非 Python 堆栈 | |

**数据库**

```bash
sqlite3 ~/.research-agent/memory.db "SELECT id, title, analyst_notes IS NOT NULL FROM papers ORDER BY created_at DESC LIMIT 3;"
```

| # | 检查点 | 预期 | 通过 □ |
|---|--------|------|--------|
| R-11 | `papers` 表 | 有对应 `id`（`arxiv:...` 或 `local:...`） | |
| R-12 | 分析结果 | `analyst_notes` / `critic_notes` 非空 JSON | |

---

### 3.2 `research discuss`

```bash
research discuss
# 或带开场：research discuss --topic "Transformer 是否过拟合合成数据？"
```

在 REPL 中：

1. 输入 2–3 轮研究问题（非 `exit`）
2. 输入 `exit` 或 `quit` 退出

| # | 检查点 | 预期 | 通过 □ |
|---|--------|------|--------|
| D-1 | 每轮回复 | 同时出现 **Analyst** + **Critic** 两个面板 | |
| D-2 | 多轮连贯性 | 后续回复能体现前文话题（工作记忆注入） | |
| D-3 | `exit` / `quit` | 正常退出 | |
| D-4 | Ctrl+C | 提示保存并退出，不崩溃 | |
| D-5 | 退出提示 | `Session saved (N turn(s), M message(s), id=...)` | |
| D-6 | Critic 规则 | 评分 ≤ 9；低分时有理由 | |

**会话持久化**

```bash
sqlite3 ~/.research-agent/memory.db \
  "SELECT role, substr(content,1,40) FROM discussions ORDER BY created_at DESC LIMIT 10;"
```

| # | 检查点 | 预期 | 通过 □ |
|---|--------|------|--------|
| D-7 | `discussions` 表 | 含 `user`、`analyst`、`critic` 角色记录 | |
| D-8 | 消息数量 | 每轮 user + analyst + critic（约 3×轮次） | |

---

## 四、数据与持久化

| 路径 | 用途 | 存在 □ |
|------|------|--------|
| `~/.research-agent/config.yaml` | 配置 | |
| `~/.research-agent/memory.db` | SQLite（papers / discussions / ideas 等） | |
| `~/.research-agent/cache/papers/` | PDF 缓存 | |

**隐私**：确认未将 `config.yaml` 或含 key 的文件提交到 git。

---

## 五、Epic 追溯（可选抽查）

| Epic | 核心能力 | 快速验证 | 通过 □ |
|------|----------|----------|--------|
| E1.1 | 工程脚手架 | §二 自动化 + `research --help` | |
| E1.2 | PDF + SQLite Paper | `research read` + `papers` 表 | |
| E1.3 | Analyst / Critic / Orchestrator | 双视角输出 + 并行分析 | |
| E1.4 | WorkingMemory | `discuss` 多轮 + `discussions` 表 | |
| E1.5 | CLI 完整接口 | `read` / `discuss` / Rich 输出 | |

---

## 六、已知限制（M1 不阻塞验收）

- 向量库 / Chroma、Searcher、Scribe、TUI：未实现（后续 Epic）
- PDF 章节解析为启发式，复杂版式可能不准
- Token 估算为 `len/4` 近似，非精确 tiktoken
- `research read` 对超长论文可能受模型上下文限制（见 planning/notes 风险）

---

## 七、验收结论

| 项目 | 结果 |
|------|------|
| 验收日期 | 2026-05-25 |
| 验收人 | 项目维护者 |
| 自动化门禁 | ☑ 通过 ☐ 未通过 |
| M1-1 ~ M1-4 | ☑ 通过 ☐ 未通过 |
| **M1 里程碑** | ☑ **通过，M1 已关闭** ☐ 不通过 |

**未通过项记录**：无。

---

## 八、通过后操作（已完成）

1. ☑ [milestones/m1-paper-understanding.md](milestones/m1-paper-understanding.md) → **DONE**
2. ☑ [PLAN.md](PLAN.md) 中 M1 → **DONE**
3. 可选 git tag：`git tag -a m1-mvp -m "M1 paper understanding MVP"`
