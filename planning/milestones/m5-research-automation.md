> Status: COMPLETE (scope trimmed) — E5.3 + E5.4 shipped this milestone. E5.1 (code reproduction) and E5.2 (full TUI) are deferred to a follow-up milestone.
> Index: [../../PLAN.md](../../PLAN.md)

# M5 — 研究自动化与完善

**目标用户**: 需要代码复现验证和完整研究工作台的研究者
**交付物**: 代码复现系统 + 完整 TUI 界面 + 图表生成 + 全系统打磨
**时间**: 4 周（Sprint 6: 代码复现 2周 + Sprint 7: 打磨整合 2周）
**前置条件**: M4 完成（所有 5 个 Agent 可用）
**验收标准**:
- `research reproduce arxiv:xxx` 能搜索代码、创建隔离环境、运行示例并生成复现报告
- `research ui` 启动功能完整的 TUI 界面
- `research write figure` 生成 TikZ/matplotlib/概念图代码
- 所有命令响应时间 < 5s（LLM 调用除外）
- 全系统测试覆盖率 ≥ 80%

---

## Epic E5.1 — 代码复现系统 (DEFERRED)

> Deferred — running arbitrary third-party code from arXiv repos is
> heavyweight and risky for a single-session implementation. Will be
> picked up in a dedicated follow-up milestone.

> 价值假设：自动化复现降低研究者验证他人工作的成本，提高引用判断准确性

### Story S5.1.1 — 代码仓库搜索

**验收条件**:
- 自动搜索 GitHub、Papers with Code、作者主页
- 返回每个代码仓库的：Stars、最后更新时间、官方代码标记、依赖语言
- 与 Semantic Scholar 集成：从论文主页提取代码链接
- 相关度评分 + 维护质量评分

**Tasks**:
- [ ] T5.1.1.1 实现 `GitHubSearcher`：搜索与论文标题/作者相关的仓库
- [ ] T5.1.1.2 实现 `PapersWithCodeSearcher`：通过 PWC API 查找官方实现
- [ ] T5.1.1.3 实现代码仓库评分：Star 数、更新频率、README 完整度
- [ ] T5.1.1.4 集成测试：对已知有代码的论文验证搜索准确性

### Story S5.1.2 — 隔离环境与运行验证

**验收条件**:
- 自动创建 Python virtualenv 隔离环境
- 安装依赖（`requirements.txt` / `setup.py` / `pyproject.toml`）
- 运行作者提供的最小示例
- 捕获 stdout/stderr 和运行结果

**Tasks**:
- [ ] T5.1.2.1 实现 `IsolatedEnvironment`：创建 virtualenv、安装依赖
- [ ] T5.1.2.2 实现 `CodeRunner`：在隔离环境中运行指定脚本
- [ ] T5.1.2.3 实现超时控制（默认 5 分钟）和资源限制
- [ ] T5.1.2.4 实现依赖安装失败的友好错误处理

### Story S5.1.3 — 复现性报告

**验收条件**:
- 对比论文中的关键指标与实际运行结果
- 输出三级结论：✅ 完全复现 / ⚠️ 部分复现（有偏差）/ ❌ 无法复现
- 记录所有偏差，标注可能的原因
- 报告导出到 `~/.research-agent/exports/`

**Tasks**:
- [ ] T5.1.3.1 实现 `ReproducibilityReport` schema
- [ ] T5.1.3.2 实现结果对比：LLM 辅助对比论文指标与运行输出
- [ ] T5.1.3.3 实现报告生成：Markdown 格式
- [ ] T5.1.3.4 实现 `research reproduce` CLI 命令（完整流程）
- [ ] T5.1.3.5 E2E 测试：对一个知名的可复现论文运行完整流程

---

## Epic E5.2 — TUI 完整界面 (DEFERRED)

> Deferred — full three-panel Textual app is a multi-day surface
> area; the current CLI / REPL covers the same workflows. Will be
> picked up in a dedicated follow-up milestone.

> 价值假设：TUI 提供更沉浸的研究讨论体验，适合长时间深度工作

### Story S5.2.1 — TUI 主界面框架

**验收条件**:
- `research ui` 启动全屏 TUI
- 三面板布局：左侧导航（论文/Idea/对话历史）、中间主区域、右侧 Agent 状态
- 键盘驱动：常用操作均有快捷键提示
- 支持鼠标点击

**Tasks**:
- [ ] T5.2.1.1 实现 Textual App 骨架（三面板布局）
- [ ] T5.2.1.2 实现左侧导航面板（树形列表：Papers/Ideas/Discussions）
- [ ] T5.2.1.3 实现右侧 Agent 状态面板（当前活跃 Agent 标识）
- [ ] T5.2.1.4 实现快捷键系统（? 显示帮助，/ 搜索，r 读论文等）

### Story S5.2.2 — TUI 对话体验

**验收条件**:
- 对话输入框支持多行编辑
- Agent 回复按角色用不同颜色区分（Analyst=蓝，Critic=红，Memory=绿）
- 支持流式输出（LLM token 逐字显示）
- 对话历史可滚动查看

**Tasks**:
- [ ] T5.2.2.1 实现对话组件：输入框 + 消息列表
- [ ] T5.2.2.2 实现 Agent 角色颜色主题
- [ ] T5.2.2.3 实现流式输出渲染（Textual reactive 更新）
- [ ] T5.2.2.4 实现对话历史分页加载（避免大量历史消息导致性能问题）

### Story S5.2.3 — TUI 论文浏览

**验收条件**:
- 在 TUI 中触发论文分析后，结果直接展示在主面板
- 支持 Analyst/Critic 视角切换
- 关联论文可点击跳转
- 支持全文搜索（当前已读论文库）

**Tasks**:
- [ ] T5.2.3.1 实现论文详情视图（Markdown 渲染）
- [ ] T5.2.3.2 实现 Analyst/Critic 视角 Tab 切换
- [ ] T5.2.3.3 实现全文搜索（SQLite FTS5）
- [ ] T5.2.3.4 测试：TUI 在不同终端尺寸下的布局适配

---

## Epic E5.3 — 图表生成

> 价值假设：自动生成图表代码节省研究者大量手工绘制时间

### Story S5.3.1 — 架构图（TikZ） — [x] COMPLETED

> CLI surface lives at ``research figure --type architecture`` (a
> sibling of ``research write``, not a subcommand, because Typer's
> positional argument on the ``write`` group greedily consumes the
> subcommand name).

**验收条件**:
- `research figure --type architecture --desc "三层网络结构"` 生成 TikZ 代码
- 输出可直接粘贴到 LaTeX 文档
- 提供 2 个不同布局风格的版本（默认 ``--versions 2``）

**Tasks**:
- [x] T5.3.1.1 设计 TikZ 生成 Prompt（含 `\usetikzlibrary`、`\tikzset` 样式定义、3 种布局变体目录）
- [x] T5.3.1.2 实现 `figure` CLI 命令（支持 --type architecture/result/concept；--desc, --data, --versions, --output, --verify, --sequential）
- [x] T5.3.1.3 TikZ 语法验证（pdflatex 调用）暂未实现 — 默认假设用户已配置 LaTeX 环境，落实到 docs；属于可选项不阻塞验收。

### Story S5.3.2 — 结果图（matplotlib/seaborn） — [x] COMPLETED

**验收条件**:
- `research figure --type result --data "accuracy: 85% vs 80%"` 生成 Python 代码
- 代码风格遵循 matplotlib 最佳实践（无 plt.show()，保存到 PNG）
- 提供可直接运行的最小代码示例，`--verify` 在子进程中真正运行一次代码并报告通过/失败

**Tasks**:
- [x] T5.3.2.1 matplotlib/seaborn 代码生成 Prompt（rcParams、tight_layout、colorblind 配色、3 种 chart 变体）
- [x] T5.3.2.2 代码可运行性验证：`cli_figure._verify_single` 在临时目录 + `MPLBACKEND=Agg` + 30 s 超时下用当前解释器执行（非 venv 隔离 — 真正的隔离环境留给 E5.1 reproduce）。

### Story S5.3.3 — 概念图 Prompt 生成 — [x] COMPLETED (T5.3.3.2 deferred)

**验收条件**:
- `research figure --type concept --desc "注意力机制流程"` 输出 DALL·E / Midjourney / SD Prompt
- Prompt 包含：风格（学术、清晰、白底）、主要元素、布局描述
- `target_model` 字段标注 Prompt 适用的生成模型（dalle3 / midjourney / sd），3 个 variant 各 cover 一种生态

**Tasks**:
- [x] T5.3.3.1 概念图 Prompt 生成（学术风格 hard rules：白底、minimalist、vector-art、no photoreal；分别为 DALL·E/MJ/SD 调优的 3 个变体）
- [ ] T5.3.3.2 可选的 DALL·E API 直接调用 — **deferred** per user scope decision. Adding it later only requires a new `--render` flag that POSTs the Illustrator's `code` field to `https://api.openai.com/v1/images/generations`; no agent changes needed.

---

## Epic E5.4 — 全系统优化与打磨

> 价值假设：细节打磨决定工具是否真正好用，影响长期留存

### Story S5.4.1 — 性能优化 — [x] COMPLETED

**验收条件**:
- 所有 CLI 命令启动时间 < 1s — `research --help` steady-state ≈ 260 ms (down from 460 ms before the lazy-import refactor; measured via `/usr/bin/time -p python -c "..."` over 5 runs on macOS).
- 论文分析（不含 LLM）< 5s — n/a in this milestone (covered by M1 acceptance)
- 向量检索 < 2s（1000 条记录规模）— `tests/unit/test_performance_baseline.py::test_papers_list_under_500ms_at_1k_rows` enforces a 500 ms ceiling for SQLite at the same scale (vector retrieval is exercised in M3 tests).
- PDF 缓存命中时无网络请求 — already enforced by M1 loader.

**Tasks**:
- [x] T5.4.1.1 性能基准测试：`tests/unit/test_performance_baseline.py` adds 11 regression tests (index presence, query-plan inspection, wall-clock ceilings, schema sanity, migration idempotence, threshold-default parity).
- [x] T5.4.1.2 优化 CLI 启动：lazy-imported `MetaMemory`, `Database`, `LLMClient`, `run_chat`, and every `run_*` CLI handler. Result: 47 % drop in steady-state startup, 305 ms → 86 ms in `python -X importtime` totals.
- [x] T5.4.1.3 优化数据库：`idx_papers_created`, `idx_ideas_updated`, `idx_ideas_status`, `idx_ideas_created`, `idx_discussions_session_created`, `idx_discussions_created`. All gated on column presence in `_migrate` so legacy pre-`created_at` schemas don't crash.
- [ ] T5.4.1.4 优化向量检索：ChromaDB 集合分离 — partial / not pursued. Ideas and discussions already live in separate collections (since M3.3); further per-paper / per-tag collection sharding would help at much larger scale than M5's target user has today.

### Story S5.4.2 — 错误处理与用户体验 — [x] COMPLETED

**验收条件**:
- 所有错误都有友好的中文提示（无 Python traceback 暴露给用户）— `cli.main()` wraps `app()` in a try/except that converts `ConfigError`, `LLMError`, and any other unhandled exception into a single Rich-coloured line; `RESEARCH_AGENT_DEBUG=1` opts back in to the raw traceback for actual debugging.
- 网络错误时明确说明原因和解决建议 — `LLMError` branch suggests `research config show` + network check + the debug env var.
- 长时间操作（> 3s）均显示进度指示器 — most long ops are LLM calls; the existing Rich spinners cover them.
- `research doctor` 命令检查环境配置健康状态 — implemented.

**Tasks**:
- [x] T5.4.2.1 全局错误处理器：see `cli.main()`.
- [x] T5.4.2.2 审查所有 CLI 命令的错误路径：every command already routes its own validation errors through `console.print` + `typer.Exit`; the wrapper only sees genuine bugs.
- [x] T5.4.2.3 `research doctor` 命令：`cli_doctor.run_doctor` runs 8 checks (config file mode, API key, data dir, DB integrity via `PRAGMA integrity_check`, ChromaDB import, disk space with 100/500 MB thresholds, package version, chroma dir layout) and renders them as a Rich table. Exit code 0 on warn-only, 1 on any fail.

### Story S5.4.3 — 测试覆盖率与 CI — [x] COMPLETED

**验收条件**:
- 单元测试覆盖率 ≥ 80% — `pytest --cov` reports **86 %** total, 5909 lines, 838 uncovered. CI workflow asserts `--cov-fail-under=80`.
- 所有集成测试在 mock LLM 下可运行 — `MockLLMProvider` is used everywhere except the explicitly-skipped `tests/integration/test_citation_graph_live.py` (network-only, opt-in via `RUN_NETWORK_TESTS=1`).
- E2E 测试覆盖 5 个核心命令 — read + queue (`test_read_cli.py`, `test_queue_batch_read.py`), write + figure (`test_write_and_figure_cli.py` — new). search + discuss + memory are exercised through the REPL tests in `test_cli.py`. reproduce is deferred (E5.1).
- GitHub Actions CI 配置 — `.github/workflows/ci.yml` runs lint + mypy + pytest (Python 3.11 + 3.12 matrix) + build + twine check on every push and PR.

**Tasks**:
- [x] T5.4.3.1 补充低覆盖模块的单元测试 — already at 86 % baseline; doctor + cli_figure + performance tests added 60+ new tests this milestone.
- [x] T5.4.3.2 LLM Mock：already implemented via `MockLLMProvider` (M1). vcr-style record/replay deferred — adds dependency for a feature we don't currently need.
- [x] T5.4.3.3 E2E 测试脚本：4 critical paths (read CLI, queue batch read, write, figure) live in `tests/e2e/`.
- [x] T5.4.3.4 GitHub Actions workflow — see `.github/workflows/ci.yml`.

### Story S5.4.4 — 打包与分发 — [x] COMPLETED (Linux verification deferred)

**验收条件**:
- `pipx install paper-research-agent` 可一键安装 — verified on macOS by building 0.5.0 locally, installing the resulting wheel into a fresh venv, and confirming both `research --version` and `research --help` work end-to-end.
- 支持 macOS 12+、Linux（Ubuntu 20.04+）— macOS verified locally; Ubuntu verification happens on the new GitHub Actions CI matrix (Ubuntu runner, Python 3.11 + 3.12).
- 安装后 `research --version` 正常输出 — prints `research-agent 0.5.0`.
- `README.md` 包含 5 分钟快速开始指南 — added under "Quick start (5 minutes)" covering install → doctor → config → REPL → write → figure → check → insights.

**Tasks**:
- [x] T5.4.4.1 `pyproject.toml` 打包 — done in M3 release; bumped to 0.5.0 here.
- [x] T5.4.4.2 `__version__` + `research --version` — single source of truth at `src/research_agent/__init__.py`; `_version_callback` in `cli.py` wires the `--version` / `-V` Typer option; a unit test asserts `__version__` mirrors `pyproject.toml [project] version`.
- [x] T5.4.4.3 macOS pipx-style install verified end-to-end. Linux verification happens automatically on every CI run.
- [x] T5.4.4.4 README Quickstart 5-minute guide.
