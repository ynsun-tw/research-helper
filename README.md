# Research Agent

Local-first conversational research companion. **Just talk to it** —
the agent has function-calling access to the full toolbox (paper
search + analysis, debate, writing, figures, self-plagiarism scan,
style training, config, environment diagnostics). Slash commands and
Typer subcommands are still there as fast paths and scripting hooks,
but the recommended way to drive Research Agent is plain English (or
Chinese — `language: zh`).

Multi-agent under the hood: Analyst + Critic + Scribe + Illustrator
+ Searcher + MemoryKeeper, all local.

## Requirements

- Python 3.9+
- An [OpenRouter](https://openrouter.ai/) API key (any OpenAI-compatible
  endpoint works; OpenRouter is the default)

## Install

> Distribution name on PyPI is **`paper-research-agent`** (the Python
> import name stays `research_agent`).

### Option A — From PyPI (recommended)

```bash
pip install paper-research-agent
```

Or with [pipx](https://pipx.pypa.io/) for an isolated install:

```bash
pipx install paper-research-agent
```

### Option B — From Test PyPI (pre-release builds only)

Pre-release smoke-test builds land on Test PyPI before each PyPI
release. The runtime dependencies (PyMuPDF, ChromaDB, openai, …) only
live on real PyPI, so you need both indexes:

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  paper-research-agent
```

```bash
pipx install \
  --index-url https://test.pypi.org/simple/ \
  --pip-args="--extra-index-url https://pypi.org/simple/" \
  paper-research-agent
```

### Option C — From source (development)

Requires Python 3.9, 3.10, 3.11, or 3.12. The steps below assume
macOS / Linux; on Windows replace the `source` line as noted.

```bash
# 1. Clone
git clone https://github.com/ynsun-tw/research-helper.git
cd research-helper

# 2. Create an isolated virtual environment (any 3.9+ Python works)
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Windows cmd:         .venv\Scripts\activate.bat

# 3. Editable install with dev extras (pytest, pytest-cov, ruff, mypy, ...)
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the install:

```bash
research --version            # → research-agent 0.5.1
python -m pytest -q           # full suite (~10s, 656 tests, no network)
python -m ruff check src tests
python -m mypy src/research_agent
```

If all four are green, you're ready to develop. Configure your
OpenRouter API key (see [Configure](#configure)) and start the REPL
with `research`.

#### Day-to-day dev commands

| Task | Command |
|------|---------|
| Run only one test file | `python -m pytest tests/unit/test_cli.py -q` |
| Run one test by keyword | `python -m pytest -k version -q` |
| Watch coverage (≥80% gate) | `python -m pytest --cov=src/research_agent --cov-report=term-missing --cov-fail-under=80` |
| Auto-fix lint issues | `python -m ruff check --fix src tests` |
| Run network-marked tests | `RUN_NETWORK_TESTS=1 python -m pytest -m network` |
| Surface a traceback on CLI error | `RESEARCH_AGENT_DEBUG=1 research <command>` |
| Rebuild distributions locally | `python -m build && python -m twine check dist/*` |

#### Repo layout (top-level)

```
src/research_agent/        # the package (cli.py is the typer entry point)
  agents/                  # Analyst, Critic, Scribe, Searcher, Illustrator,
                           # MemoryKeeper + schemas.py (Pydantic v2 outputs)
  chat/                    # REPL: session, router, slash + LLM tool registry
  core/                    # LLMProvider, Paper, Idea, debate prompts, language
  memory/                  # WorkingMemory + token-aware truncation
  parsers/                 # PyMuPDF PDF parsing
  prompts/                 # YAML system prompts per agent
  search/                  # arXiv / Semantic Scholar / GitHub clients
  storage/                 # SQLite schema + ChromaDB vector store
  style/                   # writing fingerprint, plagiarism, samples
tests/{unit,integration,e2e}
planning/                  # architecture.md, PLAN.md, notes.md (ADRs), milestones/
examples/                  # usage-scenarios-zh.md
```

After any of the install options, `research --help` should list the
writing suite (`write`, `review`, `check`, `style`) alongside `config`
and `insights`, and the conversational REPL is one `research` away.

> Latest published build: **0.5.1** on PyPI —
> [project page](https://pypi.org/project/paper-research-agent/0.5.1/).

## Configure

```bash
# API key from https://openrouter.ai/keys
research config set api_key <your-openrouter-key>

# Default model: deepseek/deepseek-chat — change to any OpenRouter model id
research config set model anthropic/claude-3.5-sonnet
research config set language zh           # or: en (default), zh, 中文

research config show
```

Configuration is stored at `~/.research-agent/config.yaml` (file mode `600`).

| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | — | OpenRouter API key |
| `model` | `deepseek/deepseek-chat` | Model slug on OpenRouter |
| `base_url` | `https://openrouter.ai/api/v1` | API base (change only if self-hosting a proxy) |
| `app_title` | `Research Agent` | Sent as `X-Title` header to OpenRouter |
| `app_url` | `https://github.com/research-agent` | Sent as `HTTP-Referer` header |
| `language` | `en` | Agent reply language: `en` or `zh` |
| `alert_threshold` | `0.8` | Cosine similarity threshold for the parked-idea alert that fires on `/read` (range `[0.0, 1.0]`; lower = more reminders) |

All local state lives under `~/.research-agent/`: `memory.db` (SQLite),
`chroma/` (vector indexes for ideas + discussions), and `cache/`
(downloaded PDFs).

## Usage

Running `research` with no arguments drops you into the conversational
REPL. **Plain English is the primary interface.** Slash commands are
fast shortcuts for power users; Typer subcommands exist so you can
script Research Agent into CI / cron / shell pipelines.

```bash
research                                       # enter the conversational shell
research config set api_key sk-or-...          # bootstrap (or `set_config` inside)
research --help                                # full subcommand reference (scripting)
```

### REPL keys

The shell uses [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/),
so line editing, history, and search work the same way they do in
bash / psql / ipython:

| Key | Effect |
|---|---|
| `Tab` | Complete `/`-commands |
| `↑` / `↓` | Walk through history (prefix-matched) |
| `Ctrl+R` | Reverse history search |
| `Ctrl+L` | Clear the screen |
| `Ctrl+A` / `Ctrl+E` | Jump to start / end of line |
| `Ctrl+C` | Cancel current input; press again on an empty prompt to exit |
| `Ctrl+D` | Exit (saves the session) |
| `?` or `/?` | Same as `/help` |

History is persisted across sessions in `~/.research-agent/repl_history`.
The prompt itself reflects loaded state, e.g.
`You(arxiv:1706.03762 | idea:9a1f00b3)>`, so you always know which paper
and idea are active.

LLM replies stream token-by-token (you see text as it's generated, not
after the whole response finishes).

### Natural language → tools

Plain text is sent to the LLM, which has function-calling access to
the entire backend. The agent picks the right tool, chains them if
needed, and surfaces a concise answer. Examples:

| What you type | What it triggers |
|---|---|
| `搜一下 sparse attention 的最新综述` | `search_arxiv` |
| `read the BERT paper I searched last week` | `recent_searches` → `load_paper` |
| `what did we conclude about positional encodings last month?` | `recall_history` |
| `who built on this paper?` | `get_citations` on the anchor paper |
| `save this for later` (after a search) | `queue_add` |
| `what's on my reading list?` | `queue_list` |
| `read the next one` | `queue_next` → `load_paper` |
| `is my environment OK?` | `run_doctor` |
| `is my style trained?` | `style_show` |
| `import my last two arxiv papers as style samples: 2305.14314, 2301.07041` | (confirms first, then) `train_style` |
| `build the fingerprint` | (confirms, then) `build_fingerprint` |
| `draft me an introduction in ~300 words about sparse top-k attention` | `draft_section` |
| `give me a TikZ diagram of a three-layer encoder` | `draft_figure` |
| `save version B to ~/intro.md` | `save_draft_to_file` |
| `does this overlap with anything I've published?` | `check_self_plagiarism` against the latest draft |
| `tighten this and address the issues` | `revise_draft` |
| `set the model to anthropic/claude-3.5-sonnet` | (confirms, then) `set_config` |
| `how am I doing this month?` | `research_insights` |

**State-mutating tools** (`train_style`, `build_fingerprint`,
`update_fingerprint`, `set_config`) are gated by a prompt-level
contract: the agent will summarise the exact action and ask for
confirmation before writing anything. Confirm or veto in natural
language; nothing hits disk until you say yes.

The full list of tools registered for the agent: `search_arxiv`,
`recent_searches`, `recall_history`, `load_paper`, `discuss_idea`,
`save_current_idea`, `list_ideas`, `queue_add`, `queue_list`,
`queue_next`, `get_citations`, `get_references`,
`suggest_search_refinement`, `research_insights`, `run_doctor`,
`style_show`, `style_history`, `draft_section`, `draft_figure`,
`save_draft_to_file`, `check_self_plagiarism`, `revise_draft`,
`train_style`, `build_fingerprint`, `update_fingerprint`,
`get_config`, `set_config`.

### Slash commands (fast paths)

If you prefer explicit control or want to skip an LLM round-trip,
every slash invokes the same backend directly:

| Command | What it does |
|---|---|
| `/search [--mode theoretical\|applied\|group:<author>] <keywords>` | Paper search with LLM relevance scoring; primary source is arXiv with a Semantic Scholar fallback if arXiv rate-limits or errors. Sorted by score; flags papers already in your library. Optional `--mode` biases the candidate set: `theoretical` (analysis / proofs), `applied` (benchmarks / experiments), or `group:"<author name>"` (quote multi-word names) |
| `/history [N]` | Recent `/search` queries across sessions, with hit counts and read markers |
| `/recall <query>` | Semantic search across **past** REPL discussions (cross-session) |
| `/read <arxiv-id \| title \| path.pdf>` | Download + Analyst + Critic; sets the conversation anchor; auto-marks the queue entry done if present |
| `/discuss <idea or follow-up>` | First turn: structured Analyst (contributions / impact / related work) + Critic (objections / score / suggestions). Follow-up turns: grounded prose, no re-scoring. Must run `/read` first. |
| `/queue` | List pending entries (alias for `/queue list`) |
| `/queue add <id> [title…]` | Save a paper for later (pending) |
| `/queue list [all\|pending\|done\|skipped\|in_progress]` | Filter the queue |
| `/queue next` | Preview the next pending entry without state change |
| `/queue read` | Load + analyze the next pending entry, auto-mark done |
| `/queue done\|skip\|remove <id>` | Manual state transitions |
| `/cites [arxiv-id]` | Papers that cite the anchor (or given) paper — forward references via Semantic Scholar |
| `/refs [arxiv-id]` | Papers cited by the anchor (or given) paper — backward references via Semantic Scholar |
| `/refine` | Ask Searcher to propose the **next** search query from your recent discussion (query + optional `--mode` + reason + confidence); interactively accept / edit / skip |
| `/insights [--since 30d\|7d\|6m\|all]` | Deterministic Markdown summary of your local activity: papers (by year, top tags / authors / venues), ideas (by status, average critic score, most-debated, top-scored) and discussion volume. No LLM call - safe to run anywhere |
| `/paper` | Summary of the current anchor paper |
| `/idea save [title]` | Persist the active debate as a saved idea |
| `/ideas` | List saved ideas with their latest critic score |
| `/idea show <id-prefix>` | Show one idea + its full score history |
| `/ideas update <id> [--status <s>] [--feedback <note>] [--condition "<phrase>"] [--clear-conditions]` | Update status, log score feedback, or pin / clear activation conditions (multiple `--condition` flags allowed) |
| `/help` | List every slash command |
| `/exit` | Persist + flush vector indexes + quit |

After every `/read`, Research Agent quietly checks your `shelved` /
`waiting` ideas; if the paper looks topically related (cosine
similarity ≥ `alert_threshold`, default `0.8`), it prints a
one-line banner with `/idea show <prefix>` shortcuts so you can
revisit context you parked earlier. Tune the trigger via
`research config set alert_threshold 0.85` (range `[0.0, 1.0]`;
lower = more reminders, higher = fewer false positives).

You can also pin **activation conditions** on a shelved idea — free-form
phrases that describe what would unblock it (a dataset release, a
checkpoint, a baseline result). Set them via
`/ideas update <id> --condition "FineWeb-Edu dataset"` (repeatable in one
command, clear with `--clear-conditions`). Every `/search` then scans
incoming hits for those phrases (case-insensitive substring) and prints
a "Shelved idea(s) may have an unblock" banner whenever a new paper
mentions one — letting search results pull an idea back into your
attention automatically.

## Quick start (5 minutes)

```bash
# 1. Install (pick one)
pipx install paper-research-agent
#  or  →  pip install paper-research-agent
#  or  →  pip install -e ".[dev]" from the repo root for a dev install

# 2. Configure
research config set api_key sk-or-...     # OpenRouter key from https://openrouter.ai/keys
research config set language zh           # or en (default)

# 3. Drop into the shell and just talk
research
```

Inside the REPL, an end-to-end research session looks like this —
plain English the whole way:

```text
You>  is my setup OK?
→ calling run_doctor()
[diagnostic table prints]
Everything is healthy.

You>  search for efficient transformer long context, applied bias
→ calling search_arxiv(query=efficient transformer long context, mode=applied)
[scored hits]

You>  read paper 2305.14314 and tell me the punchline
→ calling load_paper(source=2305.14314)
[Analyst + Critic summary]

You>  let's debate replacing dense attention with sparse top-k for 32k contexts
→ calling discuss_idea(idea=…)

You>  import my last two arxiv papers as style training samples: 2305.14314, 2301.07041
"I'll re-import these 2 papers into the style corpus, replacing any prior
samples for them. OK to proceed?"
You>  yes
→ calling train_style(sources=[arxiv:2305.14314, arxiv:2301.07041], append=false)

You>  now compute the fingerprint
→ calling build_fingerprint()

You>  draft me an introduction (~300 words) about sparse top-k attention
→ calling draft_section(section=introduction, context=…, versions=3)
[3 panels rendered]

You>  version B is the keeper — save it to ~/intro.md
→ calling save_draft_to_file(path=~/intro.md, kind=section, version=B)

You>  any overlap with my own past work?
→ calling check_self_plagiarism(target=latest:introduction:B)

You>  tighten this and address the issues
→ calling revise_draft(target=latest:introduction:B)
[issue list + revised draft]

You>  save the revision to ~/intro_revised.md
→ calling save_draft_to_file(path=~/intro_revised.md, kind=revision)

You>  /exit
```

The slash form is available for every step if you prefer explicit
control (`/search`, `/read`, `/discuss`, `/idea save`, `/queue add`,
`/insights --since 30d`, …). See [Slash commands](#slash-commands-fast-paths).

If you'd rather script Research Agent — cron job, CI artifact, batch
PDF import — every CLI subcommand still works headlessly:

```bash
research style train --dir ~/papers
research style fingerprint
research write introduction --context "..." --output drafts/intro.md
research review drafts/intro.md --section introduction --output drafts/intro.review.md
research check drafts/intro.md --output reports/intro.similarity.md
research insights --since 30d --output reports/activity.md
research doctor
```

The Typer surface is exactly the chat tools' inverse: every CLI verb
has a same-named chat tool, and vice versa.

See [`examples/end-to-end-demo.md`](examples/end-to-end-demo.md) for a
full scripted walkthrough that exercises every feature (search →
relevance scoring → queue → read → citation graph → two-phase debate →
parked-idea alerts → activation conditions → dynamic refinement →
cross-session recall → research insights) on a real paper.

## Writing assistant

Train Scribe on your own published papers so it writes in a voice
that sounds like yours, then drive the whole writing flow from chat
(natural language) or from CLI (scripting). The two surfaces are
strictly equivalent — pick the one that fits your workflow.

| Chat tool | CLI command | What it does |
|---|---|---|
| `train_style(sources?, directory?, append?)` | `research style train …` | Import paragraphs from your papers into the corpus |
| `build_fingerprint()` | `research style fingerprint` | Compute the style vector from the corpus |
| `update_fingerprint()` | `research style update` | Recompute, folding in accepted revisions, archive old version |
| `style_show()` | `research style show` | Corpus summary + fingerprint status |
| `style_history()` | `research style history` | List archived fingerprint versions |
| `draft_section(section, context?, versions?, check_against?)` | `research write <section> …` | Scribe drafts N variants |
| `draft_figure(figure_type, description, data?, verify?)` | `research figure --type … --desc …` | Illustrator generates figure code |
| `check_self_plagiarism(target, threshold?)` | `research check <file>` | Self-overlap scan (no LLM) |
| `revise_draft(target, section?)` | `research review <file>` | Analyst+Critic+Scribe auto-revision |
| `save_draft_to_file(path, kind?, …)` | (built into `--output` on the CLI side) | Persist a cached draft to disk |

The CLI sections below document the underlying flags; everything they
take is also exposed as a chat-tool argument with the same semantics.

```bash
# Pull paragraphs from a folder of PDFs
research style train --dir ~/papers

# Or hand-pick sources (arXiv ids and local PDFs may be mixed)
research style train arxiv:2301.07041 ~/papers/my-thesis.pdf

# Inspect the corpus
research style show
```

`style train` parses each source, splits it into paragraphs, drops
non-prose (references, acknowledgements, formula-dense methodology,
single-sentence captions), and writes the survivors into the
`style_samples` table under `~/.research-agent/memory.db`. Re-running
the same source replaces its prior samples by default; pass
`--append` to accumulate instead.

Once samples exist, build a **fingerprint** that captures how you
write:

```bash
research style fingerprint
```

The fingerprint is computed entirely offline (no LLM call) and lands
at `~/.research-agent/style/fingerprint.json`. It has three layers:

| Layer | What it captures |
|---|---|
| **Macro** | abstract opener, intro opener, related-work organization (chronological / thematic / comparison), avg sections per paper |
| **Micro** | sentence-length distribution (avg / median / p10 / p90), avg paragraph length, top transition words & their per-100-sentence rates, hedging / confidence / passive rates, type-token ratio |
| **Markers** | dominant citation format (`latex_cite` / `bracket_num` / `author_year` / `mixed`), figure & table reference style (`Figure` vs `Fig.`), em-dash usage, your top section titles |

`research style show` prints the corpus inventory and (if present)
the fingerprint summary, side by side.

### Draft a section with Scribe

```bash
research write abstract --words 250
research write introduction \
  --context "this paper studies sparse top-k attention for 32k contexts" \
  --versions 3 \
  --output drafts/intro.md
```

`research write <section>` invokes the Scribe agent, which produces
**three stylistic variants** by default — *concise*, *technical depth*,
and *narrative arc* — by issuing the LLM calls in parallel (use
`--sequential` to disable). Each draft is rendered in its own Rich
panel with a one-line note explaining how it differs from the others;
pass `--output drafts.md` to also save the bouquet to disk.

Supported section names (aliases in parentheses): `abstract`,
`introduction` (`intro`), `related_work` (`related`), `method`
(`methods` / `approach`), `results` (`experiments` / `evaluation`),
`discussion`, `conclusion`.

When `~/.research-agent/style/fingerprint.json` exists, the Scribe
mimics it (sentence length, transitions, hedging vs confidence
balance, citation format). Without a fingerprint it falls back to
neutral academic prose and says so in each draft's style note.

#### Context-aware writing

When you pass `--context "<description>"`, the Scribe doesn't just
parrot the description — it also pulls related material from your
memory store and injects it into the prompt:

- **Related ideas** from your library (semantic similarity ≥ 0.5,
  top 3) — title, status, last critic score, summary.
- **Recent cross-session discussion excerpts** that look topically
  relevant (top 3).
- **Existing drafts** you point it at with `--check-against PATH`
  (repeatable) — body included verbatim (truncated to ~2 KB each)
  with an explicit instruction not to duplicate or contradict.

```bash
research write conclusion \
  --context "sparse top-k attention for 32k contexts" \
  --check-against drafts/intro.md \
  --check-against drafts/method.md \
  --versions 3
```

The CLI prints a one-line `Scribe context: user context, 2 related
idea(s), 1 discussion excerpt(s), 2 draft(s) to stay consistent
with` summary before the panels so you can see what the agent saw.

### Review a draft (Analyst + Critic → Scribe)

```bash
research review drafts/intro.md --section introduction --output drafts/intro.review.md
```

`research review <file>` runs the **auto-review pipeline**:

1. **Analyst (writing mode)** — flags weak argumentation,
   missing differentiation from related work, evidence-claim
   gaps, undefined terms.
2. **Critic (writing mode)** — flags overclaim, unsupported
   conclusions, logical gaps, hedging mismatch. Runs in parallel
   with the Analyst (`asyncio.gather`).
3. **Scribe revision** — produces a single revised draft that
   addresses both reviews while preserving your fingerprinted
   voice. Skips the LLM entirely if both reviews come back empty.

The console prints four panels (original draft, analyst review,
critic review, revised draft); `--output review.md` also writes the
whole bundle to a Markdown file for diffing.

```bash
research review drafts/intro.md --section introduction --interactive
```

Pass `--interactive` to walk through each reviewer issue and
suggestion one at a time (y / N). The Scribe revision will only
address the items you accepted, and at the end you get a coloured
unified diff between original and revised. The (original, revised,
selected_*, rejected_*) tuple is persisted to the
`draft_revisions` SQLite table by default — pass `--no-save` to opt
out. The continuous-learning loop (next subsection) consumes these
rows to keep the fingerprint in sync with how you actually edit
Scribe output.

#### Continuous fingerprint learning

```bash
research style update          # refresh fingerprint from samples + accepted revisions
research style history         # list all saved fingerprint versions
```

`research style update` recomputes the fingerprint by combining the
static `style_samples` corpus with every accepted Scribe revision
(`revised_text` from the `draft_revisions` table) and bumps the
version. The previous `fingerprint.json` is archived next to it as
`fingerprint_v<N>.json`; `research style history` lists every
version side by side so you can see how your voice drifts as you
keep using Scribe.

The flow is:

1. `research style train` (one-shot, from your published papers) →
   seeds the corpus.
2. `research style fingerprint` → v1 baseline.
3. `research write <section> --context …` → drafts.
4. `research review <file> --interactive` → accept / reject
   suggestions; the (original, revised) pair is saved.
5. `research style update` → folds those accepted revisions back
   into the fingerprint as v2, v3, …

### Figure generation (M5 — `research figure`)

```bash
research figure --type architecture --desc "three-layer encoder with residual connections" --versions 2 --output figs/arch.md
research figure --type result       --desc "accuracy comparison across 3 baselines" --data "ours 85, baseline-A 80, baseline-B 78" --verify
research figure --type concept      --desc "attention mechanism flow" --versions 3 --output figs/concept.md
```

`research figure` drives the **Illustrator agent** to produce N
variant drafts of a paper figure in parallel. Three modes:

- `--type architecture` → **TikZ** snippets ready to paste into LaTeX.
  Includes `\usetikzlibrary` declarations and a `\tikzset{}` style
  block. Variants cycle through layered horizontal / hub-and-spoke /
  encoder-decoder vertical layouts.
- `--type result` → **matplotlib / seaborn Python** scripts that
  write to `output.png` (no `plt.show()`, paper-ready rcParams,
  colorblind palette). Variants cycle through grouped bar / line
  with shaded variance / paired boxplot. Pass `--verify` to actually
  execute each draft in a subprocess (30 s timeout, `MPLBACKEND=Agg`)
  and report run / fail per draft.
- `--type concept` → **text-to-image prompts** for DALL·E 3,
  Midjourney v6, and Stable Diffusion (one variant per ecosystem,
  with model-specific phrasing and a `negative_prompt` for SD/MJ).
  Direct API rendering (uploading to DALL·E) is left to a follow-up.

Each draft includes a `notes` summary of what makes it distinct and
a `suggested_use` phrase telling you which paper context it fits.
With `--output PATH` the whole bouquet is written to a Markdown file
with fenced code blocks and verification status; without it the
output stays in the terminal as syntax-highlighted Rich panels.

### Self-plagiarism check

```bash
research check drafts/intro.md --threshold 0.4 --output reports/intro.similarity.md
```

`research check <file>` scans each paragraph of the draft against
every paragraph in your `style_samples` corpus using paragraph-level
**TF-IDF + cosine similarity** (pure Python, no heavy dependencies).
The default threshold is 0.4 (per the M4 milestone); raise it for a
stricter scan or lower it to surface light echoes.

The console renders one panel summary plus a Markdown report:

- For every flagged paragraph: the draft text, the matching corpus
  paragraph (with `arxiv:<id>` source label), the similarity %, and
  concrete rewrite suggestions that scale with severity (≥ 0.7 →
  "rewrite from scratch", ≥ 0.5 → "paraphrase and cite", ≥ 0.4 →
  "trim or merge").
- Exit code is `0` for a clean check and `2` when at least one match
  trips the threshold, so it slots cleanly into CI.

## Development

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest
```

Live network tests (hit arXiv / Semantic Scholar) are skipped by
default. To run them:

```bash
RUN_NETWORK_TESTS=1 pytest -m network
```

Skip them explicitly with `pytest -m 'not network'` (already the
default via `RUN_NETWORK_TESTS` being unset).

## Planning

See [planning/PLAN.md](planning/PLAN.md) and
[planning/architecture.md](planning/architecture.md). Milestone notes
live under [`planning/milestones/`](planning/milestones/).
