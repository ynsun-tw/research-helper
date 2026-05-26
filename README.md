# Research Agent

Local-first conversational CLI for research paper understanding, critical
discussion, and literature intelligence. Multi-agent (Analyst + Critic +
Searcher + MemoryKeeper) with explicit slash commands **and** LLM tool
calling — pick whichever feels natural per turn.

## Requirements

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key (any OpenAI-compatible
  endpoint works; OpenRouter is the default)

## Install

> Distribution name on PyPI is **`paper-research-agent`** (the
> Python import name stays `research_agent`). Currently published to
> Test PyPI while the release stabilises; a production PyPI upload
> follows once the Test PyPI build has soaked.

### Option A — From Test PyPI (current)

The actual runtime dependencies (PyMuPDF, ChromaDB, openai, …) only
live on real PyPI, so you need both indexes:

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  paper-research-agent
```

Or with [pipx](https://pipx.pypa.io/) for an isolated install:

```bash
pipx install \
  --index-url https://test.pypi.org/simple/ \
  --pip-args="--extra-index-url https://pypi.org/simple/" \
  paper-research-agent
```

### Option B — From PyPI (once published)

```bash
pip install paper-research-agent        # or: pipx install paper-research-agent
```

### Option C — From source (development)

```bash
git clone https://github.com/ynsun-tw/research-helper.git
cd research-helper
pip install -e ".[dev]"
```

After any of the above, `research --help` should list the `config`
and `insights` subcommands and the REPL is one `research` away.

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
REPL. Everything else happens inside it.

```bash
research                                       # enter the REPL
research config set api_key sk-or-...          # the only remaining subcommand
research --help
```

Inside the REPL you can either type **slash commands** for explicit
control or **plain text** to let the LLM pick the right tool.

### Slash commands

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

### Natural language → tools

Plain text is sent to the LLM, which has function-calling access to the
backend. Available tools:

`search_arxiv`, `recent_searches`, `recall_history`, `load_paper`,
`discuss_idea`, `save_current_idea`, `list_ideas`, `queue_add`,
`queue_list`, `queue_next`, `get_citations`, `get_references`,
`suggest_search_refinement`, `research_insights`.

The model is instructed to chain them: `"open the BERT paper I searched
last week"` → `recent_searches` → `load_paper`. `"read the next one on my
list"` → `queue_next` → `load_paper`. `"what did we conclude about
positional encodings?"` → `recall_history` then a synthesized recap.
`"who built on this paper?"` → `get_citations` on the anchor paper.
`"what does this paper rely on?"` → `get_references`.
`"what should I search next?"` → `suggest_search_refinement` →
`search_arxiv`. `"how am I doing this month?"` → `research_insights`.

## Quick start

```bash
research config set api_key sk-or-...
research                                     # enter the conversational REPL
research insights --since 30d                # Markdown summary of recent activity
research insights --output report.md         # write the report to a file
```

```text
› /search --mode applied efficient transformer long context
› /queue add 1706.03762 Attention Is All You Need
› /read 1706.03762
› /discuss replace dense attention with top-k sparse attention for 32k contexts
› /idea save sparse-routing-attention
› /ideas update <id-prefix> --status shelved --condition "FlashAttention-3 release"
› /refine                                    # ask Searcher for the next query
› /insights --since 30d                      # weekly research review
› /exit
```

See [`examples/end-to-end-demo.md`](examples/end-to-end-demo.md) for a
full scripted walkthrough that exercises every feature (search →
relevance scoring → queue → read → citation graph → two-phase debate →
parked-idea alerts → activation conditions → dynamic refinement →
cross-session recall → research insights) on a real paper.

## Writing assistant (M4, in progress)

Train the upcoming Scribe agent on your own published papers so it
writes in a voice that actually sounds like yours. Today the M4
surface covers **sample import** (S4.1.1); fingerprint analysis +
draft generation + writing-review pipeline land in subsequent stories.

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
