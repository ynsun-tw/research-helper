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

```bash
pip install -e ".[dev]"
```

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
| `/search <keywords>` | Paper search with LLM relevance scoring; primary source is arXiv with a Semantic Scholar fallback if arXiv rate-limits or errors. Sorted by score; flags papers already in your library |
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
| `/paper` | Summary of the current anchor paper |
| `/idea save [title]` | Persist the active debate as a saved idea |
| `/ideas` | List saved ideas with their latest critic score |
| `/idea show <id-prefix>` | Show one idea + its full score history |
| `/idea update <id> status=<new>` | Change status (active / parked / shipped / dropped) |
| `/help` | List every slash command |
| `/exit` | Persist + flush vector indexes + quit |

### Natural language → tools

Plain text is sent to the LLM, which has function-calling access to the
backend. Available tools:

`search_arxiv`, `recent_searches`, `recall_history`, `load_paper`,
`discuss_idea`, `save_current_idea`, `list_ideas`, `queue_add`,
`queue_list`, `queue_next`.

The model is instructed to chain them: `"open the BERT paper I searched
last week"` → `recent_searches` → `load_paper`. `"read the next one on my
list"` → `queue_next` → `load_paper`. `"what did we conclude about
positional encodings?"` → `recall_history` then a synthesized recap.

## Quick start

```bash
research config set api_key sk-or-...
research
```

```text
› /search efficient transformer long context
› /queue add 1706.03762 Attention Is All You Need
› /read 1706.03762
› /discuss replace dense attention with top-k sparse attention for 32k contexts
› /idea save sparse-routing-attention
› /exit
```

See [`examples/end-to-end-demo.md`](examples/end-to-end-demo.md) for a
full scripted walkthrough that exercises every feature (search →
relevance scoring → queue → read → two-phase debate → save idea →
cross-session recall) on a real paper.

## Development

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest
```

## Planning

See [planning/PLAN.md](planning/PLAN.md) and
[planning/architecture.md](planning/architecture.md). Milestone notes
live under [`planning/milestones/`](planning/milestones/).
