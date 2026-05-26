# End-to-End Demo: Transformer + Sparse Attention

A 15-minute scripted walkthrough that exercises every M1 → M3 feature on a
real paper. Use it to smoke-test the CLI after pulling, to demo the agent
to someone, or as a starting point for your own session.

## What gets covered

| Capability | Slash | LLM tool | Step |
|---|---|---|---|
| arXiv search + LLM relevance scoring (M3 S3.1) | `/search` | `search_arxiv` | §1, §6 |
| Search mode bias (M3 T3.1.3.3) | `/search --mode theoretical\|applied\|group:<author>` | `search_arxiv(mode=...)` | §1 |
| Search history persistence (M3.1) | `/history` | `recent_searches` | §2, §6 |
| Reading queue (M3 S3.2.2) | `/queue add\|list\|next\|read\|done\|skip\|remove` | `queue_add`, `queue_list`, `queue_next` | §3 |
| Paper load + Analyst + Critic (M1) | `/read` | `load_paper` | §4 |
| Citation graph (M3 T3.1.2.2) | `/cites`, `/refs` | `get_citations`, `get_references` | §4.5 |
| Two-phase debate (M2 + M2 fix) | `/discuss` | `discuss_idea` | §5 |
| Save / inspect ideas (M2) | `/idea save`, `/ideas`, `/ideas show` | `save_current_idea`, `list_ideas` | §5 |
| Parked-idea alerts on `/read` (M3 T3.4.1) | implicit (banner after `/read`) | implicit | §5.4 |
| Activation conditions on shelved ideas (M3 T3.4.2) | `/ideas update <id> --condition "<phrase>"` | — | §5.5 |
| Cross-session recall (M3 S3.3.1) | `/recall` | `recall_history` | §7 |
| Dynamic search refinement (M3 S3.2.3) | `/refine` | `suggest_search_refinement` | §8 |
| Research insights / MetaMemory (M3 S3.3.2) | `/insights`, `research insights` | `research_insights` | §9 |
| Auto-mark queued paper as `done` on `/read` | implicit | implicit | §3 verification |

The flow uses **one** anchor paper — `arxiv:1706.03762` *Attention Is All
You Need* (Vaswani et al., 2017) — and **one** concrete idea:

> **Idea**: Replace dense O(n²) self-attention with **top-k sparse
> attention** (each query attends only to its k highest-scoring keys),
> targeting 32k-token contexts. Routing is a learned linear projection
> over the query.

This idea is deliberately specific: the Critic should object on quality
loss + non-differentiable top-k, the Analyst should highlight the well-
known O(n·k) win, and a follow-up turn can ask about learned routing
alternatives (Reformer, Routing Transformer, etc.). Perfect for testing
the structured-then-prose debate flow.

---

## Prerequisites

1. Working OpenRouter key:
   ```bash
   research config set api_key sk-or-...
   research config set model deepseek/deepseek-chat   # or claude / gpt-4o
   research config set language zh                    # optional
   research config show
   ```
2. Online (the demo hits `export.arxiv.org` once).
3. Fresh data dir (optional, for a clean run):
   ```bash
   rm -rf ~/.research-agent/memory.db ~/.research-agent/chroma
   ```

Now launch the REPL:

```bash
research
```

You should see the welcome panel and the `›` prompt.

---

## §1. Search arXiv with LLM relevance scoring

Try the explicit slash form first:

```text
/search efficient transformer long context
```

What to expect:

- A `Searching arXiv…` spinner, then a `Scoring relevance (LLM)…`
  spinner (this is the new Searcher agent).
- A table with **ID, Title, Year, Score, Why, Read** — rows sorted by
  the LLM's 0-1 relevance score. Each row has a short one-sentence
  reason in the `Why` column.
- Footer: `Sorted by relevance (LLM, 0-1).`

Then the natural-language form:

```text
find me three recent papers about retrieval-augmented generation
```

The model should call `search_arxiv(query=..., max_results=3)` (you'll
see `→ calling search_arxiv (query=...)` in the console), render the
same table, then produce a short plain-language summary.

Verification checkpoints:

- [ ] Scores are between 0.00 and 1.00.
- [ ] At least one row has a non-empty `Why`.
- [ ] Rows in descending score order.

### 1.b Bias the search with `--mode`

Same query, biased toward experimental work:

```text
/search --mode applied efficient transformer long context
```

You should see a faint `Biasing search toward applied papers.` hint
before the table. The candidate set should skew toward benchmark /
systems papers rather than theory. Two other modes exist:

- `/search --mode theoretical <kw>` — analysis / proofs / convergence
- `/search --mode "group:Andrej Karpathy" <kw>` — author bias (quote
  multi-word names; `shlex` parses them)

The LLM tool understands the same vocabulary: `"find me 5 applied
papers on sparse attention"` should result in
`search_arxiv(query="sparse attention", mode="applied", max_results=5)`.

---

## §2. Review search history

```text
/history
```

Both searches above should appear, most recent first, with hit counts
and a `Read 0/n` column (no papers loaded yet).

Now ask in natural language:

```text
what did I search for in this session?
```

Expect the model to call `recent_searches(limit=...)`, render the same
table, and then summarise in prose. Each hit line in the LLM-facing
text includes `score=0.xx`.

---

## §3. Build a reading queue

Queue two candidates — one transformer classic, one efficient-attention
work — for later study:

```text
/queue add 1706.03762 Attention Is All You Need
/queue add 2004.05150 Longformer: The Long-Document Transformer
/queue list
```

You should see both rows with `status=pending`.

Try the natural-language form (the LLM picks the tool):

```text
also save the Reformer paper for me — 2001.04451
```

Expect the model to call `queue_add(arxiv_id="2001.04451", title="Reformer")`
and confirm. Run `/queue list` again — there should be three pending
entries.

Inspect "what's next":

```text
/queue next
```

It should print `Next up: 1706.03762 Attention Is All You Need`.

---

## §4. Load the anchor paper (Analyst + Critic)

Now actually read it. Two equivalent options:

**Option A — slash:**

```text
/read 1706.03762
```

**Option B — natural language (exercises tool chaining):**

```text
let's read the next paper in my queue
```

In option B the model should call `queue_next()` → see
`1706.03762` → then call `load_paper(source="1706.03762")` →
finally summarise. Either way:

- arXiv PDF gets downloaded into `~/.research-agent/cache/`.
- A paper header panel renders (title, authors, year, abstract).
- An `Analyzing (Analyst + Critic)…` spinner runs.
- A **read report** prints with: Analyst contributions / method
  insights / impact, then Critic objections / score / suggestions.
- The queue auto-promotes that entry to `done`. Verify with:

  ```text
  /queue list all
  ```

  Row `1706.03762` should show `status=done` with a completed_at
  timestamp.

The anchor paper is now set; `/paper` will summarise it on demand.

---

## §4.5 Walk the citation graph

The anchor paper is now eligible for **forward** (`/cites`) and
**backward** (`/refs`) traversal via Semantic Scholar. Try both:

```text
/cites
/refs
```

What to expect:

- `/cites` → a Rich table titled `Papers that cite Attention Is All You
  Need (forward references via Semantic Scholar)`. Hits include any
  paper that mapped back to an arXiv id; non-arXiv journal/conference
  citations are filtered out (with a hint).
- `/refs` → backward references: the works *Attention Is All You Need*
  itself builds on (encoder-decoder seq2seq, attention precursors,
  byte-pair encoding, …).
- Both tables flag papers already in your library with a `✓` in the
  Read column.

You can pass an explicit arXiv id too: `/cites 1810.04805` (papers
citing BERT). The natural-language form picks the right tool:

```text
who built on this paper?            # -> get_citations
what does this paper rely on?       # -> get_references
```

Tip: pair this with the queue — if you see a citation that looks
promising, follow up with `/queue add <id> <title>` to read it later.

---

## §5. Debate the idea (structured first turn → prose follow-ups)

This is the M2 idea-workshop loop. The first turn returns the full
structured envelope; every follow-up turn is grounded prose.

### 5.1 First turn (structured)

```text
/discuss Replace dense O(n^2) self-attention with top-k sparse attention (each query attends to its k highest-scoring keys, k≪n), to scale Transformers to 32k tokens. Routing is a learned linear projection over the query.
```

Expected output:

- **Analyst panel**: supports / contributions, method intuition,
  potential impact, related work pointers (should reference Reformer,
  Longformer, Routing Transformer, BigBird, etc.).
- **Critic panel**:
  - `Support score: x/9` (Critic will never give 10; <7 must include
    reasons, see Critic prompt).
  - Numbered objections (expect at least: non-differentiable top-k,
    quality drop, training instability, kernel/IO complexity on
    GPUs).
  - Concrete suggestions for next steps.

### 5.2 Follow-up turn (prose, no scoring)

```text
/discuss what if we replace top-k with a learned routing function like the one in Routing Transformer or Switch Transformer — does that fix the non-differentiability concern?
```

Expected output:

- Two short prose blocks (Analyst then Critic).
- **No new score panel** (this is the M2 fix — only the first turn is
  structured; follow-ups are grounded conclusions).
- The reply must reference the anchor paper concretely (Vaswani's
  scaled dot-product attention, multi-head structure), not just
  general knowledge.

### 5.3 Save the idea

```text
/idea save Top-k sparse self-attention with learned routing
```

…or in natural language:

```text
save this idea — call it "sparse-routing-attention"
```

Then inspect:

```text
/ideas
/idea show <id-prefix>
```

`/ideas` shows the saved idea with its latest critic score. `/idea show`
prints title, description, status, current score, score history (every
debate turn appends a snapshot), and any user feedback.

### 5.4 Shelf the idea + watch the parked-idea alert on the next `/read`

Park the idea so we can demo the proactive recall on the next paper:

```text
/ideas update <id-prefix> --status shelved
```

Now load a *topically related* paper (Longformer is queued from §3):

```text
/queue read
```

After the Analyst + Critic report renders you should see a one-line
banner that looks roughly like:

```
Related ideas you parked previously
- Top-k sparse self-attention with learned routing (shelved, similarity 84%) — /idea show <prefix>
```

Behaviour notes:

- The banner only fires if the cosine similarity between the new
  paper (title + abstract) and a shelved/waiting idea's stored
  embedding is ≥ `alert_threshold` (default `0.80`).
- Tune via `research config set alert_threshold 0.75` (more
  reminders) or `0.9` (only near-duplicates).
- It's purely a nudge — `/read` continues normally if no idea
  matches. Failures are swallowed; a misbehaving vector store
  never crashes a paper read.

### 5.5 Pin activation conditions (literal-phrase alerts on `/search`)

Some ideas are blocked on *concrete external events* — a dataset
release, a checkpoint, a baseline result. Pin them as
**activation conditions** on the shelved idea:

```text
/ideas update <id-prefix> --condition "FlashAttention-3 release" --condition "1B sparse attention checkpoint"
```

Multiple `--condition` flags can be chained in one command; phrases
are greedily consumed until the next `--flag`, so quoting is
optional. Clear with `--clear-conditions`. View what's pinned:

```text
/idea show <id-prefix>
```

The idea panel now shows an **Activation conditions** block.

From here on, every `/search` (slash or LLM) does a literal,
case-insensitive substring match of each condition against the
title + abstract of every hit. When a paper looks like it would
unblock the idea, the banner under the search table reads:

```
Shelved idea(s) may have an unblock:
  - 2407.08608 "FlashAttention-3: Fast and Accurate Attention with Asynchrony" matches condition "FlashAttention-3 release" on Top-k sparse self-attention with learned routing — /idea show <prefix>
```

You can stress-test this by searching for the condition phrase
directly:

```text
/search FlashAttention-3 asynchronous attention
```

The unblock banner should appear under the Searcher-scored table.
Free-form phrases work great here: pick the literal vocabulary that
will appear in the future paper's abstract if it actually delivers
the thing.

---

## §6. Confirm search history sees the loads

```text
/history
```

The earlier `/search efficient transformer long context` row should
now show `1/5` (or similar) in the **Read** column — the LLM saw that
`1706.03762` is in your library and the join lit up. The `recent_searches`
LLM tool will include `[READ]` markers for the same rows.

---

## §7. Cross-session recall (the headline M3 feature)

This is the test that proves discussions actually persist into the
vector index. Two ways to trigger it:

### 7.1 In the same session (uses `exclude_session_id`)

Recall is intentionally cross-*session*, so within this session the
recall is empty. Exit and re-enter:

```text
/exit
```

Then:

```bash
research
```

Now ask:

```text
/recall sparse attention top-k routing
```

Expect a panel `Recalled N past message(s) for: …` listing the
user/analyst/critic messages from the previous session that mention
sparse attention. Each line shows the role, a short session-id prefix,
and a snippet.

The natural-language form is the same flow via the LLM tool:

```text
remind me — what did we conclude about sparse attention last session?
```

The model should call `recall_history(query="sparse attention",
limit=5)`, render the panel, and then synthesise a one-paragraph
recap citing the recalled snippets.

Verification:

- [ ] At least one of the recalled messages comes from the previous
      session id (not the current one).
- [ ] The LLM's recap references concrete points from those snippets
      (it should not just paraphrase your fresh question).

---

## §8. Ask Searcher to refine the next query

By now the session has the §5 debate transcript in working memory.
The Searcher can read it and propose **the next thing to search**:

```text
/refine
```

What happens:

1. Orchestrator hands a compact transcript snippet (last ~12 messages,
   role-prefixed, char-capped) to the Searcher agent.
2. Searcher returns a `SearchSuggestion(query, mode?, reason,
   confidence)`. The banner renders:

   ```
   Suggested next search: top-k sparse attention KV cache (mode: applied)
   Reason: Critic flagged memory bandwidth as the bottleneck.
                                                        (confidence 78%)
   ```

3. You get a one-line prompt: `Run this search? [y]es / [e]dit /
   [s]kip >`.
   - `y` (or just Enter on accept-default builds) → dispatches
     `cmd_search` immediately with `--mode applied` prepended.
   - `e` → second prompt: edit the query in-place, mode is
     preserved. Empty edit aborts.
   - `s` (or `n`) → noop.

The LLM tool form (`"what should I search next?"`) returns the same
suggestion as a single line so the model can chain into
`search_arxiv`. The Searcher anchors its proposal on the **previous**
query (stored in `ChatSession.last_search_query`), so it won't just
repeat what you already searched.

When to use: any time the conversation reveals a new angle (the
Critic raised a missing baseline, the user pivoted to a sub-problem).
Skip when the transcript is too thin — Searcher will return an empty
query and you'll see a `Searcher returned no refined query` hint
instead of garbage.

---

## §9. Research insights — your local activity dashboard

Time for the weekly review. The `/insights` slash (and the
`research insights` Typer subcommand, runnable outside the REPL)
roll up everything in your local SQLite into a Markdown report.
**No LLM call** — pure aggregation, instant, reproducible.

```text
/insights
```

You'll see a Markdown document with three sections:

```
# Research Insights — all-time
_Generated 2026-05-26 04:13:21 UTC_

## Papers
- Total: 3
- By year:
  - 2024: 2
  - 2023: 1
- Top tags:
  - nlp (2)
  - transformers (1)
- Top venues:
  - ICML (2)
  - NeurIPS (1)
- Top authors:
  - Vaswani (1)
  - ...

## Ideas
- Total: 1
- By status:
  - shelved: 1
- Average critic score: 6.00/9
- Most-debated ideas (by score history length):
  - `<prefix>` Top-k sparse self-attention with learned routing — 1 round(s)
- Highest-scoring ideas:
  - `<prefix>` Top-k sparse self-attention with learned routing — 6/9

## Discussions
- Sessions: 2
- Messages: 12
- By role:
  - user: 4
  - analyst: 3
  - critic: 3
  - system: 2
- Most recent sessions:
  - `<prefix>` — 2026-05-26 04:11:42
```

Constrain to a window with `--since`:

```text
/insights --since 7d
/insights --since 6m
/insights --since=30d
/insights --since all
```

`d` / `w` / `m` / `y` units are accepted; plain integers are days.

The same report is available outside the REPL — handy for
committing a snapshot to git or piping into a script:

```bash
research insights --since 30d                          # print to stdout
research insights --since 7d --output reports/this-week.md  # write to disk
```

Use it for: weekly research review, "am I reading too narrowly?"
self-check, finding the idea you've debated the most but never
saved.

---

## §10. Tear-down checklist

```text
/queue list all       # 1706.03762=done, others=pending
/ideas                # the sparse-routing idea is there with a score
/exit
```

After exit, the shell prints
`✓ Session saved (N turn(s), M message(s), id=…)`.

Inspect the on-disk state if you want to be thorough:

```bash
sqlite3 ~/.research-agent/memory.db "SELECT arxiv_id, status, completed_at FROM reading_queue;"
sqlite3 ~/.research-agent/memory.db "SELECT role, substr(content, 1, 60) FROM discussions ORDER BY created_at DESC LIMIT 10;"
sqlite3 ~/.research-agent/memory.db "SELECT query, COUNT(*) FROM search_queries q JOIN search_results r ON r.query_id=q.id GROUP BY q.id;"
```

You should see:

- `reading_queue`: at least one row with `status=done` and a
  non-null `completed_at`.
- `discussions`: user / analyst / critic rows from §5.
- Two or three search_queries rows from §1 and §6.
- `~/.research-agent/chroma/` exists (vector indexes for ideas + the
  new `discussions` collection if you ran the demo with a real
  OpenRouter key; otherwise the in-memory fallback was used).

---

## Tips for varying the demo

- **Swap the paper**: any arXiv id works. Good alternatives —
  `2005.14165` (GPT-3), `1810.04805` (BERT), `2010.11929` (ViT),
  `2106.09685` (LoRA). Pick something with an obvious "yes-but-also"
  trade-off so the Critic has something to bite into.
- **Swap the idea**: keep it specific and falsifiable. Vague ideas
  ("make transformers better") collapse the debate into platitudes.
- **Skip §7**: if you can't be bothered to restart the REPL, the
  recall path still exercises end-to-end via the LLM tool — just
  seed a fresh session by exiting and re-running `research`.
- **Mock mode**: setting `RESEARCH_AGENT_TEST_MODE=1` forces the
  keyword fallback for chroma; useful if you don't want a real
  vector index on disk.
- **Tune the parked-idea alert**: `research config set
  alert_threshold 0.7` to surface more reminders (looser
  similarity), or `0.9` for only near-duplicates. The threshold is
  shown in `research config show`.
- **Demo `/refine` on a thinner transcript**: start a fresh REPL,
  run only `/search` + `/read` (no `/discuss`), then `/refine` — you
  should see the "Searcher returned no refined query" path, since
  there's no debate signal to mine.
- **Commit a weekly report**: `research insights --since 7d --output
  reports/$(date +%F).md` makes the dashboard a git-friendly
  artefact.
