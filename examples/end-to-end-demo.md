# End-to-End Demo: Transformer + Sparse Attention

A 10-minute scripted walkthrough that exercises every M1 → M3 feature on a
real paper. Use it to smoke-test the CLI after pulling, to demo the agent
to someone, or as a starting point for your own session.

## What gets covered

| Capability | Slash | LLM tool | Step |
|---|---|---|---|
| arXiv search + LLM relevance scoring (M3 task 1) | `/search` | `search_arxiv` | §1, §6 |
| Search history persistence (M3.1) | `/history` | `recent_searches` | §2, §6 |
| Reading queue (M3 task 3) | `/queue add\|list\|next\|read\|done\|skip\|remove` | `queue_add`, `queue_list`, `queue_next` | §3 |
| Paper load + Analyst + Critic (M1) | `/read` | `load_paper` | §4 |
| Two-phase debate (M2 + M2 fix) | `/discuss` | `discuss_idea` | §5 |
| Save / inspect ideas (M2) | `/idea save`, `/ideas`, `/idea show` | `save_current_idea`, `list_ideas` | §5 |
| Cross-session recall (M3 task 2) | `/recall` | `recall_history` | §7 |
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

## §8. Tear-down checklist

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
