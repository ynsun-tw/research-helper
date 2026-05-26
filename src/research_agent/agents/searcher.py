"""Searcher agent — relevance scoring + dynamic refinement (M3 E3.1, E3.2).

Responsibilities now:
- ``score_hits``: rate arXiv (or S2-fallback) candidates 0-1 with a one-line
  reason. Used by ``/search`` to rank.
- ``suggest_refinement``: read the discussion transcript and propose a new
  search query + bias mode whenever the user wants to pivot. Used by the
  ``/refine`` slash and the ``suggest_search_refinement`` LLM tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_agent.agents.base import AgentResponse, BaseAgent, extract_json
from research_agent.core.llm import ChatMessage, LLMError
from research_agent.search.arxiv_search import ArxivSearchHit

VALID_MODES = {"theoretical", "applied"}


@dataclass(slots=True)
class SearchSuggestion:
    """A pivot suggestion derived from the discussion so far.

    ``mode`` is either ``None``, ``"theoretical"``, ``"applied"``, or
    ``"group:<author-name>"`` - same vocabulary as ``/search --mode``.
    ``confidence`` is a 0-1 self-rated confidence from the LLM; callers
    can use it to decide whether to auto-run or just display.
    """

    query: str
    mode: str | None
    reason: str
    confidence: float = 0.0


class Searcher(BaseAgent):
    role = "searcher"

    @property
    def prompt_name(self) -> str:
        return "searcher"

    def run(self, context: dict[str, Any]) -> AgentResponse:  # pragma: no cover
        raise NotImplementedError(
            "Searcher.run is not used directly; call score_hits instead."
        )

    def score_hits(
        self,
        query: str,
        hits: list[ArxivSearchHit],
        *,
        temperature: float = 0.2,
    ) -> list[ArxivSearchHit]:
        """Return a new list of hits annotated with relevance_score + reason.

        Falls back to ``score=None`` on LLM or parse failure so /search can
        still render and persist results.
        """
        if not hits:
            return []
        prompt = _build_prompt(query, hits)
        try:
            raw = self._chat(prompt, temperature=temperature)
            data = extract_json(raw)
        except (LLMError, ValueError):
            return list(hits)

        by_index: dict[int, tuple[float, str]] = {}
        for entry in data.get("scores", []) or []:
            try:
                idx = int(entry.get("index"))
                score = float(entry.get("score"))
            except (TypeError, ValueError):
                continue
            if not 0.0 <= score <= 1.0:
                score = max(0.0, min(score, 1.0))
            reason = str(entry.get("reason", "")).strip()
            by_index[idx] = (score, reason)

        out: list[ArxivSearchHit] = []
        for i, hit in enumerate(hits, start=1):
            scored = by_index.get(i)
            if scored is None:
                out.append(hit)
            else:
                score, reason = scored
                out.append(
                    ArxivSearchHit(
                        arxiv_id=hit.arxiv_id,
                        title=hit.title,
                        abstract=hit.abstract,
                        published=hit.published,
                        relevance_score=score,
                        relevance_reason=reason,
                    )
                )
        return out


    def suggest_refinement(
        self,
        discussion_context: str,
        *,
        previous_query: str | None = None,
        temperature: float = 0.4,
    ) -> SearchSuggestion:
        """Read recent discussion + last query and propose a refined search.

        On LLM failure / unparsable output we return a degenerate suggestion
        (empty query, empty reason, confidence=0.0) so callers can short-
        circuit without crashing. The slash command treats this as a no-op.
        """
        context = (discussion_context or "").strip()
        if not context:
            return SearchSuggestion(query="", mode=None, reason="", confidence=0.0)

        prompt = _build_refinement_prompt(context, previous_query)
        try:
            messages = [
                ChatMessage(role="system", content=_REFINEMENT_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ]
            raw = self.llm.chat(messages, temperature=temperature)
            data = extract_json(raw)
        except (LLMError, ValueError):
            return SearchSuggestion(query="", mode=None, reason="", confidence=0.0)

        return _parse_refinement(data)


def _parse_refinement(data: dict[str, Any]) -> SearchSuggestion:
    query = str(data.get("query") or "").strip()
    raw_mode = data.get("mode")
    mode: str | None = (
        None if raw_mode in (None, "") else (str(raw_mode).strip() or None)
    )
    # Validate mode shape; unknown values dropped (don't break /search).
    if mode is not None:
        normalized = mode.strip()
        if normalized.startswith("group:"):
            author = normalized[len("group:"):].strip()
            mode = f"group:{author}" if author else None
        elif normalized.lower() in VALID_MODES:
            mode = normalized.lower()
        else:
            mode = None
    reason = str(data.get("reason") or "").strip()
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return SearchSuggestion(query=query, mode=mode, reason=reason, confidence=confidence)


_REFINEMENT_SYSTEM_PROMPT = """You are Searcher, advising on the *next* literature search.

The user is in a research conversation. You will see:
- The previous search query (may be empty if this is the first search).
- A short transcript of the most recent discussion (analyst/critic/user
  messages, oldest first).

Your job: propose ONE refined search query that is most likely to surface
new, relevant papers given where the conversation went. You may also
suggest a search bias mode:

- "theoretical" — bias toward analysis / proofs / formal papers
- "applied" — bias toward benchmarks / experiments / systems papers
- "group:<author-name>" — bias toward a specific author group
- null — no mode bias

Honesty rules:
- If the transcript is too thin to refine, return an empty query and
  set confidence near 0; do not invent topics.
- Do not just repeat the previous query verbatim; either tighten it,
  pivot, or expand it based on what the discussion uncovered.
- Keep the query short (3-8 keywords).
- Keep the reason to ONE sentence (max ~25 words).
- Confidence is a 0-1 self-rating.

Output STRICTLY this JSON object and nothing else:
{
  "query": "...",
  "mode": "theoretical" | "applied" | "group:<author>" | null,
  "reason": "...",
  "confidence": 0.0
}
"""


def _build_refinement_prompt(context: str, previous_query: str | None) -> str:
    prev = (previous_query or "").strip() or "(none)"
    # Cap transcript to a generous budget; orchestrator already truncates,
    # but we want our own ceiling so a runaway memory can't OOM the prompt.
    capped = context[-4000:]
    lines = [
        f'Previous query: "{prev}"',
        "",
        "Recent discussion transcript (oldest first):",
        capped,
        "",
        "Propose the next search.",
    ]
    return "\n".join(lines)


def _build_prompt(query: str, hits: list[ArxivSearchHit]) -> str:
    lines = [f'User query: "{query}"', "", "Candidates:"]
    for i, h in enumerate(hits, start=1):
        title = h.title.strip() or "(untitled)"
        abstract = (h.abstract or "").strip() or "(no abstract)"
        if len(abstract) > 1200:
            abstract = abstract[:1200].rstrip() + "…"
        lines.append(f"[{i}] {h.arxiv_id} — {title}")
        lines.append(f"    Abstract: {abstract}")
    lines.append("")
    lines.append(
        "Score every candidate. Respond with the JSON described in the system prompt."
    )
    return "\n".join(lines)
