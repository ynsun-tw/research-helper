"""Searcher agent — relevance scoring for /search hits (M3 E3.1).

For now the agent only re-scores arXiv hits produced by ``ArxivSearcher``.
Multi-source search + citation graph (S3.1.2) will hang off this same
``score_hits`` interface.
"""

from __future__ import annotations

from typing import Any

from research_agent.agents.base import AgentResponse, BaseAgent, extract_json
from research_agent.core.llm import LLMError
from research_agent.search.arxiv_search import ArxivSearchHit


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
