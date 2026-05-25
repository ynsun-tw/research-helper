"""Format anchor papers for LLM prompts."""

from __future__ import annotations

from research_agent.core.paper import Paper

DEFAULT_PAPER_PROMPT_CHARS = 14_000


def format_paper_for_prompt(paper: Paper, *, max_chars: int = DEFAULT_PAPER_PROMPT_CHARS) -> str:
    """Serialize paper metadata and excerpt for debate grounding."""
    authors = ", ".join(paper.authors) if paper.authors else "Unknown"
    sections = "\n\n".join(
        f"### {s.title}\n{s.content[:2500]}" for s in paper.sections[:10]
    )
    body = paper.full_text[:max_chars] if paper.full_text else sections
    if len(body) > max_chars:
        body = body[:max_chars] + "\n…(truncated)"
    year = str(paper.year) if paper.year else "n/a"
    return (
        f"Title: {paper.title}\n"
        f"Authors: {authors}\n"
        f"ID: {paper.id}\n"
        f"Year: {year}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        f"Sections:\n{sections or '(none detected)'}\n\n"
        f"Full text excerpt:\n{body}"
    )
