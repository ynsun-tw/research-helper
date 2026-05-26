"""MetaMemory — aggregate analytics across papers, ideas, and discussions (M3 S3.3.2).

This module is the cross-session "research dashboard". Unlike ``MemoryKeeper``
(which surfaces individual recalls), MetaMemory rolls up:

- **Research interests**: which topics / venues / years dominate the user's
  reading list (T3.3.2.2).
- **Decision patterns**: how ideas evolve - current status distribution,
  engagement (debate rounds), and stand-out winners / losers (T3.3.2.3).
- **Activity**: session and message counts over time (helps the user see
  research velocity, not just volume).

All inputs come from SQLite (`papers`, `ideas`, `discussions`). We
intentionally do NOT call the LLM here - this is a deterministic local
report that should run instantly and stay reproducible.

The ``/insights`` slash + ``research insights`` Typer subcommand render the
report (T3.3.2.4) as a single Markdown blob via ``InsightsReport.to_markdown``
plus a Rich console summary.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from research_agent.core.idea import IDEA_STATUSES
from research_agent.storage.database import Database

# Number of items shown in "Top N" tables. Same cap everywhere keeps the
# Markdown output predictable.
TOP_N = 5


@dataclass(slots=True)
class InsightsReport:
    """All numbers MetaMemory computes for one report."""

    period: str  # "all-time" or "last <N> days"
    generated_at: str  # ISO timestamp for the report header

    # Papers
    paper_count: int = 0
    papers_by_year: list[tuple[int, int]] = field(default_factory=list)
    top_tags: list[tuple[str, int]] = field(default_factory=list)
    top_authors: list[tuple[str, int]] = field(default_factory=list)
    top_venues: list[tuple[str, int]] = field(default_factory=list)

    # Ideas
    idea_count: int = 0
    ideas_by_status: dict[str, int] = field(default_factory=dict)
    avg_critic_score: float | None = None
    most_engaged_ideas: list[tuple[str, str, int]] = field(default_factory=list)
    """(id, title, score_history length) sorted by engagement desc."""
    top_scored_ideas: list[tuple[str, str, float]] = field(default_factory=list)
    """(id, title, critic_score) sorted by score desc."""

    # Discussions
    session_count: int = 0
    message_count: int = 0
    role_counts: dict[str, int] = field(default_factory=dict)
    recent_session_ids: list[tuple[str, str]] = field(default_factory=list)
    """(session_id, latest message ISO timestamp) - newest first."""

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Research Insights — {self.period}")
        lines.append(f"_Generated {self.generated_at}_\n")

        # --- Papers ---
        lines.append("## Papers")
        lines.append(f"- Total: **{self.paper_count}**")
        if self.papers_by_year:
            lines.append("- By year:")
            for year, count in self.papers_by_year:
                lines.append(f"  - {year}: {count}")
        if self.top_tags:
            lines.append("- Top tags:")
            for tag, count in self.top_tags:
                lines.append(f"  - {tag} ({count})")
        if self.top_venues:
            lines.append("- Top venues:")
            for venue, count in self.top_venues:
                lines.append(f"  - {venue} ({count})")
        if self.top_authors:
            lines.append("- Top authors:")
            for author, count in self.top_authors:
                lines.append(f"  - {author} ({count})")
        lines.append("")

        # --- Ideas ---
        lines.append("## Ideas")
        lines.append(f"- Total: **{self.idea_count}**")
        if self.ideas_by_status:
            lines.append("- By status:")
            for status in IDEA_STATUSES:
                if self.ideas_by_status.get(status):
                    lines.append(
                        f"  - {status}: {self.ideas_by_status[status]}"
                    )
        if self.avg_critic_score is not None:
            lines.append(
                f"- Average critic score: **{self.avg_critic_score:.2f}/9**"
            )
        if self.most_engaged_ideas:
            lines.append("- Most-debated ideas (by score history length):")
            for idea_id, title, n_rounds in self.most_engaged_ideas:
                lines.append(
                    f"  - `{idea_id[:8]}` {title} — {n_rounds} round(s)"
                )
        if self.top_scored_ideas:
            lines.append("- Highest-scoring ideas:")
            for idea_id, title, score in self.top_scored_ideas:
                lines.append(f"  - `{idea_id[:8]}` {title} — {score:.0f}/9")
        lines.append("")

        # --- Discussions ---
        lines.append("## Discussions")
        lines.append(f"- Sessions: **{self.session_count}**")
        lines.append(f"- Messages: **{self.message_count}**")
        if self.role_counts:
            lines.append("- By role:")
            for role, count in sorted(
                self.role_counts.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  - {role}: {count}")
        if self.recent_session_ids:
            lines.append("- Most recent sessions:")
            for sid, ts in self.recent_session_ids:
                lines.append(f"  - `{sid[:8]}…` — {ts}")
        return "\n".join(lines)


class MetaMemory:
    """Compute :class:`InsightsReport` snapshots from the SQLite database."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def compute(
        self,
        *,
        since_days: int | None = None,
        top_n: int = TOP_N,
    ) -> InsightsReport:
        """Roll up the database into a single :class:`InsightsReport`.

        ``since_days`` filters papers / ideas / discussions by their
        ``created_at`` timestamp. ``None`` returns the all-time picture.
        """
        now = datetime.now(tz=UTC)
        since_clause = ""
        params: tuple[str, ...] = ()
        period = "all-time"
        if since_days is not None and since_days > 0:
            since_ts = now - timedelta(days=since_days)
            since_clause = " WHERE created_at >= ?"
            params = (since_ts.strftime("%Y-%m-%d %H:%M:%S"),)
            period = f"last {since_days} days"

        report = InsightsReport(
            period=period,
            generated_at=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        self._fill_papers(report, since_clause, params, top_n)
        self._fill_ideas(report, since_clause, params, top_n)
        self._fill_discussions(report, since_clause, params, top_n)
        return report

    # ------------------------------------------------------------------ papers

    def _fill_papers(
        self,
        report: InsightsReport,
        since_clause: str,
        params: tuple[str, ...],
        top_n: int,
    ) -> None:
        rows = self.db.conn.execute(
            f"SELECT year, tags, authors, venue FROM papers{since_clause}",
            params,
        ).fetchall()
        report.paper_count = len(rows)

        year_counter: Counter[int] = Counter()
        tag_counter: Counter[str] = Counter()
        author_counter: Counter[str] = Counter()
        venue_counter: Counter[str] = Counter()
        for row in rows:
            year = row["year"]
            if isinstance(year, int):
                year_counter[year] += 1
            for tag in _parse_json_list(row["tags"]):
                tag = tag.strip()
                if tag:
                    tag_counter[tag] += 1
            for author in _parse_json_list(row["authors"]):
                author = author.strip()
                if author:
                    author_counter[author] += 1
            venue = (row["venue"] or "").strip()
            if venue:
                venue_counter[venue] += 1

        report.papers_by_year = sorted(year_counter.items(), reverse=True)
        report.top_tags = tag_counter.most_common(top_n)
        report.top_authors = author_counter.most_common(top_n)
        report.top_venues = venue_counter.most_common(top_n)

    # ------------------------------------------------------------------- ideas

    def _fill_ideas(
        self,
        report: InsightsReport,
        since_clause: str,
        params: tuple[str, ...],
        top_n: int,
    ) -> None:
        rows = self.db.conn.execute(
            f"SELECT id, title, status, critic_score, score_history "
            f"FROM ideas{since_clause}",
            params,
        ).fetchall()
        report.idea_count = len(rows)
        status_counter: Counter[str] = Counter()
        critic_scores: list[float] = []
        engagement: list[tuple[str, str, int]] = []
        scored: list[tuple[str, str, float]] = []
        for row in rows:
            status = (row["status"] or "active").strip()
            status_counter[status] += 1
            score = row["critic_score"]
            if isinstance(score, (int, float)):
                critic_scores.append(float(score))
                scored.append((row["id"], row["title"], float(score)))
            history = _parse_json_list(row["score_history"])
            engagement.append((row["id"], row["title"], len(history)))

        report.ideas_by_status = dict(status_counter)
        if critic_scores:
            report.avg_critic_score = sum(critic_scores) / len(critic_scores)
        engagement.sort(key=lambda t: (-t[2], t[1]))
        report.most_engaged_ideas = [e for e in engagement if e[2] > 0][:top_n]
        scored.sort(key=lambda t: (-t[2], t[1]))
        report.top_scored_ideas = scored[:top_n]

    # ------------------------------------------------------------- discussions

    def _fill_discussions(
        self,
        report: InsightsReport,
        since_clause: str,
        params: tuple[str, ...],
        top_n: int,
    ) -> None:
        rows = self.db.conn.execute(
            f"SELECT session_id, role, created_at FROM discussions"
            f"{since_clause}",
            params,
        ).fetchall()
        report.message_count = len(rows)
        sessions: set[str] = set()
        role_counter: Counter[str] = Counter()
        latest_by_session: dict[str, str] = {}
        for row in rows:
            sid = row["session_id"]
            sessions.add(sid)
            role_counter[row["role"]] += 1
            ts = row["created_at"]
            if isinstance(ts, str):
                existing = latest_by_session.get(sid, "")
                if ts > existing:
                    latest_by_session[sid] = ts
        report.session_count = len(sessions)
        report.role_counts = dict(role_counter)
        recent = sorted(latest_by_session.items(), key=lambda kv: kv[1], reverse=True)
        report.recent_session_ids = recent[:top_n]


def _parse_json_list(raw: object) -> list[str]:
    """Best-effort JSON-list parser. Returns ``[]`` for NULL / malformed."""
    if not raw or not isinstance(raw, str):
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]
