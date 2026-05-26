"""Idea persistence and lifecycle (SQLite)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict

from research_agent.core.idea import (
    IDEA_STATUSES,
    Idea,
    IdeaStatus,
    InvalidStatusTransition,
    ScoreHistoryEntry,
    can_transition,
)
from research_agent.storage.database import Database


class IdeaRepository:
    """CRUD and score history for :class:`Idea`."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        title: str,
        description: str = "",
        *,
        idea_id: str | None = None,
        status: IdeaStatus = "active",
    ) -> Idea:
        iid = idea_id or str(uuid.uuid4())
        idea = Idea(id=iid, title=title, description=description, status=status)
        with self.db.conn:
            self.db.conn.execute(
                """
                INSERT INTO ideas (
                    id, title, description, status,
                    analyst_score, critic_score, critic_objections,
                    related_papers, score_history, user_score_feedback,
                    activation_conditions
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idea.id,
                    idea.title,
                    idea.description,
                    idea.status,
                    idea.analyst_score,
                    idea.critic_score,
                    json.dumps(idea.critic_objections, ensure_ascii=False),
                    json.dumps(idea.related_papers, ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                ),
            )
        return idea

    def get(self, idea_id: str) -> Idea | None:
        row = self.db.conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        return _row_to_idea(row) if row else None

    def save(self, idea: Idea) -> None:
        with self.db.conn:
            self.db.conn.execute(
                """
                INSERT INTO ideas (
                    id, title, description, status,
                    analyst_score, critic_score, critic_objections,
                    related_papers, score_history, user_score_feedback,
                    activation_conditions
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    status = excluded.status,
                    analyst_score = excluded.analyst_score,
                    critic_score = excluded.critic_score,
                    critic_objections = excluded.critic_objections,
                    related_papers = excluded.related_papers,
                    score_history = excluded.score_history,
                    user_score_feedback = excluded.user_score_feedback,
                    activation_conditions = excluded.activation_conditions,
                    updated_at = CURRENT_TIMESTAMP
                """,
                _idea_to_row(idea),
            )

    def list_all(self) -> list[Idea]:
        rows = self.db.conn.execute(
            "SELECT * FROM ideas ORDER BY updated_at DESC, created_at DESC"
        ).fetchall()
        return [_row_to_idea(r) for r in rows]

    def list_by_status(self) -> dict[IdeaStatus, list[Idea]]:
        grouped: dict[IdeaStatus, list[Idea]] = {s: [] for s in IDEA_STATUSES}
        for idea in self.list_all():
            grouped.setdefault(idea.status, []).append(idea)
        return grouped

    def update_status(self, idea_id: str, new_status: IdeaStatus) -> Idea:
        idea = self.get(idea_id)
        if idea is None:
            raise KeyError(f"Idea not found: {idea_id}")
        if not can_transition(idea.status, new_status):
            raise InvalidStatusTransition(
                f"Cannot transition idea {idea_id} from {idea.status!r} to {new_status!r}"
            )
        idea.status = new_status
        self.save(idea)
        return idea

    def append_score(
        self,
        idea_id: str,
        score: float,
        reason: str,
        *,
        session_id: str = "",
    ) -> Idea:
        idea = self.get(idea_id)
        if idea is None:
            raise KeyError(f"Idea not found: {idea_id}")
        entry = ScoreHistoryEntry(score=score, reason=reason, session_id=session_id)
        idea.score_history.append(entry)
        idea.critic_score = score
        self.save(idea)
        return idea

    def add_user_score_feedback(self, idea_id: str, note: str) -> Idea:
        idea = self.get(idea_id)
        if idea is None:
            raise KeyError(f"Idea not found: {idea_id}")
        idea.user_score_feedback.append(note.strip())
        self.save(idea)
        return idea

    def add_activation_condition(self, idea_id: str, condition: str) -> Idea:
        """Append a free-form activation condition (deduped, case-insensitive).

        Conditions are user-supplied phrases describing what *would* unblock a
        shelved idea (e.g. "needs FineWeb-Edu dataset", "1B model checkpoint
        released"). The search-result alert in `chat/tools` looks for these
        phrases inside incoming paper titles/abstracts.
        """
        idea = self.get(idea_id)
        if idea is None:
            raise KeyError(f"Idea not found: {idea_id}")
        cleaned = condition.strip()
        if not cleaned:
            return idea
        lowered = cleaned.lower()
        existing_lowered = {c.lower() for c in idea.activation_conditions}
        if lowered in existing_lowered:
            return idea
        idea.activation_conditions.append(cleaned)
        self.save(idea)
        return idea

    def clear_activation_conditions(self, idea_id: str) -> Idea:
        idea = self.get(idea_id)
        if idea is None:
            raise KeyError(f"Idea not found: {idea_id}")
        idea.activation_conditions = []
        self.save(idea)
        return idea

    def session_ids(self, idea_id: str) -> list[str]:
        rows = self.db.conn.execute(
            """
            SELECT DISTINCT session_id FROM discussions
            WHERE idea_id = ?
            ORDER BY session_id
            """,
            (idea_id,),
        ).fetchall()
        return [str(r["session_id"]) for r in rows]


def _idea_to_row(idea: Idea) -> tuple[object, ...]:
    return (
        idea.id,
        idea.title,
        idea.description,
        idea.status,
        idea.analyst_score,
        idea.critic_score,
        json.dumps(idea.critic_objections, ensure_ascii=False),
        json.dumps(idea.related_papers, ensure_ascii=False),
        json.dumps([asdict(e) for e in idea.score_history], ensure_ascii=False),
        json.dumps(idea.user_score_feedback, ensure_ascii=False),
        json.dumps(idea.activation_conditions, ensure_ascii=False),
    )


def _row_to_idea(row: sqlite3.Row) -> Idea:
    objections = json.loads(row["critic_objections"]) if row["critic_objections"] else []
    related = json.loads(row["related_papers"]) if row["related_papers"] else []
    history_raw = row["score_history"]
    feedback_raw = row["user_score_feedback"]
    history: list[ScoreHistoryEntry] = []
    if history_raw:
        for item in json.loads(history_raw):
            if isinstance(item, dict):
                history.append(
                    ScoreHistoryEntry(
                        score=float(item.get("score", 0)),
                        reason=str(item.get("reason", "")),
                        session_id=str(item.get("session_id", "")),
                    )
                )
    feedback: list[str] = json.loads(feedback_raw) if feedback_raw else []
    # `activation_conditions` was added in M3 T3.4.2 and is missing on older
    # rows; tolerate absence and treat as empty.
    try:
        conditions_raw = row["activation_conditions"]
    except (IndexError, KeyError):
        conditions_raw = None
    conditions: list[str] = json.loads(conditions_raw) if conditions_raw else []
    return Idea(
        id=row["id"],
        title=row["title"],
        description=row["description"] or "",
        status=row["status"] or "active",
        analyst_score=row["analyst_score"],
        critic_score=row["critic_score"],
        critic_objections=[str(o) for o in objections],
        related_papers=[str(p) for p in related],
        score_history=history,
        user_score_feedback=[str(f) for f in feedback],
        activation_conditions=[str(c) for c in conditions],
    )
