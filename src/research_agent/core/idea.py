"""Idea entity and lifecycle rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

IdeaStatus = Literal["active", "shelved", "waiting", "experimenting", "abandoned", "completed"]

IDEA_STATUSES: tuple[IdeaStatus, ...] = (
    "active",
    "shelved",
    "waiting",
    "experimenting",
    "abandoned",
    "completed",
)

ALLOWED_TRANSITIONS: dict[IdeaStatus, frozenset[IdeaStatus]] = {
    "active": frozenset({"shelved", "waiting", "experimenting", "abandoned", "completed"}),
    "waiting": frozenset({"active", "shelved", "abandoned"}),
    "shelved": frozenset({"active", "abandoned"}),
    "experimenting": frozenset({"active", "completed", "abandoned"}),
    "abandoned": frozenset({"active"}),
    "completed": frozenset({"active"}),
}


@dataclass(slots=True)
class ScoreHistoryEntry:
    score: float
    reason: str
    session_id: str = ""


@dataclass(slots=True)
class Idea:
    id: str
    title: str
    description: str = ""
    status: IdeaStatus = "active"
    analyst_score: float | None = None
    critic_score: float | None = None
    critic_objections: list[str] = field(default_factory=list)
    related_papers: list[str] = field(default_factory=list)
    score_history: list[ScoreHistoryEntry] = field(default_factory=list)
    user_score_feedback: list[str] = field(default_factory=list)


class InvalidStatusTransition(Exception):
    """Raised when an Idea status change is not allowed."""


def can_transition(current: IdeaStatus, new: IdeaStatus) -> bool:
    if current == new:
        return True
    return new in ALLOWED_TRANSITIONS.get(current, frozenset())
