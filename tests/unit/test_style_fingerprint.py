"""Unit tests for :class:`Fingerprint` (de)serialization."""

from __future__ import annotations

import json
from pathlib import Path

from research_agent.style.fingerprint import (
    Fingerprint,
    MacroFingerprint,
    MicroFingerprint,
    PersonalMarkers,
)


def _filled() -> Fingerprint:
    return Fingerprint(
        macro=MacroFingerprint(
            abstract_opener="we propose a new",
            abstract_avg_sentences=4.5,
            intro_opener="recurrent models have long",
            intro_avg_paragraphs=3.0,
            related_work_strategy="thematic",
            section_count_avg=6.5,
            notes="",
        ),
        micro=MicroFingerprint(
            avg_sentence_length=21.0,
            median_sentence_length=20.0,
            p10_sentence_length=10.0,
            p90_sentence_length=35.0,
            avg_paragraph_length=4.5,
            sentence_count=120,
            transition_freq={"however": 2.5},
            hedging_per_100=3.2,
            confidence_per_100=1.1,
            passive_per_100=4.0,
            type_token_ratio=0.42,
        ),
        markers=PersonalMarkers(
            top_section_titles=["Introduction", "Method"],
            citation_format="latex_cite",
            figure_ref_format="Figure",
            table_ref_format="Table",
            em_dash_per_100=0.5,
        ),
        sample_count=42,
        paper_count=3,
        version=1,
    )


def test_to_dict_round_trip() -> None:
    fp = _filled()
    d = fp.to_dict()
    restored = Fingerprint.from_dict(d)
    assert restored.macro.abstract_opener == fp.macro.abstract_opener
    assert restored.micro.avg_sentence_length == fp.micro.avg_sentence_length
    assert restored.micro.transition_freq == {"however": 2.5}
    assert restored.markers.citation_format == "latex_cite"
    assert restored.sample_count == 42


def test_save_and_load_to_disk(tmp_path: Path) -> None:
    fp = _filled()
    path = tmp_path / "fp.json"
    fp.save_to(path)
    assert path.exists()
    assert fp.created_at  # save populates a timestamp

    loaded = Fingerprint.load_from(path)
    assert loaded.micro.transition_freq == {"however": 2.5}
    assert loaded.created_at == fp.created_at
    assert loaded.paper_count == 3


def test_from_dict_tolerates_extra_and_missing_keys(tmp_path: Path) -> None:
    raw = {
        "macro": {"abstract_opener": "x", "obsolete_field": 42},
        "micro": {"avg_sentence_length": 12.5},
        "markers": {},
        "sample_count": 7,
        "future_top_level": "ignored",
    }
    fp = Fingerprint.from_dict(raw)
    assert fp.macro.abstract_opener == "x"
    assert fp.micro.avg_sentence_length == 12.5
    assert fp.sample_count == 7
    assert fp.paper_count == 0


def test_load_from_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "fp.json"
    path.write_text(json.dumps(["not a dict"]), encoding="utf-8")
    try:
        Fingerprint.load_from(path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on non-dict payload")


def test_save_to_preserves_history(tmp_path: Path) -> None:
    path = tmp_path / "fp.json"
    v1 = _filled()
    v1.save_to(path)
    assert v1.version == 1

    v2 = _filled()
    v2.save_to(path, preserve_history=True)
    # The new fingerprint bumped to v2 and the prior file moved aside.
    assert v2.version == 2
    archive = tmp_path / "fp_v1.json"
    assert archive.exists()
    loaded_archive = Fingerprint.load_from(archive)
    assert loaded_archive.version == 1
    # Current file holds v2.
    assert Fingerprint.load_from(path).version == 2

    v3 = _filled()
    v3.save_to(path, preserve_history=True)
    assert v3.version == 3
    assert (tmp_path / "fp_v2.json").exists()


def test_save_to_with_no_history_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "fp.json"
    _filled().save_to(path)
    second = _filled()
    second.save_to(path)  # default preserve_history=False
    # No archive created.
    assert not (tmp_path / "fp_v1.json").exists()
    # Version still 1 (no bump because no history kept).
    assert Fingerprint.load_from(path).version == 1


def test_save_to_archive_slot_collision(tmp_path: Path) -> None:
    """When the archive name is already taken (re-run with same v) we
    append an incrementing suffix instead of clobbering it."""
    path = tmp_path / "fp.json"
    _filled().save_to(path)
    # Pre-create the v1 archive slot
    (tmp_path / "fp_v1.json").write_text("{}", encoding="utf-8")
    _filled().save_to(path, preserve_history=True)
    # Original archive remained, suffix-numbered archive was used instead.
    assert (tmp_path / "fp_v1.json").exists()
    assert (tmp_path / "fp_v1-1.json").exists()
