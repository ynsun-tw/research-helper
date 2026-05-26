"""Unit tests for ``research style train|show`` CLI surface + service.

We drive ``run_style_train`` directly so we can monkeypatch the PDF
loader and avoid touching arXiv / disk in unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.cli_style import (
    run_style_fingerprint,
    run_style_history,
    run_style_show,
    run_style_train,
    run_style_update,
)
from research_agent.config import Config
from research_agent.core.paper import Paper, Section
from research_agent.storage.database import Database
from research_agent.storage.draft_revisions import DraftRevisionRepository
from research_agent.style.fingerprint import Fingerprint
from research_agent.style.samples import StyleSampleRepository

runner = CliRunner()


def _fake_paper(paper_id: str, body: str) -> Paper:
    return Paper(
        id=paper_id,
        title="A title",
        authors=["Alice"],
        abstract="",
        sections=[Section(title="Introduction", content=body)],
        full_text="",
        pdf_path=Path("/tmp/x.pdf"),
    )


_PROSE_A = (
    "Recurrent networks have dominated sequence modeling for years. "
    "We argue that they can be replaced by attention-only architectures. "
    "This section motivates that claim with three experiments."
)
_PROSE_B = (
    "Our second body paragraph extends the previous argument with new "
    "evidence from large-scale language pretraining. We show consistent "
    "improvements across all model sizes considered."
)


def test_style_train_imports_arxiv(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.load_paper",
        lambda src, *, cache_dir: _fake_paper(
            "arxiv:1234.5678", _PROSE_A + "\n\n" + _PROSE_B
        ),
    )
    cfg = Config.load(config_dir)
    result = run_style_train(cfg, Console(), sources=["arxiv:1234.5678"])
    assert result.sources_processed == 1
    assert result.sources_failed == 0
    assert result.paragraphs_added == 2

    db = Database(cfg.db_path)
    try:
        repo = StyleSampleRepository(db)
        assert repo.count() == 2
        assert repo.count_by_paper() == {"arxiv:1234.5678": 2}
    finally:
        db.close()


def test_style_train_replaces_prior_samples_by_default(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.load_paper",
        lambda src, *, cache_dir: _fake_paper(
            "arxiv:1234.5678", _PROSE_A + "\n\n" + _PROSE_B
        ),
    )
    cfg = Config.load(config_dir)
    run_style_train(cfg, Console(), sources=["arxiv:1234.5678"])
    # Second run on same source should NOT duplicate; replace mode default.
    result = run_style_train(cfg, Console(), sources=["arxiv:1234.5678"])
    assert result.paragraphs_added == 2

    db = Database(cfg.db_path)
    try:
        repo = StyleSampleRepository(db)
        assert repo.count() == 2  # not 4
    finally:
        db.close()


def test_style_train_append_keeps_prior(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.load_paper",
        lambda src, *, cache_dir: _fake_paper("arxiv:1234.5678", _PROSE_A + "\n\n" + _PROSE_B),
    )
    cfg = Config.load(config_dir)
    run_style_train(cfg, Console(), sources=["arxiv:1234.5678"])
    run_style_train(cfg, Console(), sources=["arxiv:1234.5678"], replace=False)
    db = Database(cfg.db_path)
    try:
        repo = StyleSampleRepository(db)
        assert repo.count() == 4
    finally:
        db.close()


def test_style_train_continues_on_bad_source(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    def fake_load(src: str, *, cache_dir: Path) -> Paper:
        if src == "arxiv:bad":
            raise FileNotFoundError("nope")
        return _fake_paper(src, _PROSE_A + "\n\n" + _PROSE_B)

    monkeypatch.setattr("research_agent.cli_style.load_paper", fake_load)
    cfg = Config.load(config_dir)
    result = run_style_train(cfg, Console(), sources=["arxiv:bad", "arxiv:ok"])
    assert result.sources_processed == 1
    assert result.sources_failed == 1
    assert result.paragraphs_added == 2


def test_style_train_directory_scan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_dir: Path
) -> None:
    folder = tmp_path / "papers"
    folder.mkdir()
    (folder / "one.pdf").write_bytes(b"%PDF-fake")
    (folder / "two.pdf").write_bytes(b"%PDF-fake")
    (folder / "ignore.txt").write_text("not a pdf")

    def fake_load(src: str, *, cache_dir: Path) -> Paper:
        return _fake_paper(f"local:{Path(src).stem}", _PROSE_A + "\n\n" + _PROSE_B)

    monkeypatch.setattr("research_agent.cli_style.load_paper", fake_load)
    cfg = Config.load(config_dir)
    result = run_style_train(cfg, Console(), directory=folder)
    assert result.sources_processed == 2
    assert result.paragraphs_added == 4


def test_style_train_no_sources_prints_hint(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    result = run_style_train(cfg, Console())
    assert result.sources_processed == 0
    assert result.paragraphs_added == 0


def test_style_show_empty_corpus(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    assert run_style_show(cfg, Console()) == 0


def test_style_show_after_training(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.load_paper",
        lambda src, *, cache_dir: _fake_paper("arxiv:9999.9999", _PROSE_A + "\n\n" + _PROSE_B),
    )
    cfg = Config.load(config_dir)
    run_style_train(cfg, Console(), sources=["arxiv:9999.9999"])
    assert run_style_show(cfg, Console()) == 0


def test_style_train_help() -> None:
    result = runner.invoke(app, ["style", "train", "--help"])
    assert result.exit_code == 0
    assert "arxiv" in result.stdout.lower()


def test_style_show_help() -> None:
    result = runner.invoke(app, ["style", "show", "--help"])
    assert result.exit_code == 0
    assert "summary" in result.stdout.lower()


def test_style_fingerprint_requires_samples(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    code = run_style_fingerprint(cfg, Console())
    assert code == 1
    assert not cfg.fingerprint_path.exists()


def test_style_fingerprint_writes_json(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.load_paper",
        lambda src, *, cache_dir: _fake_paper("arxiv:9999.9999", _PROSE_A + "\n\n" + _PROSE_B),
    )
    cfg = Config.load(config_dir)
    run_style_train(cfg, Console(), sources=["arxiv:9999.9999"])
    code = run_style_fingerprint(cfg, Console())
    assert code == 0
    assert cfg.fingerprint_path.exists()
    fp = Fingerprint.load_from(cfg.fingerprint_path)
    assert fp.sample_count == 2
    assert fp.paper_count == 1
    assert fp.micro.sentence_count > 0
    assert fp.created_at  # non-empty timestamp


def test_style_show_renders_fingerprint(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.load_paper",
        lambda src, *, cache_dir: _fake_paper("arxiv:9999.9999", _PROSE_A + "\n\n" + _PROSE_B),
    )
    cfg = Config.load(config_dir)
    run_style_train(cfg, Console(), sources=["arxiv:9999.9999"])
    run_style_fingerprint(cfg, Console())
    # show should run without raising even when both corpus + fp exist
    assert run_style_show(cfg, Console()) == 0


def test_style_fingerprint_help() -> None:
    result = runner.invoke(app, ["style", "fingerprint", "--help"])
    assert result.exit_code == 0
    assert "fingerprint" in result.stdout.lower()


# --- S4.1.3: continuous fingerprint learning -------------------------------


def test_style_update_requires_corpus(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    assert run_style_update(cfg, Console()) == 1


def test_style_update_combines_samples_and_revisions(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.load_paper",
        lambda src, *, cache_dir: _fake_paper("arxiv:0001.0001", _PROSE_A + "\n\n" + _PROSE_B),
    )
    cfg = Config.load(config_dir)
    run_style_train(cfg, Console(), sources=["arxiv:0001.0001"])
    run_style_fingerprint(cfg, Console())
    assert Fingerprint.load_from(cfg.fingerprint_path).version == 1

    db = Database(cfg.db_path)
    try:
        rev_repo = DraftRevisionRepository(db)
        rev_repo.add(
            section="introduction",
            original_text="Old.",
            revised_text=_PROSE_A + "\n\n" + _PROSE_B,
            interactive=True,
        )
    finally:
        db.close()

    code = run_style_update(cfg, Console())
    assert code == 0
    fp = Fingerprint.load_from(cfg.fingerprint_path)
    assert fp.version == 2
    # Two base prose paragraphs + two revision-derived = 4 samples.
    assert fp.sample_count == 4
    archive = cfg.style_dir / "fingerprint_v1.json"
    assert archive.exists()


def test_style_history_lists_versions(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    monkeypatch.setattr(
        "research_agent.cli_style.load_paper",
        lambda src, *, cache_dir: _fake_paper("arxiv:0001.0001", _PROSE_A + "\n\n" + _PROSE_B),
    )
    cfg = Config.load(config_dir)
    run_style_train(cfg, Console(), sources=["arxiv:0001.0001"])
    run_style_fingerprint(cfg, Console())
    # Bump twice
    run_style_update(cfg, Console())
    run_style_update(cfg, Console())
    code = run_style_history(cfg, Console())
    assert code == 0
    # Confirm filesystem has v1 + v2 archives + current fingerprint.json
    assert (cfg.style_dir / "fingerprint_v1.json").exists()
    assert (cfg.style_dir / "fingerprint_v2.json").exists()
    assert Fingerprint.load_from(cfg.fingerprint_path).version == 3


def test_style_history_no_style_dir(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    assert run_style_history(cfg, Console()) == 0


def test_style_update_help() -> None:
    result = runner.invoke(app, ["style", "update", "--help"])
    assert result.exit_code == 0
    assert "fingerprint" in result.stdout.lower()


def test_style_history_help() -> None:
    result = runner.invoke(app, ["style", "history", "--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower()
