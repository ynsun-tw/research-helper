"""Unit tests for ``research check`` CLI."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from research_agent.cli import app
from research_agent.cli_check import run_check
from research_agent.cli_style import run_style_train
from research_agent.config import Config
from research_agent.core.paper import Paper, Section

runner = CliRunner()


def test_check_help() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    assert "threshold" in result.stdout.lower()


def test_run_check_empty_draft_raises(config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    try:
        run_check(cfg, Console(), draft_text="   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError on empty draft")


def test_run_check_no_corpus_returns_clean(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    draft = (
        "Recurrent neural networks have long dominated sequence modeling "
        "but their sequential nature limits parallelization."
    )
    result = run_check(cfg, Console(), draft_text=draft)
    assert result.report.is_clean
    assert result.report.draft_paragraph_count == 1


def test_run_check_detects_verbatim_reuse(
    monkeypatch, tmp_path: Path, config_dir: Path
) -> None:
    """Integration: train on a paper, then check a draft that reuses one of
    its paragraphs verbatim → flagged."""
    paragraph = (
        "Recurrent neural networks have long dominated sequence modeling "
        "but their sequential nature limits parallelization across the time "
        "dimension. We propose to overcome this with attention-only "
        "architectures that scale better in practice."
    )
    second = (
        "Our experiments span a range of model sizes from 100M to 7B "
        "parameters. We observe consistent improvements across all "
        "benchmarks we tested."
    )

    def fake_load(src: str, *, cache_dir: Path) -> Paper:
        return Paper(
            id="arxiv:0001.0001",
            title="t",
            authors=[],
            abstract="",
            sections=[
                Section(title="Introduction", content=paragraph + "\n\n" + second),
            ],
            full_text="",
            pdf_path=Path("/tmp/x.pdf"),
        )

    monkeypatch.setattr("research_agent.cli_style.load_paper", fake_load)
    cfg = Config.load(config_dir)
    run_style_train(cfg, Console(), sources=["arxiv:0001.0001"])

    # Draft re-uses the first paragraph verbatim plus a fresh second paragraph
    draft = (
        paragraph
        + "\n\nThis second paragraph is brand new prose about completely "
        "different topics like baking bread and roasting chicken."
    )
    result = run_check(cfg, Console(), draft_text=draft, threshold=0.4)
    assert not result.report.is_clean
    assert any(m.similarity > 0.9 for m in result.report.matches)


def test_run_check_writes_report(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    out = tmp_path / "report.md"
    samples = [("arxiv:0001", "Some prose about transformers and attention.")]
    result = run_check(
        cfg,
        Console(),
        draft_text="Some prose about transformers and attention.",
        threshold=0.4,
        output=out,
        samples_override=samples,
    )
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "Self-plagiarism check" in body
    # Verbatim → should be flagged
    assert not result.report.is_clean


def test_run_check_below_threshold_clean(tmp_path: Path, config_dir: Path) -> None:
    cfg = Config.load(config_dir)
    samples = [
        ("arxiv:0001", "Recurrent networks dominate sequence modeling for years."),
    ]
    draft = (
        "Self-attention is the core building block of modern transformer "
        "architectures used in vision and language."
    )
    result = run_check(
        cfg,
        Console(),
        draft_text=draft,
        threshold=0.7,
        samples_override=samples,
    )
    assert result.report.is_clean
