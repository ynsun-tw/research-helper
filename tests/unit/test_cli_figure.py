"""Unit tests for the ``research figure`` CLI surface (E5.3)."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from research_agent.agents.illustrator import FigureDraft, Illustrator
from research_agent.cli import app
from research_agent.cli_figure import (
    VerifyOutcome,
    _persist_drafts,
    _render_drafts,
    _verify_single,
    run_figure,
)
from research_agent.config import Config
from research_agent.core.llm import MockLLMProvider

# --- helpers ----------------------------------------------------------------


def _make_cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg._data_dir_override = tmp_path  # type: ignore[attr-defined]
    return cfg


def _arch_response(label: str) -> str:
    return json.dumps(
        {
            "code": f"\\begin{{tikzpicture}} % {label} \\end{{tikzpicture}}",
            "style_label": label,
            "notes": "deterministic",
            "suggested_use": "overview",
        }
    )


def _result_response(label: str, code: str) -> str:
    return json.dumps(
        {
            "code": code,
            "style_label": label,
            "notes": "deterministic",
            "suggested_use": "results",
        }
    )


# --- run_figure -------------------------------------------------------------


def test_run_figure_returns_drafts_matching_versions(tmp_path: Path) -> None:
    llm = MockLLMProvider(
        responses=[_arch_response("v1"), _arch_response("v2")]
    )
    illustrator = Illustrator(llm, figure_type="architecture")
    console = Console(record=True, width=120)
    result = run_figure(
        _make_cfg(tmp_path),
        console,
        figure_type="architecture",
        description="three layer encoder",
        versions=2,
        illustrator=illustrator,
        parallel=False,
    )
    assert result.figure_type == "architecture"
    assert len(result.drafts) == 2
    assert result.verifications == []


def test_run_figure_writes_output_file(tmp_path: Path) -> None:
    llm = MockLLMProvider(responses=[_arch_response("layered")])
    illustrator = Illustrator(llm, figure_type="architecture")
    console = Console(record=True, width=120)
    out = tmp_path / "figures" / "out.md"
    result = run_figure(
        _make_cfg(tmp_path),
        console,
        figure_type="architecture",
        description="encoder",
        versions=1,
        output=out,
        illustrator=illustrator,
        parallel=False,
    )
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "# Illustrator figures — architecture" in body
    assert "Version A" in body
    assert "```latex" in body
    assert result.drafts[0].code in body


def test_run_figure_normalizes_aliases(tmp_path: Path) -> None:
    llm = MockLLMProvider(responses=[_arch_response("v")])
    illustrator = Illustrator(llm, figure_type="architecture")
    console = Console(record=True, width=120)
    result = run_figure(
        _make_cfg(tmp_path),
        console,
        figure_type="Pipeline",
        description="encoder",
        versions=1,
        illustrator=illustrator,
        parallel=False,
    )
    assert result.figure_type == "architecture"


def test_run_figure_verify_skipped_for_non_result(tmp_path: Path) -> None:
    llm = MockLLMProvider(responses=[_arch_response("v")])
    illustrator = Illustrator(llm, figure_type="architecture")
    console = Console(record=True, width=120)
    result = run_figure(
        _make_cfg(tmp_path),
        console,
        figure_type="architecture",
        description="encoder",
        versions=1,
        verify=True,
        illustrator=illustrator,
        parallel=False,
    )
    assert result.verifications == []
    rendered = console.export_text()
    assert "--verify only applies to --type result" in rendered


# --- _verify_single ---------------------------------------------------------


def test_verify_single_passes_on_clean_code() -> None:
    draft = FigureDraft(
        figure_type="result",
        version="A",
        style_label="x",
        code="x = 1 + 1\n",
        code_language="python",
        notes="",
        suggested_use="",
    )
    outcome = _verify_single(draft)
    assert outcome.ok is True
    assert outcome.timed_out is False
    assert outcome.version == "A"


def test_verify_single_fails_on_bad_code() -> None:
    draft = FigureDraft(
        figure_type="result",
        version="B",
        style_label="x",
        code="raise RuntimeError('boom')\n",
        code_language="python",
        notes="",
        suggested_use="",
    )
    outcome = _verify_single(draft)
    assert outcome.ok is False
    assert "boom" in outcome.stderr
    assert outcome.timed_out is False


def test_verify_single_captures_syntax_error() -> None:
    draft = FigureDraft(
        figure_type="result",
        version="C",
        style_label="x",
        code="def broken(:\n",
        code_language="python",
        notes="",
        suggested_use="",
    )
    outcome = _verify_single(draft)
    assert outcome.ok is False
    # Python's SyntaxError tail line varies but always mentions Syntax.
    assert "Syntax" in outcome.stderr or "syntax" in outcome.stderr.lower()


# --- _persist_drafts --------------------------------------------------------


def test_persist_drafts_round_trip_with_verifications(tmp_path: Path) -> None:
    drafts = [
        FigureDraft(
            figure_type="result",
            version="A",
            style_label="grouped bar",
            code="import matplotlib.pyplot as plt\nplt.bar([1,2],[3,4])\n",
            code_language="python",
            notes="bar chart",
            suggested_use="results section",
        ),
        FigureDraft(
            figure_type="result",
            version="B",
            style_label="line",
            code="raise RuntimeError('x')",
            code_language="python",
            notes="line chart",
            suggested_use="trend",
        ),
    ]
    verifications = [
        VerifyOutcome(version="A", ok=True, stderr=""),
        VerifyOutcome(
            version="B",
            ok=False,
            stderr="Traceback ...\nRuntimeError: x",
            timed_out=False,
        ),
    ]
    out = tmp_path / "figs.md"
    _persist_drafts(out, "result", "compare methods", drafts, verifications)
    text = out.read_text(encoding="utf-8")
    assert "Version A" in text
    assert "Version B" in text
    assert "ran successfully" in text
    assert "runtime error" in text
    assert "RuntimeError" in text


def test_persist_drafts_handles_concept_with_negative_prompt(tmp_path: Path) -> None:
    drafts = [
        FigureDraft(
            figure_type="concept",
            version="A",
            style_label="SD weighted",
            code="clean schematic of attention, (line art:1.2)",
            code_language="text",
            notes="for SD",
            suggested_use="overview",
            target_model="sd",
            negative_prompt="photorealistic, text",
        ),
    ]
    out = tmp_path / "concept.md"
    _persist_drafts(out, "concept", "attention", drafts, [])
    text = out.read_text(encoding="utf-8")
    assert "target model" in text
    assert "negative prompt" in text
    assert "photorealistic" in text


# --- _render_drafts (UI smoke) ----------------------------------------------


def test_render_drafts_prints_each_version() -> None:
    drafts = [
        FigureDraft(
            figure_type="architecture",
            version="A",
            style_label="layered",
            code="\\begin{tikzpicture}\\end{tikzpicture}",
            code_language="tikz",
            notes="note A",
            suggested_use="overview",
        ),
        FigureDraft(
            figure_type="architecture",
            version="B",
            style_label="hub",
            code="\\begin{tikzpicture}\\end{tikzpicture}",
            code_language="tikz",
            notes="note B",
            suggested_use="overview",
        ),
    ]
    console = Console(record=True, width=120)
    _render_drafts(console, "architecture", drafts, verifications=[])
    out = console.export_text()
    assert "Version A" in out
    assert "Version B" in out
    assert "layered" in out
    assert "hub" in out


def test_render_drafts_handles_empty() -> None:
    console = Console(record=True, width=120)
    _render_drafts(console, "architecture", [], verifications=[])
    out = console.export_text()
    assert "produced no drafts" in out


# --- CLI: end-to-end via the typer app --------------------------------------


def test_cli_figure_requires_description(tmp_path: Path, monkeypatch) -> None:
    # _ensure_api_key reads config; stub it.
    cfg = _make_cfg(tmp_path)
    cfg._api_key_override = "fake-key"  # type: ignore[attr-defined]

    def _fake_ensure() -> Config:
        return cfg

    monkeypatch.setattr("research_agent.cli._ensure_api_key", _fake_ensure)
    runner = CliRunner()
    res = runner.invoke(app, ["figure", "--type", "architecture", "--desc", ""])
    assert res.exit_code == 1
    assert "--desc is required" in res.output


def test_cli_figure_rejects_negative_versions(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path)
    cfg._api_key_override = "fake-key"  # type: ignore[attr-defined]
    monkeypatch.setattr("research_agent.cli._ensure_api_key", lambda: cfg)
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["figure", "--type", "architecture", "--desc", "x", "--versions", "0"],
    )
    assert res.exit_code == 1
    assert "positive" in res.output


def test_cli_figure_dispatch_calls_run_figure(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: typer parses args, our run_figure stub fires."""
    cfg = _make_cfg(tmp_path)
    cfg._api_key_override = "fake-key"  # type: ignore[attr-defined]
    monkeypatch.setattr("research_agent.cli._ensure_api_key", lambda: cfg)

    captured: dict = {}

    def _fake_run_figure(*args, **kwargs):
        captured.update(kwargs)
        # Minimal duck-type return to satisfy the caller.
        return None

    # Lazy import: research_agent.cli imports run_figure inside the
    # command body, so we must patch the source module.
    monkeypatch.setattr("research_agent.cli_figure.run_figure", _fake_run_figure)
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "figure",
            "--type", "result",
            "--desc", "accuracy vs baseline",
            "--data", "ours 85, baseline 80",
            "--versions", "2",
            "--verify",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["figure_type"] == "result"
    assert captured["description"] == "accuracy vs baseline"
    assert captured["data"] == "ours 85, baseline 80"
    assert captured["versions"] == 2
    assert captured["verify"] is True
