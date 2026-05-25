"""Unit tests for the PDF parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.parsers.pdf import PDFParseError, parse_pdf


def test_parse_sample_pdf_extracts_title(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    assert "Attention" in paper.title


def test_parse_sample_pdf_extracts_sections(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    titles = " | ".join(paper.section_titles())
    assert "Abstract" in titles
    assert "Introduction" in titles
    assert "Method" in titles
    assert "References" in titles


def test_parse_sample_pdf_extracts_abstract(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    assert "attention" in paper.abstract.lower()


def test_parse_sample_pdf_full_text_present(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    assert "Transformer" in paper.full_text


def test_parse_assigns_local_id_when_not_provided(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf)
    assert paper.id.startswith("local:")
    assert paper.pdf_path == sample_pdf


def test_parse_uses_explicit_paper_id(sample_pdf: Path) -> None:
    paper = parse_pdf(sample_pdf, paper_id="arxiv:1706.03762")
    assert paper.id == "arxiv:1706.03762"


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PDFParseError):
        parse_pdf(tmp_path / "missing.pdf")
