"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return tmp_path / "research-agent"


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A small PDF with title, authors, abstract, and two numbered sections."""
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Attention Is Almost All You Need", fontsize=22)
    page.insert_text((72, 110), "Alice Smith, Bob Jones, Carol Wei", fontsize=10)
    page.insert_text((72, 150), "Abstract", fontsize=14)
    page.insert_text(
        (72, 170),
        "We propose a new architecture that replaces recurrence with attention.",
        fontsize=10,
    )
    page.insert_text((72, 210), "1. Introduction", fontsize=14)
    page.insert_text((72, 230), "Recurrent models have dominated sequence modeling.", fontsize=10)
    page.insert_text((72, 270), "2. Method", fontsize=14)
    page.insert_text((72, 290), "The Transformer relies entirely on self-attention.", fontsize=10)
    page.insert_text((72, 330), "References", fontsize=14)
    page.insert_text((72, 350), "[1] Bahdanau et al., 2014.", fontsize=10)
    doc.save(path)
    doc.close()
    return path
