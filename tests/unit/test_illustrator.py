"""Unit tests for the Illustrator agent (E5.3 — figure generation)."""

from __future__ import annotations

import json

import pytest

from research_agent.agents.illustrator import (
    FIGURE_TYPES,
    FigureDraft,
    Illustrator,
    _build_user_prompt,
    _parse_figure,
    normalize_figure_type,
)
from research_agent.core.llm import MockLLMProvider

# --- normalize_figure_type --------------------------------------------------


def test_normalize_canonical() -> None:
    for t in FIGURE_TYPES:
        assert normalize_figure_type(t) == t


def test_normalize_aliases() -> None:
    assert normalize_figure_type("Pipeline") == "architecture"
    assert normalize_figure_type("ARCH") == "architecture"
    assert normalize_figure_type("plot") == "result"
    assert normalize_figure_type("results") == "result"
    assert normalize_figure_type("schematic") == "concept"
    assert normalize_figure_type("Illustration") == "concept"


def test_normalize_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown figure type"):
        normalize_figure_type("flowchart")


def test_normalize_empty() -> None:
    with pytest.raises(ValueError):
        normalize_figure_type("")


# --- _parse_figure ----------------------------------------------------------


def test_parse_figure_well_formed_architecture() -> None:
    raw = json.dumps(
        {
            "code": "\\begin{tikzpicture}\\end{tikzpicture}",
            "style_label": "layered horizontal",
            "notes": "stacks layers vertically",
            "suggested_use": "system overview",
        }
    )
    parsed = _parse_figure(raw, figure_type="architecture")
    assert parsed["code"].startswith("\\begin{tikzpicture}")
    assert parsed["style_label"] == "layered horizontal"
    assert parsed["notes"] == "stacks layers vertically"


def test_parse_figure_well_formed_concept_keeps_target_model() -> None:
    raw = json.dumps(
        {
            "code": "minimal flat schematic of attention",
            "style_label": "flat schematic for DALL·E 3",
            "notes": "uses natural language",
            "suggested_use": "method overview",
            "target_model": "dalle3",
            "negative_prompt": "",
        }
    )
    parsed = _parse_figure(raw, figure_type="concept")
    assert parsed["target_model"] == "dalle3"
    assert parsed["negative_prompt"] == ""


def test_parse_figure_falls_back_when_not_json() -> None:
    raw = "this is not JSON at all"
    parsed = _parse_figure(raw, figure_type="architecture")
    # We treat the whole reply as the code when JSON parsing fails.
    assert parsed["code"] == raw
    assert parsed["style_label"] == ""


def test_parse_figure_missing_optional_fields_backfilled() -> None:
    raw = json.dumps({"code": "x = 1"})
    parsed = _parse_figure(raw, figure_type="result")
    assert parsed["code"] == "x = 1"
    # Required fields backfilled to empty strings.
    assert parsed["style_label"] == ""
    assert parsed["notes"] == ""
    assert parsed["suggested_use"] == ""


def test_parse_figure_strips_whitespace_in_fields() -> None:
    raw = json.dumps(
        {
            "code": "  \\begin{tikzpicture}  ",
            "style_label": "  hub  ",
            "notes": "\t pads \n",
            "suggested_use": "  intro fig  ",
        }
    )
    parsed = _parse_figure(raw, figure_type="architecture")
    assert parsed["code"] == "\\begin{tikzpicture}"
    assert parsed["style_label"] == "hub"
    assert parsed["notes"] == "pads"


def test_parse_figure_strips_markdown_fences() -> None:
    raw = (
        "Here is the figure:\n"
        "```json\n"
        + json.dumps({"code": "\\begin{tikzpicture}\\end{tikzpicture}",
                       "style_label": "layered", "notes": "n",
                       "suggested_use": "u"})
        + "\n```\n"
    )
    parsed = _parse_figure(raw, figure_type="architecture")
    assert "tikzpicture" in parsed["code"]


# --- _build_user_prompt -----------------------------------------------------


def test_build_user_prompt_includes_variant_directive() -> None:
    out = _build_user_prompt(
        figure_type="architecture",
        description="three-layer encoder",
        data="",
        variant_label="layered horizontal",
        variant_directive="use left-to-right flow",
    )
    assert "layered horizontal" in out
    assert "use left-to-right flow" in out
    assert "three-layer encoder" in out
    assert "Figure type: architecture" in out


def test_build_user_prompt_data_included_when_nonempty() -> None:
    out = _build_user_prompt(
        figure_type="result",
        description="accuracy comparison",
        data="ours 85, baseline 80",
        variant_label="grouped bar",
        variant_directive="render bars side by side",
    )
    assert "ours 85, baseline 80" in out


def test_build_user_prompt_data_omitted_when_empty() -> None:
    out = _build_user_prompt(
        figure_type="architecture",
        description="just a sketch",
        data="",
        variant_label="layered horizontal",
        variant_directive="layered flow",
    )
    assert "Quantitative data" not in out


# --- Illustrator.generate ---------------------------------------------------


def _ok_arch_response(label: str = "layered") -> str:
    return json.dumps(
        {
            "code": f"\\begin{{tikzpicture}} % {label} \\end{{tikzpicture}}",
            "style_label": label,
            "notes": "deterministic for tests",
            "suggested_use": "overview",
        }
    )


def test_generate_returns_n_drafts_with_distinct_versions() -> None:
    llm = MockLLMProvider(
        responses=[_ok_arch_response("v1"), _ok_arch_response("v2"), _ok_arch_response("v3")]
    )
    agent = Illustrator(llm, figure_type="architecture")
    drafts = agent.generate("encoder", n=3, parallel=False)
    assert len(drafts) == 3
    assert [d.version for d in drafts] == ["A", "B", "C"]
    assert all(d.figure_type == "architecture" for d in drafts)
    assert all(d.code_language == "tikz" for d in drafts)


def test_generate_n_zero_returns_empty_list() -> None:
    llm = MockLLMProvider(responses=[])
    agent = Illustrator(llm, figure_type="architecture")
    assert agent.generate("anything", n=0) == []


def test_generate_cycles_variants_when_n_exceeds_table() -> None:
    # Architecture has 3 variants — at n=5 we wrap around.
    llm = MockLLMProvider(responses=[_ok_arch_response(f"r{i}") for i in range(5)])
    agent = Illustrator(llm, figure_type="architecture")
    drafts = agent.generate("encoder", n=5, parallel=False)
    assert len(drafts) == 5
    assert [d.version for d in drafts] == ["A", "B", "C", "D", "E"]


def test_generate_result_uses_python_code_language() -> None:
    raw = json.dumps(
        {
            "code": "import matplotlib.pyplot as plt\nplt.figure()\n",
            "style_label": "grouped bar",
            "notes": "n",
            "suggested_use": "u",
        }
    )
    llm = MockLLMProvider(responses=[raw])
    agent = Illustrator(llm, figure_type="result")
    drafts = agent.generate("accuracy", n=1, parallel=False)
    assert drafts[0].code_language == "python"
    assert "matplotlib" in drafts[0].code


def test_generate_concept_carries_target_model() -> None:
    raw = json.dumps(
        {
            "code": "clean schematic illustration",
            "style_label": "flat schematic",
            "notes": "n",
            "suggested_use": "u",
            "target_model": "dalle3",
            "negative_prompt": "",
        }
    )
    llm = MockLLMProvider(responses=[raw])
    agent = Illustrator(llm, figure_type="concept")
    drafts = agent.generate("attention flow", n=1, parallel=False)
    assert drafts[0].target_model == "dalle3"
    assert drafts[0].code_language == "text"


def test_generate_falls_back_when_llm_returns_garbage() -> None:
    llm = MockLLMProvider(responses=["totally not JSON at all"])
    agent = Illustrator(llm, figure_type="architecture")
    drafts = agent.generate("encoder", n=1, parallel=False)
    # Even on garbage we should get one Draft with the raw text as code.
    assert len(drafts) == 1
    assert drafts[0].code == "totally not JSON at all"
    assert drafts[0].style_label  # backfilled from variant table


def test_generate_parallel_emits_same_count_as_sequential() -> None:
    seq_llm = MockLLMProvider(responses=[_ok_arch_response()] * 3)
    par_llm = MockLLMProvider(responses=[_ok_arch_response()] * 3)
    seq_agent = Illustrator(seq_llm, figure_type="architecture")
    par_agent = Illustrator(par_llm, figure_type="architecture")
    seq = seq_agent.generate("x", n=3, parallel=False)
    par = par_agent.generate("x", n=3, parallel=True)
    assert len(seq) == len(par) == 3


# --- Illustrator.run (orchestrator adapter) ---------------------------------


def test_run_returns_first_draft_via_agentresponse() -> None:
    llm = MockLLMProvider(responses=[_ok_arch_response("only")])
    agent = Illustrator(llm, figure_type="architecture")
    resp = agent.run({"description": "encoder", "n": 1, "parallel": False})
    assert resp.role == "illustrator"
    assert "tikzpicture" in resp.content
    drafts = resp.metadata["drafts"]
    assert isinstance(drafts, list) and len(drafts) == 1
    assert isinstance(drafts[0], FigureDraft)


def test_run_rejects_empty_description() -> None:
    llm = MockLLMProvider(responses=[])
    agent = Illustrator(llm, figure_type="architecture")
    with pytest.raises(ValueError, match="description"):
        agent.run({"description": ""})
