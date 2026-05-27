"""CLI handler for ``research write figure --type ...`` (E5.3).

Drives the :class:`Illustrator` agent: build N parallel variant
drafts of a TikZ diagram (S5.3.1), matplotlib script (S5.3.2), or
text-to-image prompt (S5.3.3); render each draft as a Rich syntax
panel; optionally write the bouquet to a file; optionally
``--verify`` runnable Python by exec-ing it in a subprocess.

The subprocess verification is intentionally pragmatic: a fresh
``python -c`` with a 30 s timeout, the user's current interpreter,
and matplotlib's ``Agg`` backend forced via ``MPLBACKEND``. We do
not spin up a venv per draft — that's the deferred S5.1.2
isolation story. ``--verify`` here is "did this run at all", not
"does it run in a clean environment".
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from research_agent.agents.illustrator import (
    FigureDraft,
    Illustrator,
    normalize_figure_type,
)
from research_agent.config import Config
from research_agent.core.llm import LLMClient

VERIFY_TIMEOUT_SECONDS = 30


@dataclass
class VerifyOutcome:
    """Result of running a single matplotlib draft via subprocess."""

    version: str
    ok: bool
    stderr: str
    timed_out: bool = False


@dataclass
class FigureResult:
    figure_type: str
    drafts: list[FigureDraft]
    verifications: list[VerifyOutcome]


def run_figure(
    cfg: Config,
    console: Console,
    *,
    figure_type: str,
    description: str,
    data: str = "",
    versions: int = 2,
    output: Path | None = None,
    verify: bool = False,
    parallel: bool = True,
    illustrator: Illustrator | None = None,
) -> FigureResult:
    """Generate ``versions`` figure drafts and render them.

    ``illustrator`` is injectable for tests; in production we build
    it from ``cfg`` ourselves.
    """
    canonical_type = normalize_figure_type(figure_type)

    if illustrator is None:
        llm = LLMClient.from_config(cfg)
        illustrator = Illustrator(
            llm, figure_type=canonical_type, language=cfg.language
        )

    drafts = illustrator.generate(
        description,
        data=data,
        n=versions,
        parallel=parallel,
    )

    verifications: list[VerifyOutcome] = []
    if verify:
        if canonical_type != "result":
            console.print(
                "[yellow]--verify only applies to --type result;"
                " skipping verification.[/yellow]"
            )
        else:
            verifications = _verify_drafts(drafts, console=console)

    _render_drafts(console, canonical_type, drafts, verifications)

    if output is not None:
        _persist_drafts(output, canonical_type, description, drafts, verifications)
        console.print(f"[green]✓[/green] Wrote figure drafts to [bold]{output}[/bold]")

    return FigureResult(
        figure_type=canonical_type,
        drafts=drafts,
        verifications=verifications,
    )


def _verify_drafts(
    drafts: list[FigureDraft],
    *,
    console: Console,
) -> list[VerifyOutcome]:
    """Run each draft's Python in a subprocess; return per-draft outcomes."""
    outcomes: list[VerifyOutcome] = []
    for d in drafts:
        outcome = _verify_single(d)
        outcomes.append(outcome)
        if outcome.ok:
            console.print(f"[green]✓[/green] Version {d.version} ran successfully.")
        elif outcome.timed_out:
            console.print(
                f"[red]✗[/red] Version {d.version} timed out after "
                f"{VERIFY_TIMEOUT_SECONDS}s."
            )
        else:
            first_err_line = (outcome.stderr.strip().splitlines() or [""])[-1]
            console.print(
                f"[red]✗[/red] Version {d.version} failed: {first_err_line}"
            )
    return outcomes


def _verify_single(draft: FigureDraft) -> VerifyOutcome:
    """Exec one draft's code in a subprocess with timeout + Agg backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "figure.py"
        script_path.write_text(draft.code, encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=VERIFY_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return VerifyOutcome(
                version=draft.version, ok=False, stderr="", timed_out=True
            )
        return VerifyOutcome(
            version=draft.version,
            ok=result.returncode == 0,
            stderr=result.stderr,
            timed_out=False,
        )


# -------------------------------------------------------------- rendering


_LANGUAGE_FOR_SYNTAX: dict[str, str] = {
    "tikz": "latex",
    "python": "python",
    "text": "text",
}


def _render_drafts(
    console: Console,
    figure_type: str,
    drafts: list[FigureDraft],
    verifications: list[VerifyOutcome],
) -> None:
    if not drafts:
        console.print(
            f"[yellow]Illustrator produced no drafts for type={figure_type}.[/yellow]"
        )
        return
    verify_by_version = {v.version: v for v in verifications}
    for d in drafts:
        header_bits = [
            f"[bold]Version {d.version}[/bold]",
            f"{figure_type}",
            d.style_label or "(no label)",
        ]
        if d.target_model:
            header_bits.append(f"target: {d.target_model}")
        v = verify_by_version.get(d.version)
        if v is not None:
            header_bits.append(
                "[green]✓ runs[/green]" if v.ok else "[red]✗ failed[/red]"
            )
        header = " · ".join(header_bits)

        lang = _LANGUAGE_FOR_SYNTAX.get(d.code_language, "text")
        body = Syntax(d.code or "(empty)", lang, theme="ansi_dark", word_wrap=True)
        console.print(Panel(body, title=header, border_style="cyan"))

        meta_lines: list[str] = []
        if d.notes:
            meta_lines.append(f"[dim]notes:[/dim] {d.notes}")
        if d.suggested_use:
            meta_lines.append(f"[dim]suggested use:[/dim] {d.suggested_use}")
        if d.negative_prompt:
            meta_lines.append(f"[dim]negative:[/dim] {d.negative_prompt}")
        if meta_lines:
            console.print("\n".join(meta_lines))
        console.print()


def _persist_drafts(
    path: Path,
    figure_type: str,
    description: str,
    drafts: list[FigureDraft],
    verifications: list[VerifyOutcome],
) -> None:
    """Persist drafts to a Markdown file with one fenced block per draft."""
    path.parent.mkdir(parents=True, exist_ok=True)
    verify_by_version = {v.version: v for v in verifications}
    lines: list[str] = [
        f"# Illustrator figures — {figure_type}\n",
        f"_Description: {description.strip() or '(none)'}_\n",
    ]
    lang_for_fence = {
        "tikz": "latex",
        "python": "python",
        "text": "",
    }
    for d in drafts:
        lines.append(f"## Version {d.version} — {d.style_label}\n")
        if d.target_model:
            lines.append(f"**target model:** `{d.target_model}`\n")
        if d.notes:
            lines.append(f"_{d.notes}_\n")
        if d.suggested_use:
            lines.append(f"**suggested use:** {d.suggested_use}\n")
        v = verify_by_version.get(d.version)
        if v is not None:
            status = "✓ ran successfully" if v.ok else (
                "✗ timed out" if v.timed_out else "✗ runtime error"
            )
            lines.append(f"**verification:** {status}\n")
            if not v.ok and v.stderr:
                tail = "\n".join(v.stderr.strip().splitlines()[-6:])
                lines.append(f"```\n{tail}\n```\n")
        fence_lang = lang_for_fence.get(d.code_language, "")
        lines.append(f"```{fence_lang}")
        lines.append(d.code.rstrip())
        lines.append("```")
        if d.negative_prompt:
            lines.append("")
            lines.append(f"**negative prompt:** `{d.negative_prompt}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
