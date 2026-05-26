"""CLI handler for ``research write <section>``.

Drives the :class:`Scribe` agent: load the fingerprint, fan out N
parallel drafts, render them as a Rich panel-per-draft, and
optionally write the bouquet to a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from research_agent.agents.scribe import Draft, Scribe, normalize_section
from research_agent.config import Config
from research_agent.core.llm import LLMClient
from research_agent.style.fingerprint import Fingerprint


@dataclass(slots=True)
class WriteResult:
    section: str
    drafts: list[Draft]


def run_write(
    cfg: Config,
    console: Console,
    *,
    section: str,
    context: str = "",
    target_words: int = 300,
    versions: int = 3,
    output: Path | None = None,
    parallel: bool = True,
    scribe: Scribe | None = None,
) -> WriteResult:
    """Generate ``versions`` drafts of ``section`` and render them.

    ``scribe`` is injectable so tests can hand in a Scribe wired to a
    :class:`MockLLMProvider` instead of going through the real LLM.
    """
    canonical = normalize_section(section)

    fingerprint: Fingerprint | None = None
    if cfg.fingerprint_path.exists():
        try:
            fingerprint = Fingerprint.load_from(cfg.fingerprint_path)
        except Exception as exc:
            console.print(
                f"[yellow]Could not load fingerprint:[/yellow] {exc} "
                "— falling back to generic academic style."
            )
            fingerprint = None
    else:
        console.print(
            "[yellow]No style fingerprint at[/yellow] "
            f"{cfg.fingerprint_path}.\n"
            "Run [bold]research style train[/bold] + "
            "[bold]research style fingerprint[/bold] first for a voice match."
        )

    if scribe is None:
        llm = LLMClient.from_config(cfg)
        scribe = Scribe(llm, language=cfg.language)

    drafts = scribe.generate(
        canonical,
        fingerprint=fingerprint,
        context=context,
        target_words=target_words,
        n=versions,
        parallel=parallel,
    )

    _render_drafts(console, canonical, drafts)
    if output is not None:
        _persist_drafts(output, canonical, drafts)
        console.print(f"[green]✓[/green] Wrote drafts to [bold]{output}[/bold]")

    return WriteResult(section=canonical, drafts=drafts)


def _render_drafts(console: Console, section: str, drafts: list[Draft]) -> None:
    if not drafts:
        console.print(f"[yellow]Scribe produced no drafts for {section}.[/yellow]")
        return
    for d in drafts:
        header = (
            f"[bold]Version {d.version}[/bold] · {d.variant_label} · "
            f"~{d.word_count} words (target {d.target_words})"
        )
        body = d.text or "[dim](empty)[/dim]"
        if d.style_note:
            body = f"[italic dim]{d.style_note}[/italic dim]\n\n{body}"
        console.print(Panel(body, title=header, border_style="cyan"))


def _persist_drafts(path: Path, section: str, drafts: list[Draft]) -> None:
    """Write drafts to a Markdown file, one section per draft."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# Scribe drafts — {section}\n"]
    for d in drafts:
        lines.append(f"## Version {d.version} — {d.variant_label}\n")
        if d.style_note:
            lines.append(f"_{d.style_note}_\n")
        lines.append(f"*~{d.word_count} words (target {d.target_words})*\n")
        lines.append("")
        lines.append(d.text or "")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
