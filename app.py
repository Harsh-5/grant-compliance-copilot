"""CLI for the Grant Compliance Copilot.

    python app.py ask "What is the minimum wage for a reported placement?"
    python app.py audit
    python app.py ask "Which participants are flagged?" --llm
"""
from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from src.agent import Copilot
from src.rules import load_report, run_checks, summarize

console = Console()


def cmd_ask(args) -> None:
    bot = Copilot(use_llm=args.llm)
    ans = bot.ask(args.question)
    console.print(f"\n[bold]Q:[/bold] {ans.question}")
    console.print(f"[dim]mode: {ans.mode}[/dim]\n")
    console.print(ans.text)
    if ans.citations:
        console.print(f"\n[dim]Citations: {', '.join(ans.citations)}[/dim]")


def cmd_audit(args) -> None:
    findings = run_checks(load_report())
    summary = summarize(findings)

    table = Table(title="Compliance findings", header_style="bold")
    for col in ("Participant", "Rule", "Severity", "Detail"):
        table.add_column(col, overflow="fold")
    for f in findings:
        colour = {"critical": "red", "high": "yellow", "medium": "cyan"}[f.severity]
        table.add_row(f.participant_id, f.rule, f"[{colour}]{f.severity}[/{colour}]", f.detail)
    console.print(table)
    console.print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Grant Compliance Copilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="Ask a policy or data question")
    a.add_argument("question")
    a.add_argument("--llm", action="store_true", help="Use the optional LLM synthesis layer")
    a.set_defaults(func=cmd_ask)

    d = sub.add_parser("audit", help="Run the full deterministic compliance audit")
    d.set_defaults(func=cmd_audit)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
