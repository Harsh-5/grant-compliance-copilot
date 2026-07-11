"""The copilot: routes a question to the right tool, then answers with citations.

Routing is intentionally boring — a cheap classifier decides between two tools:

    policy question  -> retrieve clauses           (RAG)
    data question    -> run the rules engine       (deterministic code)
    both             -> audit mode                 (findings explained by policy)

An LLM *could* do this routing. It doesn't need to: the decision is a keyword check, and
spending a model call on it would add cost and latency for no accuracy gain. Use the model
where it adds value (synthesis), not where a rule already wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import llm
from .guardrails import REFUSAL, enforce_citations, is_grounded
from .retriever import Hit, TfidfRetriever
from .rules import Finding, load_report, run_checks, summarize

DATA_WORDS = {
    "participant", "participants", "record", "records", "flag", "flagged", "finding",
    "findings", "audit", "violation", "violations", "issue", "issues", "our data",
    "report", "who", "how many", "which",
}


@dataclass
class Answer:
    question: str
    mode: str                                  # "policy" | "data" | "audit" | "refused"
    text: str
    citations: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)


class Copilot:
    def __init__(self, use_llm: bool = False):
        self.retriever = TfidfRetriever()
        self.use_llm = use_llm

    # -- routing -----------------------------------------------------------
    @staticmethod
    def _route(q: str) -> str:
        ql = q.lower()
        data = any(w in ql for w in DATA_WORDS)
        return "data" if data else "policy"

    # -- tools -------------------------------------------------------------
    def _policy_answer(self, q: str, hits: list[Hit]) -> str:
        context = "\n\n".join(f"[{h.chunk.clause_id}] {h.chunk.text}" for h in hits)
        if self.use_llm:
            out = llm.synthesize(q, context)
            if out:
                return out
        # Extractive fallback — zero cost, fully grounded, always available.
        lines = [f"Based on {len(hits)} policy clause(s):", ""]
        for h in hits:
            lines.append(f"• [{h.chunk.clause_id}] {h.chunk.title} — {h.chunk.text.split('. ', 1)[-1]}")
        return "\n".join(lines)

    def _audit(self) -> tuple[list[Finding], dict]:
        df = load_report()
        findings = run_checks(df)
        return findings, summarize(findings)

    # -- entrypoint --------------------------------------------------------
    def ask(self, question: str) -> Answer:
        route = self._route(question)

        if route == "data":
            findings, summary = self._audit()
            hits = self.retriever.search(question)
            head = (
                f"Audit of {summary['participants_flagged']} flagged participant(s) — "
                f"{summary['total_findings']} finding(s): "
                + ", ".join(f"{k}={v}" for k, v in summary["by_severity"].items())
            )
            body = "\n".join(
                f"• {f.participant_id} — [{f.rule}] ({f.severity}) {f.detail}" for f in findings
            )
            text = f"{head}\n\n{body}" if findings else "No compliance findings. All records pass."
            return Answer(
                question=question, mode="audit", text=text,
                citations=sorted({f.rule for f in findings}),
                findings=[f.to_dict() for f in findings],
            )

        hits = self.retriever.search(question)
        if not is_grounded(hits):                       # guardrail: refuse, don't guess
            return Answer(question=question, mode="refused", text=REFUSAL)

        text = enforce_citations(self._policy_answer(question, hits), hits)
        return Answer(
            question=question, mode="policy", text=text,
            citations=[h.chunk.clause_id for h in hits],
        )
