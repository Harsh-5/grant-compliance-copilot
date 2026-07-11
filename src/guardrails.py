"""Guardrails.

Three rules, all of which exist because an audit answer that is confidently wrong is worse
than no answer at all:

1. GROUNDING  — never answer a policy question without at least one retrieved clause.
2. SCOPE      — refuse questions the policy corpus cannot support, instead of guessing.
3. CITATION   — every policy claim returned must carry the clause ID it came from.
"""
from __future__ import annotations

from .retriever import Hit

REFUSAL = (
    "I can't answer that from the grant policy documents I have. "
    "I only answer questions grounded in the loaded policies "
    "(reporting, placement & wage, eligibility). "
    "Rephrase, or point me at the policy document that covers it."
)


def is_grounded(hits: list[Hit]) -> bool:
    """No relevant clause retrieved -> we do not answer. This is the anti-hallucination gate."""
    return bool(hits)


def enforce_citations(answer: str, hits: list[Hit]) -> str:
    """A policy answer without a citation is not a usable audit artefact."""
    if not hits:
        return REFUSAL
    if not any(h.chunk.clause_id in answer for h in hits):
        cites = ", ".join(h.chunk.clause_id for h in hits)
        answer = f"{answer}\n\nSources: {cites}"
    return answer
