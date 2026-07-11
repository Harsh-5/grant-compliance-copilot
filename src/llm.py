"""Optional LLM layer.

The copilot is fully functional without any API key: in extractive mode it returns the
retrieved policy clauses and the rules-engine findings verbatim. The LLM is an *optional
synthesis layer* on top — it phrases the answer, it does not decide the answer.

That boundary is the point. Facts come from retrieval and deterministic code; the model
only makes them readable. Swapping providers means editing this one file.
"""
from __future__ import annotations

import os

SYSTEM = (
    "You are a grant compliance assistant. Answer ONLY from the policy excerpts provided. "
    "Cite the clause ID (e.g. PLW-203) for every claim. If the excerpts do not contain the "
    "answer, say so plainly. Never invent a policy or a number."
)


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def synthesize(question: str, context: str) -> str | None:
    """Return a natural-language answer, or None if no LLM is configured/reachable."""
    if not available():
        return None
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            max_tokens=600,
            system=SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Policy excerpts:\n{context}\n\nQuestion: {question}",
            }],
        )
        return resp.content[0].text.strip()
    except Exception as e:  # never let the optional layer break the core product
        return f"[LLM unavailable: {e}. Falling back to extractive answer.]"
