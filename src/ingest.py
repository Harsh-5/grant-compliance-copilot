"""Load policy documents and split them into citable chunks.

Chunking strategy: split on markdown headings, so every chunk maps 1:1 to a real
policy clause (e.g. "PLW-203"). That means every citation we return is a clause ID a
compliance officer can actually look up — not an arbitrary 500-token window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import POLICY_DIR

CLAUSE_RE = re.compile(r"^##\s+([A-Z]{3}-\d{3})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    clause_id: str      # e.g. "PLW-203"
    title: str          # e.g. "Minimum wage standard"
    text: str
    source: str         # filename

    @property
    def citation(self) -> str:
        return f"{self.clause_id} ({self.source})"


def load_chunks(policy_dir: Path | None = None) -> list[Chunk]:
    policy_dir = policy_dir or POLICY_DIR
    chunks: list[Chunk] = []

    for path in sorted(policy_dir.glob("*.md")):
        clause_id = title = None
        buf: list[str] = []

        def flush() -> None:
            if clause_id and buf:
                body = " ".join(l.strip() for l in buf if l.strip())
                chunks.append(
                    Chunk(clause_id=clause_id, title=title, text=f"{title}. {body}", source=path.name)
                )

        for line in path.read_text().splitlines():
            m = CLAUSE_RE.match(line)
            if m:
                flush()
                buf = []
                clause_id, title = m.group(1), m.group(2).strip()
            elif clause_id:
                buf.append(line)
        flush()

    if not chunks:
        raise RuntimeError(f"No policy clauses found in {policy_dir}")
    return chunks
