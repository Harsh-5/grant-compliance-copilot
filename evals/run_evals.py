"""Evaluation harness.

Three metrics, because "it seemed to work" is not a result:

  recall@1 / recall@3  — did retrieval surface the correct policy clause?
  grounding rate       — did every answered question carry a citation?
  refusal correctness  — did we refuse the out-of-scope questions (and only those)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import Copilot            # noqa: E402
from src.retriever import TfidfRetriever  # noqa: E402

# Questions the corpus genuinely cannot answer — the copilot MUST refuse these.
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I cook risotto?",
]


def main() -> int:
    golden = yaml.safe_load((Path(__file__).parent / "golden_set.yaml").read_text())
    retriever = TfidfRetriever()
    bot = Copilot(use_llm=False)

    hit1 = hit3 = grounded = 0
    misses: list[str] = []

    for row in golden:
        hits = retriever.search(row["question"], k=3)
        ids = [h.chunk.clause_id for h in hits]
        if ids[:1] == [row["expect_clause"]]:
            hit1 += 1
        if row["expect_clause"] in ids:
            hit3 += 1
        else:
            misses.append(f"{row['question']} -> got {ids}, want {row['expect_clause']}")

        ans = bot.ask(row["question"])
        if ans.citations:
            grounded += 1

    refused = sum(1 for q in OUT_OF_SCOPE if bot.ask(q).mode == "refused")

    n = len(golden)
    print(f"Retrieval recall@1 : {hit1}/{n}  ({hit1/n:.0%})")
    print(f"Retrieval recall@3 : {hit3}/{n}  ({hit3/n:.0%})")
    print(f"Grounding rate     : {grounded}/{n}  ({grounded/n:.0%})")
    print(f"Correct refusals   : {refused}/{len(OUT_OF_SCOPE)}")
    if misses:
        print("\nMisses:")
        for m in misses:
            print("  -", m)

    ok = hit3 == n and grounded == n and refused == len(OUT_OF_SCOPE)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
