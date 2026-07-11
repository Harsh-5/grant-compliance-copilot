# Grant Compliance Copilot

An agentic RAG system that answers grant-compliance questions from policy documents **with
citations**, and audits participant reporting data against those policies with a
**deterministic rules engine**.

Built from a real problem: across 16 grant programs and 40+ partner organizations, the
expensive failure isn't a slow report — it's a report that's *confidently wrong* and gets
caught in an audit. So this system is designed around one constraint: **every answer must be
traceable to a policy clause.**

```bash
pip install -r requirements.txt
python app.py audit
python app.py ask "What is the minimum hourly wage a placement must meet?"
```

Runs with **zero API keys**. The LLM layer is optional.

---

## What it does

**Policy questions → RAG with enforced citations**

```
$ python app.py ask "What is the minimum hourly wage a placement must meet?"
mode: policy

• [PLW-203] Minimum wage standard — Reported placements must meet or exceed the applicable
  federal minimum wage of $7.25 per hour...

Citations: PLW-203, PLW-204, PLW-202
```

**Data questions → deterministic audit, every finding tied to a clause**

```
$ python app.py audit
Audit of 9 flagged participant(s) — 9 findings: critical=1, high=6, medium=2

• P-1008 — [REP-103] (critical) Record contains a Social Security Number. PII must be removed.
• P-1005 — [PLW-203] (high)     Reported wage $6.50/hr is below the $7.25 federal minimum.
• P-1003 — [PLW-202] (high)     Placement date 2025-01-10 precedes enrollment date 2025-02-01.
• P-1013 — [PLW-201] (high)     Reported as employed but only 25 days employed (min 30).
```

**Out-of-scope questions → refused, not guessed**

```
$ python app.py ask "What is the capital of France?"
mode: refused
I can't answer that from the grant policy documents I have.
```

---

## Design decisions (the interesting part)

**1. The rules engine is code, not prompts.**
"Is $6.50 below $7.25?" is arithmetic. Routing that through an LLM costs money, adds latency,
and can be wrong — unacceptable when a funder will audit the output. The rules engine *finds*
violations deterministically and reproducibly; the LLM's only job is to *explain* them. Facts
come from code and retrieval. The model makes them readable.

**2. Retrieval is TF-IDF by default, not embeddings.**
For a few hundred short, keyword-dense regulatory clauses, lexical search is free, offline,
instant, and fully explainable — and it scores **100% recall@1** on the golden set (below).
Embeddings are a genuine upgrade for paraphrase-heavy queries, so `retriever.py` is written
behind an interface you can swap without touching the agent. The point: reach for the
expensive tool when the evals say you need it, not because it's fashionable.

**3. Chunking maps 1:1 to policy clauses.**
Chunks are split on clause headings, not arbitrary token windows, so every citation returned
(`PLW-203`) is an ID a compliance officer can actually look up. A citation you can't verify
isn't a citation.

**4. Guardrails are the product, not a feature.**
No relevant clause retrieved → refuse. Policy claim without a clause ID → blocked. In an audit
context, a confidently wrong answer is worse than no answer.

---

## Evaluation

Because "it seemed to work" is not a result:

```
$ python evals/run_evals.py
Retrieval recall@1 : 10/10  (100%)
Retrieval recall@3 : 10/10  (100%)
Grounding rate     : 10/10  (100%)
Correct refusals   : 2/2
RESULT: PASS
```

`evals/golden_set.yaml` pins each question to the clause that *must* be retrieved to answer it.
This is what justifies the TF-IDF decision with evidence rather than vibes — and it's the
regression net for swapping in embeddings later.

```
$ python -m pytest tests/ -q
7 passed
```

---

## Architecture

```
question
   │
   ├─ route (keyword check — cheap, no model call)
   │
   ├── policy question ──► TF-IDF retriever ──► guardrail: grounded?
   │                                              ├─ no  ──► refuse
   │                                              └─ yes ──► answer + enforced citations
   │                                                            └─ optional LLM synthesis
   │
   └── data question ────► pandas rules engine ──► findings, each tagged with its clause
```

```
src/
├── ingest.py      clause-aware chunking of policy docs
├── retriever.py   TF-IDF retrieval (swappable for embeddings)
├── rules.py       deterministic compliance checks (REP / PLW / ELG)
├── guardrails.py  grounding, scope, citation enforcement
├── llm.py         optional synthesis layer (provider-agnostic)
└── agent.py       routing + orchestration
```

## Rules implemented

| Clause | Check | Severity |
|---|---|---|
| REP-103 | No PII (SSN) in submitted records | critical |
| REP-102 | Required fields present | high |
| PLW-202 | Placement date not before enrollment | high |
| PLW-203 | Wage ≥ federal minimum ($7.25) | high |
| PLW-201 | ≥30 consecutive days employed to count a placement | high |
| ELG-302 | 90-day follow-up documented | medium |
| ELG-303 | Exit status is a permitted value | medium |

## Optional LLM layer

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY
python app.py ask "Are we allowed to include SSNs?" --llm
```

The system degrades gracefully: no key, bad key, or network failure all fall back to the
extractive answer. The optional layer can never break the core product.

## Roadmap

- Swap TF-IDF → embeddings and let the eval harness decide if it's worth the cost
- Multi-turn memory for follow-up questions ("what about the wage rule?")
- Export findings to a funder-ready corrective-action report
- Ingest real policy PDFs (currently markdown)

---

**Note on data:** all participant data in `data/` is synthetic, generated for this project,
with violations deliberately seeded to exercise each rule. No real participant data is used.

Built by **Harsharan Gorli** — [Portfolio](https://harsh-5.github.io/harsharan-portfolio/) ·
[LinkedIn](https://www.linkedin.com/in/harsharan-preet/)
