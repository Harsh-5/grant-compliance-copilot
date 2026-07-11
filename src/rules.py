"""Deterministic compliance rules engine (pandas).

DESIGN NOTE (deliberate): these checks are code, not prompts.

"Is this wage below $7.25?" is arithmetic. Asking an LLM to do it costs money, adds
latency, and can be wrong — an unacceptable trade in an audit context where the answer
must be reproducible and defensible to a funder. The LLM's job is to *explain* findings;
the rules engine's job is to *find* them. Every finding carries the policy clause it
violates, so the output is directly auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .config import MIN_EMPLOYED_DAYS, MIN_WAGE, REPORT_CSV, VALID_EXIT_STATUSES

REQUIRED_FIELDS = ["participant_id", "enrollment_date", "program_track", "exit_status"]


@dataclass
class Finding:
    participant_id: str
    rule: str          # policy clause id, e.g. "PLW-203"
    severity: str      # "critical" | "high" | "medium"
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_report(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or REPORT_CSV, dtype=str).fillna("")
    for col in ("hourly_wage", "days_employed"):
        df[col + "_num"] = pd.to_numeric(df[col], errors="coerce")
    for col in ("enrollment_date", "placement_date"):
        df[col + "_dt"] = pd.to_datetime(df[col], errors="coerce")
    return df


def run_checks(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    add = findings.append

    for _, r in df.iterrows():
        pid = r["participant_id"] or "<missing id>"

        # REP-103 — PII must never be present. Critical: this is a funder-reportable breach.
        if r.get("ssn", "").strip():
            add(Finding(pid, "REP-103", "critical",
                        "Record contains a Social Security Number. PII must be removed before submission."))

        # REP-102 — record completeness
        missing = [f for f in REQUIRED_FIELDS if not str(r.get(f, "")).strip()]
        if missing:
            add(Finding(pid, "REP-102", "high",
                        f"Incomplete record — missing required field(s): {', '.join(missing)}."))

        # ELG-303 — exit status must be a permitted value
        exit_status = r["exit_status"].strip()
        if exit_status and exit_status not in VALID_EXIT_STATUSES:
            add(Finding(pid, "ELG-303", "medium",
                        f"Exit status '{exit_status}' is not a permitted value."))

        placed = bool(r["placement_date"].strip())
        if placed:
            # PLW-202 — placement date cannot precede enrollment
            e, p = r["enrollment_date_dt"], r["placement_date_dt"]
            if pd.notna(e) and pd.notna(p) and p < e:
                add(Finding(pid, "PLW-202", "high",
                            f"Placement date {p.date()} precedes enrollment date {e.date()}."))

            # PLW-203 — minimum wage
            w = r["hourly_wage_num"]
            if pd.notna(w) and w < MIN_WAGE:
                add(Finding(pid, "PLW-203", "high",
                            f"Reported wage ${w:.2f}/hr is below the ${MIN_WAGE:.2f} federal minimum."))

            # PLW-201 — 30-day employment threshold for a countable placement
            d = r["days_employed_num"]
            if pd.notna(d) and d < MIN_EMPLOYED_DAYS and exit_status == "employed":
                add(Finding(pid, "PLW-201", "high",
                            f"Reported as employed but only {int(d)} days employed "
                            f"(minimum {MIN_EMPLOYED_DAYS} consecutive days)."))

            # ELG-302 — 90-day follow-up documentation
            if r["followup_90day"].strip().lower() != "yes" and exit_status == "employed":
                add(Finding(pid, "ELG-302", "medium",
                            "Placed participant is missing 90-day follow-up documentation."))

    return findings


def summarize(findings: list[Finding]) -> dict:
    by_sev: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    for f in findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        by_rule[f.rule] = by_rule.get(f.rule, 0) + 1
    return {
        "total_findings": len(findings),
        "participants_flagged": len({f.participant_id for f in findings}),
        "by_severity": dict(sorted(by_sev.items())),
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
    }
