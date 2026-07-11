import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import Copilot
from src.ingest import load_chunks
from src.rules import load_report, run_checks


def test_every_clause_is_parsed():
    chunks = load_chunks()
    ids = {c.clause_id for c in chunks}
    assert {"REP-101", "REP-103", "PLW-203", "ELG-301"} <= ids
    assert len(chunks) >= 10


def test_pii_is_flagged_critical():
    findings = run_checks(load_report())
    pii = [f for f in findings if f.rule == "REP-103"]
    assert pii and all(f.severity == "critical" for f in pii)
    assert pii[0].participant_id == "P-1008"


def test_wage_below_minimum_is_flagged():
    findings = run_checks(load_report())
    wage = {f.participant_id for f in findings if f.rule == "PLW-203"}
    assert {"P-1005", "P-1009"} <= wage      # $6.50 and $7.00 are below $7.25


def test_placement_before_enrollment_is_flagged():
    findings = run_checks(load_report())
    assert any(f.rule == "PLW-202" and f.participant_id == "P-1003" for f in findings)


def test_clean_record_is_not_flagged():
    findings = run_checks(load_report())
    assert not [f for f in findings if f.participant_id == "P-1001"]


def test_out_of_scope_question_is_refused():
    assert Copilot().ask("What is the capital of France?").mode == "refused"


def test_policy_answer_is_cited():
    ans = Copilot().ask("What is the minimum hourly wage for a placement?")
    assert ans.mode == "policy"
    assert "PLW-203" in ans.citations
