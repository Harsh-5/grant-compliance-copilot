"""Central config. Paths and tunables live here so nothing is hard-coded downstream."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
REPORT_CSV = ROOT / "data" / "reporting" / "participants.csv"

# Retrieval
TOP_K = 3
MIN_RELEVANCE = 0.05          # below this we refuse rather than guess (guardrail)

# Compliance rule thresholds (sourced from the policy docs — keep in sync)
MIN_WAGE = 7.25               # PLW-203
MIN_EMPLOYED_DAYS = 30        # PLW-201
MIN_AGE = 18                  # ELG-301
VALID_EXIT_STATUSES = {"employed", "training_completed", "withdrew", "exited_other"}
