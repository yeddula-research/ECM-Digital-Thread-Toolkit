"""
Central configuration: paths, domain constants, the demonstration scenario
that drives the synthetic dataset generator, and the Bottleneck Severity
Score settings.

Everything here that describes records, months or operating points belongs
to the synthetic demonstration dataset (see ``DATASET_DISCLOSURE``); none of
it is taken from the paper's case-study data or results.
"""
from __future__ import annotations
import calendar as _calendar
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DASHBOARD_DIR = RESULTS_DIR / "dashboard"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_TABLES_DIR,
           RESULTS_FIGURES_DIR, RESULTS_DASHBOARD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_DATASET_CSV = DATA_RAW_DIR / "ecm_demonstration_dataset.csv"
LINKED_DATASET_CSV = DATA_PROCESSED_DIR / "ecm_records_linked.csv"
NLP_SCORED_CSV = DATA_PROCESSED_DIR / "ecm_records_nlp_scored.csv"
KPI_MONTHLY_CSV = DATA_PROCESSED_DIR / "kpi_monthly.csv"
BSS_STAGE_CSV = DATA_PROCESSED_DIR / "bottleneck_severity_by_stage.csv"

# --------------------------------------------------------------------------- #
# Random seed (fixed, so every run generates the same dataset)
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Domain constants (Section III.F / III.A)
# --------------------------------------------------------------------------- #
N_RECORDS = 1200
N_MONTHS = 6

# Calendar anchor for the 6-month operational window. All KPI interpolation
# internally still runs on a plain relative index (month 1..N_MONTHS, see
# data_generation.GenerationParams) — these two constants only control how
# that window is *dated* for the record-level `request_date` field and for
# every chart/dashboard axis label, so results read as a normal dated
# operational report. Shift the whole window by changing these two values.
DATASET_START_YEAR = 2025
DATASET_START_MONTH = 1  # January


def month_index_to_year_month(month_index: int) -> tuple[int, int]:
    """Map a relative month index (1..N_MONTHS) to an absolute (year, month)
    pair, anchored at DATASET_START_YEAR / DATASET_START_MONTH."""
    total = (DATASET_START_MONTH - 1) + (month_index - 1)
    year = DATASET_START_YEAR + total // 12
    month = total % 12 + 1
    return year, month


MONTH_LABELS = [
    f"{_calendar.month_abbr[m]} {y}"
    for y, m in (month_index_to_year_month(i) for i in range(1, N_MONTHS + 1))
]

ECM_CLASSES = [
    "Safety-Critical",
    "Performance-Related",
    "Regulatory Compliance",
    "Cost-Driven",
    "Customer-Requested",
]

# Class shares in the demonstration scenario: equal for all five classes.
ECM_CLASS_PRIOR = {
    "Safety-Critical": 0.20,
    "Performance-Related": 0.20,
    "Regulatory Compliance": 0.20,
    "Cost-Driven": 0.20,
    "Customer-Requested": 0.20,
}

LIFECYCLE_STAGES = [
    "Design Review",
    "Quality Approval",
    "Manufacturing",
    "Final Release",
]

SOURCE_SYSTEMS = [
    "PLM", "ERP", "CAD", "MMS", "Supplier Portal", "ECN Database",
]

# --------------------------------------------------------------------------- #
# Demonstration scenario — month-1 and month-6 operating points that the
# synthetic dataset generator interpolates between (see
# data_generation.GenerationParams). They are round, illustrative values
# chosen for this toolkit so that the pipeline has a before/after trend to
# measure; they are an input assumption of the demonstration, not an outcome
# the toolkit measures, and they are not the paper's reported values (those
# are recorded separately in published.py).
# --------------------------------------------------------------------------- #
DEMO_SCENARIO = {
    "Change Cycle Time (days)": {"month_1": 20.0, "month_6": 14.0},
    "Approval Pending Rate (%)": {"month_1": 30.0, "month_6": 20.0},
    "First-Pass Approval Rate (%)": {"month_1": 60.0, "month_6": 72.0},
    "Traceability Coverage Index (%)": {"month_1": 55.0, "month_6": 80.0},
    "Documentation Errors (%)": {"month_1": 15.0, "month_6": 10.0},
}

# One sentence, used verbatim wherever the dataset is described.
DATASET_DISCLOSURE = (
    "The demonstration dataset is synthetic: 1,200 engineering change records "
    "for a fictional heavy-machinery manufacturer, generated with a fixed "
    "random seed by scripts/01_generate_dataset.py from the illustrative "
    "scenario in config.DEMO_SCENARIO; it is not the paper's case-study data, "
    "and its before/after trend is an input of the scenario rather than a "
    "measured effect."
)

# BSS weighting coefficients (Eq. 3): alpha + beta + gamma = 1
BSS_ALPHA = 0.5   # cycle-time-overrun weight
BSS_BETA = 0.3    # queue-depth weight
BSS_GAMMA = 0.2   # (1 - first-pass approval rate) weight
BSS_FLAG_THRESHOLD = 1.0  # BSS above this is flagged as a critical bottleneck

# Organizational baseline cycle time for Eq. 3: the demonstration scenario's
# month-1 mean change cycle time.
CCT_BASELINE_DAYS = DEMO_SCENARIO["Change Cycle Time (days)"]["month_1"]
