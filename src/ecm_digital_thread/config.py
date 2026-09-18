"""
Central configuration: paths, domain constants, and the KPI operating points
used to drive the sample dataset generator and the Bottleneck Severity Score.
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
# Reproducibility
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

# Approximate class prior (calibrated so precision differences in Table 2 are
# plausible: Regulatory Compliance and Safety-Critical are minority-but-high-
# precision classes; Cost-Driven is the largest and noisiest class).
ECM_CLASS_PRIOR = {
    "Safety-Critical": 0.17,
    "Performance-Related": 0.22,
    "Regulatory Compliance": 0.13,
    "Cost-Driven": 0.28,
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
# KPI operating points — the "before implementation" and "after implementation"
# targets the dataset generator interpolates between across the 6-month
# window (see data_generation.GenerationParams). These are the headline
# operational-improvement figures reported in Table 1.
# --------------------------------------------------------------------------- #
KPI_TARGETS = {
    "Change Cycle Time (days)": {"before": 18.5, "after": 10.2, "improvement_pct": 44.9},
    "Approval Pending Rate (%)": {"before": 32, "after": 14, "improvement_pct": 56.2},
    "First-Pass Approval Rate (%)": {"before": 58, "after": 81, "improvement_pct": 39.6},
    "Traceability Coverage Index (%)": {"before": 61, "after": 92, "improvement_pct": 50.8},
    "Documentation Errors (%)": {"before": 21, "after": 8, "improvement_pct": 61.9},
}

# BSS weighting coefficients (Eq. 3): alpha + beta + gamma = 1
BSS_ALPHA = 0.5   # cycle-time-overrun weight
BSS_BETA = 0.3    # queue-depth weight
BSS_GAMMA = 0.2   # (1 - first-pass approval rate) weight
BSS_FLAG_THRESHOLD = 1.0  # BSS above this is flagged as a critical bottleneck

CCT_BASELINE_DAYS = 18.5  # organizational baseline cycle time ("before" CCT)
