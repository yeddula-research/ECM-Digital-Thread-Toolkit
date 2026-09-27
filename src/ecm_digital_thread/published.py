"""
The paper's published results, recorded for side-by-side checking.

Tables 1-4 below are copied exactly as printed in the ICCBI-2026
proceedings (pp. 660-662). They are reference data only: nothing in the
demonstration pipeline reads them. The helpers below run consistency checks
on the published tables themselves, so a reader can confirm that their
reported percentages, averages and F1 scores follow from their own values.
"""
from __future__ import annotations

import pandas as pd

SOURCE = (
    "Yeddula, Shaikh, Choksi, ICCBI-2026 proceedings, pp. 655-663, "
    "ISBN 979-8-3315-6379-0"
)

# Table 1 - ECM Performance Metrics Before and After Implementation (p. 660).
TABLE1 = pd.DataFrame(
    [
        ("Change Cycle Time (days)", 18.5, 10.2, 44.9, "down"),
        ("Approval Pending Rate (%)", 32, 14, 56.2, "down"),
        ("First-Pass Approval Rate (%)", 58, 81, 39.6, "up"),
        ("Traceability Coverage Index (%)", 61, 92, 50.8, "up"),
        ("Documentation Errors (%)", 21, 8, 61.9, "down"),
    ],
    columns=["Metric", "Before Implementation", "After Implementation", "Improvement (%)", "Direction"],
)

# Table 2 - NLP Model Performance for Change Request Classification (p. 661).
TABLE2 = pd.DataFrame(
    [
        ("Safety-Critical", 0.91, 0.89, 0.90),
        ("Performance-Related", 0.88, 0.86, 0.87),
        ("Regulatory Compliance", 0.93, 0.91, 0.92),
        ("Cost-Driven", 0.85, 0.83, 0.84),
        ("Customer-Requested", 0.87, 0.85, 0.86),
        ("Overall Average", 0.89, 0.87, 0.88),
    ],
    columns=["Category", "Precision", "Recall", "F1-Score"],
)

# Table 3 - Comparison with Existing AI-Based ECM Frameworks (p. 662).
TABLE3 = pd.DataFrame(
    [
        ("Traditional ECM", "No", "No", "Limited", "Manual", "Partial"),
        ("AI-Based Predictive ECM", "Partial", "Limited", "Moderate", "Semi-Automatic", "Moderate"),
        ("Process Mining ECM", "Partial", "No", "Moderate", "Automatic", "Moderate"),
        ("Proposed Framework", "Yes", "Yes", "Real-Time Power BI", "Automatic", "High"),
    ],
    columns=["Framework", "Digital Thread", "NLP Automation", "Real-Time Dashboard", "Bottleneck Detection", "Traceability"],
)

# Table 4 - Quantitative Benchmarking Against Existing ECM Frameworks (p. 662).
TABLE4 = pd.DataFrame(
    [
        ("Traditional ECM Systems", 12.4, 52, 48, 18),
        ("Process Mining-Based ECM", 26.8, 64, 67, 34),
        ("AI-Assisted Predictive ECM", 35.2, 73, 79, 46),
        ("Proposed Framework", 44.9, 81, 92, 61.9),
    ],
    columns=[
        "Framework",
        "Change Cycle Time Reduction (%)",
        "First-Pass Approval Rate (%)",
        "Traceability Coverage (%)",
        "Documentation Error Reduction (%)",
    ],
)


def table1_improvement_check() -> pd.DataFrame:
    """Recompute each Table 1 improvement from its before/after values."""
    out = TABLE1.copy()
    before = out["Before Implementation"].astype(float)
    after = out["After Implementation"].astype(float)
    out["Recomputed (%)"] = (100.0 * (after - before).abs() / before).round(2)
    out["Difference (pp)"] = (out["Recomputed (%)"] - out["Improvement (%)"]).round(2)
    return out[["Metric", "Before Implementation", "After Implementation", "Improvement (%)", "Recomputed (%)", "Difference (pp)"]]


def table2_f1_check() -> pd.DataFrame:
    """Recompute each Table 2 F1 as the harmonic mean of its precision and recall."""
    out = TABLE2.copy()
    p, r = out["Precision"], out["Recall"]
    out["Recomputed F1"] = (2 * p * r / (p + r)).round(3)
    out["Difference"] = (out["Recomputed F1"] - out["F1-Score"]).round(3)
    return out


def table2_average_check() -> dict:
    """Mean of the five class rows of Table 2 versus its 'Overall Average' row."""
    classes = TABLE2[TABLE2["Category"] != "Overall Average"]
    overall = TABLE2[TABLE2["Category"] == "Overall Average"].iloc[0]
    return {
        col: {"mean_of_classes": round(float(classes[col].mean()), 3), "reported": float(overall[col])}
        for col in ("Precision", "Recall", "F1-Score")
    }


def table4_matches_table1() -> dict:
    """The 'Proposed Framework' row of Table 4 against the corresponding Table 1 values."""
    t1 = TABLE1.set_index("Metric")
    proposed = TABLE4.set_index("Framework").loc["Proposed Framework"]
    pairs = {
        "Change Cycle Time Reduction (%)": float(t1.loc["Change Cycle Time (days)", "Improvement (%)"]),
        "First-Pass Approval Rate (%)": float(t1.loc["First-Pass Approval Rate (%)", "After Implementation"]),
        "Traceability Coverage (%)": float(t1.loc["Traceability Coverage Index (%)", "After Implementation"]),
        "Documentation Error Reduction (%)": float(t1.loc["Documentation Errors (%)", "Improvement (%)"]),
    }
    return {col: {"table4": float(proposed[col]), "table1": value, "match": float(proposed[col]) == value}
            for col, value in pairs.items()}
