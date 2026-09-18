#!/usr/bin/env python3
"""Step 5 — generate Tables 1-4, Figures 2-5, and the analytics dashboard."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecm_digital_thread import config as C
from ecm_digital_thread.dashboard import build_dashboard_data, write_html_dashboard, export_sqlite

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})


def table1_before_after(monthly_kpis: pd.DataFrame) -> pd.DataFrame:
    before = monthly_kpis.iloc[0]
    after = monthly_kpis.iloc[-1]
    rows = []
    for col in ["Change Cycle Time (days)", "Approval Pending Rate (%)",
                "First-Pass Approval Rate (%)", "Traceability Coverage Index (%)",
                "Documentation Error Rate (%)", "Change Propagation Index (%)"]:
        b, a = before[col], after[col]
        pct = 100.0 * (a - b) / b if b else 0.0
        direction = "down" if pct < 0 else "up"
        rows.append({
            "Metric": col, "Before Implementation": round(b, 2), "After Implementation": round(a, 2),
            "Improvement (%)": f"{abs(pct):.1f}% {direction}",
        })
    return pd.DataFrame(rows)


def table2_nlp_report() -> pd.DataFrame:
    report_path = C.RESULTS_TABLES_DIR / "table2_nlp_classification_report.json"
    report = json.loads(report_path.read_text())
    labels = sorted(C.ECM_CLASS_PRIOR.keys())
    rows = []
    for label in labels:
        m = report[label]
        rows.append({"Category": label, "Precision": round(m["precision"], 2),
                      "Recall": round(m["recall"], 2), "F1-Score": round(m["f1-score"], 2)})
    macro = report["macro avg"]
    rows.append({"Category": "Overall Average", "Precision": round(macro["precision"], 2),
                  "Recall": round(macro["recall"], 2), "F1-Score": round(macro["f1-score"], 2)})
    return pd.DataFrame(rows)


TABLE3_QUALITATIVE = pd.DataFrame([
    {"Framework": "Traditional ECM", "Digital Thread": "No", "NLP Automation": "No",
     "Real-Time Dashboard": "Limited", "Bottleneck Detection": "Manual", "Traceability": "Partial"},
    {"Framework": "AI-Based Predictive ECM", "Digital Thread": "Partial", "NLP Automation": "Limited",
     "Real-Time Dashboard": "Moderate", "Bottleneck Detection": "Semi-Automatic", "Traceability": "Moderate"},
    {"Framework": "Process Mining ECM", "Digital Thread": "Partial", "NLP Automation": "No",
     "Real-Time Dashboard": "Moderate", "Bottleneck Detection": "Automatic", "Traceability": "Moderate"},
    {"Framework": "Proposed Framework", "Digital Thread": "Yes", "NLP Automation": "Yes",
     "Real-Time Dashboard": "Real-Time", "Bottleneck Detection": "Automatic", "Traceability": "High"},
])


def table4_benchmarking(monthly_kpis: pd.DataFrame) -> pd.DataFrame:
    """Table 4 as published in the paper (quantitative benchmarking against

    other frameworks from the literature). All four rows -- including
    "Proposed Framework" -- are the paper's own reported figures, quoted
    verbatim. This table is a literature citation, not a live computation:
    the three comparison frameworks aren't implemented in this repo, so
    there is nothing to regenerate them from, and the "Proposed Framework"
    row must stay identical to the paper's Table 4 rather than being
    recomputed from this run's demonstration dataset (see Table 1 / the
    figures above for the toolkit's own reproducible demonstration output,
    which is a separate, clearly-labeled number by design). The
    `monthly_kpis` parameter is accepted for interface compatibility with
    the rest of the pipeline but is intentionally unused here.
    """
    del monthly_kpis  # unused: this table's figures come from the paper, not this run

    rows = [
        {"Framework": "Traditional ECM Systems", "Change Cycle Time Reduction (%)": 12.4,
         "First-Pass Approval Rate (%)": 52, "Traceability Coverage (%)": 48, "Documentation Error Reduction (%)": 18},
        {"Framework": "Process Mining-Based ECM", "Change Cycle Time Reduction (%)": 26.8,
         "First-Pass Approval Rate (%)": 64, "Traceability Coverage (%)": 67, "Documentation Error Reduction (%)": 34},
        {"Framework": "AI-Assisted Predictive ECM", "Change Cycle Time Reduction (%)": 35.2,
         "First-Pass Approval Rate (%)": 73, "Traceability Coverage (%)": 79, "Documentation Error Reduction (%)": 46},
        {"Framework": "Proposed Framework", "Change Cycle Time Reduction (%)": 44.9,
         "First-Pass Approval Rate (%)": 81, "Traceability Coverage (%)": 92, "Documentation Error Reduction (%)": 61.9},
    ]
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, name: str) -> None:
    csv_path = C.RESULTS_TABLES_DIR / f"{name}.csv"
    md_path = C.RESULTS_TABLES_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False))


def _month_xticks(ax, monthly_kpis: pd.DataFrame) -> None:
    months = monthly_kpis["month"].tolist()
    labels = [C.MONTH_LABELS[m - 1] if 1 <= m <= len(C.MONTH_LABELS) else str(m) for m in months]
    ax.set_xticks(months)
    ax.set_xticklabels(labels, rotation=20, ha="right")


def figure2_cycle_time(monthly_kpis: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(monthly_kpis["month"], monthly_kpis["Change Cycle Time (days)"], marker="o", color="#2563eb")
    _month_xticks(ax, monthly_kpis)
    ax.set_xlabel("Month"); ax.set_ylabel("Cycle Time (Days)")
    ax.set_title("Figure 2: Change Cycle Time Reduction")
    fig.tight_layout()
    fig.savefig(C.RESULTS_FIGURES_DIR / "figure2_change_cycle_time_reduction.png", dpi=150)
    plt.close(fig)


def figure3_bottleneck_distribution(bss_by_stage_month: pd.DataFrame, months: list) -> None:
    m_first, m_last = months[0], months[-1]
    before = bss_by_stage_month[bss_by_stage_month["month"] == m_first].set_index("workflow_stage")
    after = bss_by_stage_month[bss_by_stage_month["month"] == m_last].set_index("workflow_stage")
    stages = C.LIFECYCLE_STAGES

    x = range(len(stages))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - width / 2 for i in x], [before.loc[s, "BSS"] if s in before.index else 0 for s in stages],
           width=width, label="Before Implementation", color="#2563eb")
    ax.bar([i + width / 2 for i in x], [after.loc[s, "BSS"] if s in after.index else 0 for s in stages],
           width=width, label="After Implementation", color="#f97316")
    ax.set_xticks(list(x)); ax.set_xticklabels(stages, rotation=15)
    ax.set_ylabel("Bottleneck Severity Score (BSS)")
    ax.set_title("Figure 3: Approval Pipeline Bottleneck Distribution")
    ax.axhline(C.BSS_FLAG_THRESHOLD, color="red", linestyle="--", linewidth=1, label="Bottleneck threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.RESULTS_FIGURES_DIR / "figure3_bottleneck_distribution.png", dpi=150)
    plt.close(fig)


def figure4_fpar_trend(monthly_kpis: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(monthly_kpis["month"], monthly_kpis["First-Pass Approval Rate (%)"], marker="o", color="#16a34a")
    _month_xticks(ax, monthly_kpis)
    ax.set_xlabel("Month"); ax.set_ylabel("Approval Rate (%)")
    ax.set_title("Figure 4: First-Pass Approval Rate Trend")
    fig.tight_layout()
    fig.savefig(C.RESULTS_FIGURES_DIR / "figure4_first_pass_approval_trend.png", dpi=150)
    plt.close(fig)


def figure5_tci_trend(monthly_kpis: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(monthly_kpis["month"], monthly_kpis["Traceability Coverage Index (%)"], marker="o", color="#9333ea")
    _month_xticks(ax, monthly_kpis)
    ax.set_xlabel("Month"); ax.set_ylabel("Traceability Coverage (%)")
    ax.set_title("Figure 5: Traceability Coverage Improvement")
    fig.tight_layout()
    fig.savefig(C.RESULTS_FIGURES_DIR / "figure5_traceability_coverage_improvement.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    monthly_kpis = pd.read_csv(C.KPI_MONTHLY_CSV)
    bss = pd.read_csv(C.BSS_STAGE_CSV)
    df = pd.read_csv(C.NLP_SCORED_CSV) if C.NLP_SCORED_CSV.exists() else pd.read_csv(C.RAW_DATASET_CSV)

    t1 = table1_before_after(monthly_kpis); save_table(t1, "table1_ecm_performance_before_after")
    t2 = table2_nlp_report(); save_table(t2, "table2_nlp_classification_report")
    save_table(TABLE3_QUALITATIVE, "table3_qualitative_framework_comparison")
    t4 = table4_benchmarking(monthly_kpis); save_table(t4, "table4_quantitative_benchmarking")

    figure2_cycle_time(monthly_kpis)
    figure3_bottleneck_distribution(bss, monthly_kpis["month"].tolist())
    figure4_fpar_trend(monthly_kpis)
    figure5_tci_trend(monthly_kpis)

    nlp_report_path = C.RESULTS_TABLES_DIR / "table2_nlp_classification_report.json"
    nlp_report = json.loads(nlp_report_path.read_text())
    class_counts = df["ecm_class"].value_counts()
    dash_data = build_dashboard_data(monthly_kpis, bss, class_counts, nlp_report)
    html_path = write_html_dashboard(dash_data)
    db_path = export_sqlite(df, monthly_kpis, bss)

    print(f"[5/5] Tables written to {C.RESULTS_TABLES_DIR}")
    print(f"      Figures written to {C.RESULTS_FIGURES_DIR}")
    print(f"      Dashboard: {html_path}")
    print(f"      SQLite export: {db_path}")
    print("\nTable 1 — ECM Performance Metrics Before and After Implementation:\n", t1.to_string(index=False))
    print("\nTable 4 — Quantitative Benchmarking Against Existing ECM Frameworks:\n", t4.to_string(index=False))
