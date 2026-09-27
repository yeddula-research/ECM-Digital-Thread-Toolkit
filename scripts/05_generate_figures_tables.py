#!/usr/bin/env python3
"""Step 5 — demonstration-dataset tables, figures and the analytics dashboard.

The paper's own Tables 1-4 are written separately by 06_published_values.py.
"""
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
    """KPIs of the first and last month of the demonstration dataset."""
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
            "Metric": col, "Month 1": round(b, 2), f"Month {C.N_MONTHS}": round(a, 2),
            "Change (%)": f"{abs(pct):.1f}% {direction}",
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
    ax.set_title("Change cycle time by month (demonstration dataset)")
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
           width=width, label="Month 1", color="#2563eb")
    ax.bar([i + width / 2 for i in x], [after.loc[s, "BSS"] if s in after.index else 0 for s in stages],
           width=width, label=f"Month {C.N_MONTHS}", color="#f97316")
    ax.set_xticks(list(x)); ax.set_xticklabels(stages, rotation=15)
    ax.set_ylabel("Bottleneck Severity Score (BSS)")
    ax.set_title("Bottleneck Severity Score by stage (demonstration dataset)")
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
    ax.set_title("First-pass approval rate by month (demonstration dataset)")
    fig.tight_layout()
    fig.savefig(C.RESULTS_FIGURES_DIR / "figure4_first_pass_approval_trend.png", dpi=150)
    plt.close(fig)


def figure5_tci_trend(monthly_kpis: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(monthly_kpis["month"], monthly_kpis["Traceability Coverage Index (%)"], marker="o", color="#9333ea")
    _month_xticks(ax, monthly_kpis)
    ax.set_xlabel("Month"); ax.set_ylabel("Traceability Coverage (%)")
    ax.set_title("Traceability Coverage Index by month (demonstration dataset)")
    fig.tight_layout()
    fig.savefig(C.RESULTS_FIGURES_DIR / "figure5_traceability_coverage_improvement.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    monthly_kpis = pd.read_csv(C.KPI_MONTHLY_CSV)
    bss = pd.read_csv(C.BSS_STAGE_CSV)
    df = pd.read_csv(C.NLP_SCORED_CSV) if C.NLP_SCORED_CSV.exists() else pd.read_csv(C.RAW_DATASET_CSV)

    t1 = table1_before_after(monthly_kpis); save_table(t1, "table1_ecm_performance_before_after")
    t2 = table2_nlp_report(); save_table(t2, "table2_nlp_classification_report")

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

    print(f"[5/6] Tables written to {C.RESULTS_TABLES_DIR}")
    print(f"      Figures written to {C.RESULTS_FIGURES_DIR}")
    print(f"      Dashboard: {html_path}")
    print(f"      SQLite export: {db_path}")
    print("\nDemonstration KPIs, month 1 vs month 6 (this run):\n", t1.to_string(index=False))
    print("\nNLP classification report (this run):\n", t2.to_string(index=False))
