#!/usr/bin/env python3
"""Step 6 — the paper's Tables 1-4 and consistency checks on them.

Nothing here is computed from the demonstration dataset: the tables are
copied from the paper (source: the paper), and the checks recompute the
paper's own percentages, F1 scores and averages from its own values.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ecm_digital_thread import config as C
from ecm_digital_thread import published as P


def save_table(df: pd.DataFrame, name: str, title: str) -> None:
    df.to_csv(C.RESULTS_TABLES_DIR / f"{name}.csv", index=False)
    (C.RESULTS_TABLES_DIR / f"{name}.md").write_text(
        f"**{title}**\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    save_table(P.TABLE1, "table1_ecm_performance_before_after_as_reported",
               "Table 1 - ECM Performance Metrics Before and After Implementation (source: the paper)")
    save_table(P.TABLE2, "table2_nlp_classification_report_as_reported",
               "Table 2 - NLP Model Performance for Change Request Classification (source: the paper)")
    save_table(P.TABLE3, "table3_qualitative_framework_comparison_as_reported",
               "Table 3 - Comparison with Existing AI-Based ECM Frameworks (source: the paper)")
    save_table(P.TABLE4, "table4_quantitative_benchmarking_as_reported",
               "Table 4 - Quantitative Benchmarking Against Existing ECM Frameworks (source: the paper)")

    t1 = P.table1_improvement_check()
    f1 = P.table2_f1_check()
    avg = P.table2_average_check()
    t4 = P.table4_matches_table1()
    save_table(t1, "table1_improvement_check_as_reported",
               "Table 1 improvements recomputed from the paper's before/after values")
    save_table(f1, "table2_f1_check_as_reported",
               "Table 2 F1 scores recomputed from the paper's precision and recall")
    (C.RESULTS_TABLES_DIR / "published_consistency_checks.json").write_text(
        json.dumps({"table2_overall_average": avg, "table4_proposed_row_vs_table1": t4}, indent=2) + "\n",
        encoding="utf-8")

    print(f"[6/6] The paper's Tables 1-4 ({P.SOURCE}) written to {C.RESULTS_TABLES_DIR}/*_as_reported.*")
    print(f"      Table 1: largest gap between reported and recomputed improvement: "
          f"{t1['Difference (pp)'].abs().max():.2f} percentage points")
    print(f"      Table 2: largest gap between reported F1 and harmonic mean of P and R: "
          f"{f1['Difference'].abs().max():.3f}")
    for col, v in avg.items():
        print(f"      Table 2 overall average, {col}: mean of classes {v['mean_of_classes']:.3f}, reported {v['reported']:.2f}")
    print(f"      Table 4 'Proposed Framework' row matches Table 1: {all(v['match'] for v in t4.values())}")
