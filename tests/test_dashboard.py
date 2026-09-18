import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from ecm_digital_thread.dashboard import build_dashboard_data, write_html_dashboard


def test_dashboard_data_is_json_serializable(tmp_path):
    monthly_kpis = pd.DataFrame({
        "month": [1, 2],
        "Change Cycle Time (days)": [18.0, 10.0],
        "Approval Pending Rate (%)": [30.0, 15.0],
        "First-Pass Approval Rate (%)": [60.0, 80.0],
        "Traceability Coverage Index (%)": [60.0, 90.0],
        "Documentation Error Rate (%)": [20.0, 9.0],
        "Change Propagation Index (%)": [70.0, 90.0],
    })
    bss = pd.DataFrame({
        "month": [1, 1, 2, 2],
        "workflow_stage": ["Design Review", "Final Release", "Design Review", "Final Release"],
        "BSS": [0.8, 0.6, 0.5, 0.4],
    })
    class_counts = pd.Series({"Cost-Driven": 5, "Safety-Critical": 3})
    nlp_report = {c: {"precision": 0.8, "recall": 0.8, "f1-score": 0.8} for c in
                  ["Safety-Critical", "Performance-Related", "Regulatory Compliance",
                   "Cost-Driven", "Customer-Requested"]}

    data = build_dashboard_data(monthly_kpis, bss, class_counts, nlp_report)
    json.dumps(data)  # must not raise

    out = write_html_dashboard(data, out_path=tmp_path / "index.html")
    html = out.read_text()
    assert "<html" in html
    assert "chart.js" in html.lower() or "chart.umd" in html.lower()
