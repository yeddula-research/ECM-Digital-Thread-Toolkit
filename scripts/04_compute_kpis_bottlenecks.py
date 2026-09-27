#!/usr/bin/env python3
"""Step 4 — DAX-style KPI measures + Bottleneck Severity Score (Eq. 3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecm_digital_thread import config as C
from ecm_digital_thread.kpi_analytics import run

if __name__ == "__main__":
    monthly_kpis, bss = run()
    print(f"[4/6] Monthly KPI table written -> {C.KPI_MONTHLY_CSV}")
    print(f"      Bottleneck Severity Score table written -> {C.BSS_STAGE_CSV}")
    flagged = bss[bss["is_bottleneck"]]
    if len(flagged):
        print("      Stages flagged as critical bottlenecks (BSS > "
              f"{C.BSS_FLAG_THRESHOLD}):")
        print(flagged[["month", "workflow_stage", "BSS"]].to_string(index=False))
    else:
        print("      No stage exceeded the bottleneck threshold.")
