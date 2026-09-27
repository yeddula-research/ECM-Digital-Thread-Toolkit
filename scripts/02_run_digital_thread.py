#!/usr/bin/env python3
"""Step 2 — Digital Thread integration + Traceability Coverage Index (Eq. 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecm_digital_thread import config as C
from ecm_digital_thread.digital_thread import run, traceability_coverage_index

if __name__ == "__main__":
    udm = run()
    overall_tci = traceability_coverage_index(udm)
    print(f"[2/6] Digital Thread UDM written -> {C.LINKED_DATASET_CSV}")
    print(f"      Overall Traceability Coverage Index (TCI): {overall_tci}%")
