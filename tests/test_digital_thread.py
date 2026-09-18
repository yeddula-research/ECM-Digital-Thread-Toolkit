import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from ecm_digital_thread.digital_thread import (
    compute_full_linkage, traceability_coverage_index, build_unified_data_model,
)


def _sample_df(rows):
    return pd.DataFrame(rows, columns=[
        "change_order_id", "month", "source_system", "ecm_class", "workflow_stage",
        "cycle_time_days", "queue_depth", "first_pass_approval", "approval_pending",
        "documentation_error", "linked_design", "linked_manufacturing",
        "linked_quality", "linked_maintenance", "part_number", "assembly_id",
        "responsible_team", "affected_subsystem",
    ])


def test_tci_all_linked_is_100():
    rows = [["ECN-1", 1, "PLM", "Cost-Driven", "Design Review", 5.0, 1, True, False,
              False, True, True, True, True, "PN-1", "ASM-1", "T", "S"]]
    df = _sample_df(rows)
    assert traceability_coverage_index(df) == 100.0


def test_tci_none_linked_is_0():
    rows = [["ECN-1", 1, "PLM", "Cost-Driven", "Design Review", 5.0, 1, True, False,
              False, False, True, True, True, "PN-1", "ASM-1", "T", "S"]]
    df = _sample_df(rows)
    assert traceability_coverage_index(df) == 0.0  # not ALL phases linked


def test_tci_is_between_0_and_100_on_mixed_data():
    rows = [
        ["ECN-1", 1, "PLM", "Cost-Driven", "Design Review", 5.0, 1, True, False,
         False, True, True, True, True, "PN-1", "ASM-1", "T", "S"],
        ["ECN-2", 1, "PLM", "Cost-Driven", "Design Review", 5.0, 1, True, False,
         False, False, True, True, True, "PN-2", "ASM-2", "T", "S"],
    ]
    df = _sample_df(rows)
    tci = traceability_coverage_index(df)
    assert 0.0 <= tci <= 100.0
    assert tci == 50.0


def test_grouped_tci_returns_series():
    rows = [
        ["ECN-1", 1, "PLM", "Cost-Driven", "Design Review", 5.0, 1, True, False,
         False, True, True, True, True, "PN-1", "ASM-1", "T", "S"],
        ["ECN-2", 2, "PLM", "Cost-Driven", "Design Review", 5.0, 1, True, False,
         False, False, True, True, True, "PN-2", "ASM-2", "T", "S"],
    ]
    df = _sample_df(rows)
    result = traceability_coverage_index(df, group_col="month")
    assert result.loc[1] == 100.0
    assert result.loc[2] == 0.0


def test_udm_has_fully_linked_column():
    rows = [["ECN-1", 1, "PLM", "Cost-Driven", "Design Review", 5.0, 1, True, False,
              False, True, True, True, True, "PN-1", "ASM-1", "T", "S"]]
    df = _sample_df(rows)
    udm = build_unified_data_model(df)
    assert "fully_linked" in udm.columns
    assert bool(udm.loc[0, "fully_linked"]) is True
