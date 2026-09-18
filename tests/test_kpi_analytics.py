import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from ecm_digital_thread.kpi_analytics import bottleneck_severity_score, kpi_summary


def _make_df():
    # Two stages, hand-computable BSS.
    rows = []
    # Stage A: cct=20, queue=8, fpar=0.5 (five records)
    for i in range(5):
        rows.append(dict(
            change_order_id=f"A{i}", month=1, workflow_stage="Design Review",
            cycle_time_days=20.0, queue_depth=8, first_pass_approval=(i % 2 == 0),
            approval_pending=False, documentation_error=False,
            linked_design=True, linked_manufacturing=True, linked_quality=True,
            linked_maintenance=True, ecm_class="Cost-Driven",
        ))
    # Stage B: cct=10, queue=2, fpar=1.0
    for i in range(5):
        rows.append(dict(
            change_order_id=f"B{i}", month=1, workflow_stage="Final Release",
            cycle_time_days=10.0, queue_depth=2, first_pass_approval=True,
            approval_pending=False, documentation_error=False,
            linked_design=True, linked_manufacturing=True, linked_quality=True,
            linked_maintenance=True, ecm_class="Cost-Driven",
        ))
    return pd.DataFrame(rows)


def test_bss_manual_calculation():
    df = _make_df()
    baseline = 18.5
    alpha, beta, gamma = 0.5, 0.3, 0.2
    result = bottleneck_severity_score(df, cct_baseline=baseline, alpha=alpha, beta=beta, gamma=gamma)

    q_avg = df["queue_depth"].mean()  # (8*5 + 2*5)/10 = 5.0
    assert q_avg == pytest.approx(5.0)

    stage_a = result[result["workflow_stage"] == "Design Review"].iloc[0]
    # first_pass_approval = (i % 2 == 0) for i in 0..4 -> True,False,True,False,True -> mean 0.6
    expected_bss_a = alpha * (20.0 / baseline) + beta * (8 / q_avg) + gamma * (1 - 0.6)
    assert stage_a["BSS"] == pytest.approx(expected_bss_a, abs=1e-3)

    stage_b = result[result["workflow_stage"] == "Final Release"].iloc[0]
    expected_bss_b = alpha * (10.0 / baseline) + beta * (2 / q_avg) + gamma * (1 - 1.0)
    assert stage_b["BSS"] == pytest.approx(expected_bss_b, abs=1e-3)

    # Stage A should be more of a bottleneck than Stage B (higher cct, queue, lower fpar).
    assert stage_a["BSS"] > stage_b["BSS"]


def test_bss_requires_weights_sum_to_one():
    df = _make_df()
    with pytest.raises(AssertionError):
        bottleneck_severity_score(df, alpha=0.5, beta=0.5, gamma=0.5)


def test_kpi_summary_shapes():
    df = _make_df()
    overall = kpi_summary(df)
    assert len(overall) == 1
    grouped = kpi_summary(df, group_col="workflow_stage")
    assert len(grouped) == 2
