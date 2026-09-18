import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ecm_digital_thread import config as C
from ecm_digital_thread.data_generation import generate_dataset


def test_record_count_and_columns():
    df = generate_dataset(n_records=300, seed=1)
    assert len(df) == 300
    expected_cols = {
        "change_order_id", "request_date", "month", "month_label", "source_system",
        "ecm_class", "workflow_stage",
        "cycle_time_days", "queue_depth", "first_pass_approval", "approval_pending",
        "documentation_error", "linked_design", "linked_manufacturing",
        "linked_quality", "linked_maintenance", "part_number", "assembly_id",
        "responsible_team", "affected_subsystem", "raw_text",
    }
    assert expected_cols.issubset(set(df.columns))


def test_request_date_within_configured_calendar_window():
    import calendar as _calendar
    import datetime as _dt
    df = generate_dataset(n_records=300, seed=5)
    dates = pd.to_datetime(df["request_date"])
    window_start = _dt.date(C.DATASET_START_YEAR, C.DATASET_START_MONTH, 1)
    end_year, end_month = C.month_index_to_year_month(C.N_MONTHS)
    window_end = _dt.date(end_year, end_month, _calendar.monthrange(end_year, end_month)[1])
    assert dates.min().date() >= window_start
    assert dates.max().date() <= window_end


def test_classes_are_from_known_set():
    df = generate_dataset(n_records=200, seed=2)
    assert set(df["ecm_class"].unique()).issubset(set(C.ECM_CLASSES))


def test_deterministic_with_fixed_seed():
    df1 = generate_dataset(n_records=150, seed=7)
    df2 = generate_dataset(n_records=150, seed=7)
    assert df1.equals(df2)


def test_different_seeds_differ():
    df1 = generate_dataset(n_records=150, seed=7)
    df2 = generate_dataset(n_records=150, seed=8)
    assert not df1["raw_text"].equals(df2["raw_text"])


def test_months_within_range():
    df = generate_dataset(n_records=300, seed=3)
    assert df["month"].min() >= 1
    assert df["month"].max() <= C.N_MONTHS


def test_cycle_time_positive():
    df = generate_dataset(n_records=300, seed=4)
    assert (df["cycle_time_days"] > 0).all()
