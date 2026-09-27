"""
Digital Thread integration layer — implements Section III.A (Layer 2) and
III.B of the paper: a Unified Data Model (UDM) that establishes bidirectional
traceability links across lifecycle phases (design, manufacturing, quality,
maintenance) for every engineering change record, and the Traceability
Coverage Index (Eq. 1):

    TCI = (N_linked / N_total) * 100

A record counts as "linked" only if it has a traceable connection to *every*
lifecycle phase touched by that change (i.e. all four linkage flags true) —
this is a stricter, auditable definition of "fully connected" than counting
any single link, consistent with the paper's framing of TCI as full
cross-lifecycle traceability.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

LINK_COLUMNS = [
    "linked_design", "linked_manufacturing", "linked_quality", "linked_maintenance",
]


def compute_full_linkage(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `fully_linked` boolean column: True iff all lifecycle-phase
    linkage flags are True for that record."""
    df = df.copy()
    df["fully_linked"] = df[LINK_COLUMNS].all(axis=1)
    df["link_count"] = df[LINK_COLUMNS].sum(axis=1)
    return df


def traceability_coverage_index(df: pd.DataFrame, group_col: str | None = None) -> pd.Series | float:
    """Eq. 1: TCI = N_linked / N_total * 100.

    If `group_col` is provided (e.g. 'month'), returns a Series of TCI per
    group; otherwise returns a single float for the whole dataset.
    """
    df = compute_full_linkage(df) if "fully_linked" not in df.columns else df
    if group_col is None:
        n_total = len(df)
        n_linked = int(df["fully_linked"].sum())
        return 0.0 if n_total == 0 else round(100.0 * n_linked / n_total, 2)

    def _tci(g):
        n_total = len(g)
        n_linked = int(g["fully_linked"].sum())
        return 0.0 if n_total == 0 else round(100.0 * n_linked / n_total, 2)

    return df.groupby(group_col).apply(_tci, include_groups=False)


def build_unified_data_model(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a normalized Unified Data Model (UDM) view: one row per
    engineering change record with a standardized schema mapping the record
    to its lifecycle-phase linkage state, workflow stage and source system —
    the structure a real Digital Thread middleware layer would expose to
    downstream NLP and analytics consumers (Layers 3 and 4 in Fig. 1)."""
    df = compute_full_linkage(df)
    udm_cols = [
        "change_order_id", "request_date", "month", "month_label",
        "source_system", "ecm_class",
        "workflow_stage", "cycle_time_days", "queue_depth",
        "first_pass_approval", "approval_pending", "documentation_error",
        "fully_linked", "link_count",
        "linked_design", "linked_manufacturing", "linked_quality", "linked_maintenance",
        "part_number", "assembly_id", "responsible_team", "affected_subsystem",
    ]
    # Tolerate a df built without the calendar columns (e.g. hand-built test
    # fixtures that predate this field) rather than raising a KeyError.
    udm_cols = [c for c in udm_cols if c in df.columns]
    return df[udm_cols].copy()


def run(input_csv=None, output_csv=None) -> pd.DataFrame:
    input_csv = input_csv or C.RAW_DATASET_CSV
    output_csv = output_csv or C.LINKED_DATASET_CSV
    df = pd.read_csv(input_csv)
    df = compute_full_linkage(df)
    udm = build_unified_data_model(df)
    udm.to_csv(output_csv, index=False)
    return udm


if __name__ == "__main__":
    udm = run()
    overall_tci = traceability_coverage_index(udm)
    monthly_tci = traceability_coverage_index(udm, group_col="month")
    print(f"Overall TCI: {overall_tci}%")
    print("Monthly TCI:")
    print(monthly_tci)
