"""
Power BI-equivalent analytics layer — replicates Section III.D of the paper:
custom DAX-style measures for the core ECM KPIs (Change Cycle Time, Approval
Pending Rate, First-Pass Approval Rate, Change Propagation Index,
Documentation Error Rate) and the Bottleneck Severity Score (Eq. 3):

    BSS_i = alpha * (CCT_i_bar / CCT_baseline_bar)
          + beta  * (Q_i / Q_avg)
          + gamma * (1 - FPAR_i)
    subject to alpha + beta + gamma == 1

Every measure here is implemented in plain pandas so it can be recomputed by
anyone without Power BI/DAX — the numeric definitions are kept 1:1 with the
paper text so a reader can check the code against the equations directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .digital_thread import compute_full_linkage, traceability_coverage_index

MAX_LINKS = 4  # linked_design, linked_manufacturing, linked_quality, linked_maintenance


def _rate(series: pd.Series) -> float:
    return 0.0 if len(series) == 0 else round(100.0 * series.mean(), 2)


def change_cycle_time(df: pd.DataFrame, group_col: str | None = None):
    """Mean Change Cycle Time (CCT), in days."""
    if group_col is None:
        return round(df["cycle_time_days"].mean(), 2)
    return df.groupby(group_col)["cycle_time_days"].mean().round(2)


def approval_pending_rate(df: pd.DataFrame, group_col: str | None = None):
    if group_col is None:
        return _rate(df["approval_pending"])
    return df.groupby(group_col)["approval_pending"].mean().mul(100).round(2)


def first_pass_approval_rate(df: pd.DataFrame, group_col: str | None = None):
    if group_col is None:
        return _rate(df["first_pass_approval"])
    return df.groupby(group_col)["first_pass_approval"].mean().mul(100).round(2)


def documentation_error_rate(df: pd.DataFrame, group_col: str | None = None):
    if group_col is None:
        return _rate(df["documentation_error"])
    return df.groupby(group_col)["documentation_error"].mean().mul(100).round(2)


def change_propagation_index(df: pd.DataFrame, group_col: str | None = None):
    """Change Propagation Index (CPI): average share of lifecycle phases
    (design/manufacturing/quality/maintenance) a change request's Digital
    Thread linkage actually reaches, expressed as a percentage of the
    maximum possible (4) phases. The paper names CPI as a KPI (Section
    III.D) without publishing a closed-form definition; this operational
    definition is documented here explicitly so it can be swapped for an
    organization-specific one without touching the rest of the pipeline.
    """
    df = compute_full_linkage(df) if "link_count" not in df.columns else df
    if group_col is None:
        return round(100.0 * df["link_count"].mean() / MAX_LINKS, 2)
    return (df.groupby(group_col)["link_count"].mean() / MAX_LINKS * 100).round(2)


def kpi_summary(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Bundle all KPIs (+ TCI) into one table, optionally grouped (e.g. by
    month), mirroring a Power BI KPI card row."""
    df = compute_full_linkage(df)
    if group_col is None:
        return pd.DataFrame([{
            "Change Cycle Time (days)": change_cycle_time(df),
            "Approval Pending Rate (%)": approval_pending_rate(df),
            "First-Pass Approval Rate (%)": first_pass_approval_rate(df),
            "Traceability Coverage Index (%)": traceability_coverage_index(df),
            "Documentation Error Rate (%)": documentation_error_rate(df),
            "Change Propagation Index (%)": change_propagation_index(df),
        }])

    out = pd.DataFrame({
        "Change Cycle Time (days)": change_cycle_time(df, group_col),
        "Approval Pending Rate (%)": approval_pending_rate(df, group_col),
        "First-Pass Approval Rate (%)": first_pass_approval_rate(df, group_col),
        "Traceability Coverage Index (%)": traceability_coverage_index(df, group_col),
        "Documentation Error Rate (%)": documentation_error_rate(df, group_col),
        "Change Propagation Index (%)": change_propagation_index(df, group_col),
    })
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Bottleneck Severity Score (Eq. 3)
# --------------------------------------------------------------------------- #

def bottleneck_severity_score(df: pd.DataFrame,
                               cct_baseline: float = C.CCT_BASELINE_DAYS,
                               alpha: float = C.BSS_ALPHA,
                               beta: float = C.BSS_BETA,
                               gamma: float = C.BSS_GAMMA,
                               group_col: str | None = None) -> pd.DataFrame:
    """Eq. 3: BSS_i = alpha*(CCT_i/CCT_baseline) + beta*(Q_i/Q_avg) + gamma*(1-FPAR_i)

    Computed per workflow stage, optionally within each group (e.g. per
    month, to drive a time-varying bottleneck heatmap). Returns a tidy
    DataFrame with one row per (group, stage).
    """
    assert abs((alpha + beta + gamma) - 1.0) < 1e-9, "alpha+beta+gamma must equal 1"

    def _per_period(sub: pd.DataFrame) -> pd.DataFrame:
        stage_cct = sub.groupby("workflow_stage")["cycle_time_days"].mean()
        stage_q = sub.groupby("workflow_stage")["queue_depth"].mean()
        stage_fpar = sub.groupby("workflow_stage")["first_pass_approval"].mean()
        q_avg = sub["queue_depth"].mean() if sub["queue_depth"].mean() > 0 else 1e-9

        rows = []
        for stage in C.LIFECYCLE_STAGES:
            cct_i = stage_cct.get(stage, np.nan)
            q_i = stage_q.get(stage, np.nan)
            fpar_i = stage_fpar.get(stage, np.nan)
            if np.isnan(cct_i):
                continue
            bss = (alpha * (cct_i / cct_baseline)
                   + beta * (q_i / q_avg)
                   + gamma * (1 - fpar_i))
            rows.append({
                "workflow_stage": stage,
                "mean_cct_days": round(cct_i, 2),
                "mean_queue_depth": round(q_i, 2),
                "first_pass_approval_rate": round(fpar_i, 4),
                "BSS": round(float(bss), 4),
                "is_bottleneck": bool(bss > C.BSS_FLAG_THRESHOLD),
            })
        return pd.DataFrame(rows)

    if group_col is None:
        result = _per_period(df)
        result.insert(0, "group", "overall")
        return result

    frames = []
    for key, sub in df.groupby(group_col):
        piece = _per_period(sub)
        piece.insert(0, group_col, key)
        frames.append(piece)
    return pd.concat(frames, ignore_index=True)


def run(input_csv=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_csv = input_csv or C.LINKED_DATASET_CSV
    df = pd.read_csv(input_csv)

    monthly_kpis = kpi_summary(df, group_col="month")
    monthly_kpis.to_csv(C.KPI_MONTHLY_CSV, index=False)

    bss_by_stage_month = bottleneck_severity_score(df, group_col="month")
    bss_by_stage_month.to_csv(C.BSS_STAGE_CSV, index=False)

    return monthly_kpis, bss_by_stage_month


if __name__ == "__main__":
    monthly_kpis, bss = run()
    print(monthly_kpis)
    print(bss)
