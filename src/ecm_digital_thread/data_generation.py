"""
Synthetic ECM dataset generator: produces the 1,200-record demonstration
dataset used throughout this toolkit, dated across a 6-month window
(``config.DATASET_START_YEAR`` / ``config.DATASET_START_MONTH``, currently
January-June 2025) for a fictional heavy-machinery manufacturer:

    * structured fields  — change order id, timestamps, workflow stage,
      approval outcome, BOM/linkage flags, source system
    * unstructured text  — maintenance logs / NCRs / supplier emails /
      field-service notes / engineering comments, embedding part numbers,
      assembly IDs, subsystems and responsible teams (ground truth for the
      NER module)
    * 5-class label       — Safety-Critical, Performance-Related,
      Regulatory Compliance, Cost-Driven, Customer-Requested

Month-over-month distribution parameters (mean cycle time, approval
probabilities, linkage rate, documentation-error rate) are interpolated
between the month-1 and month-6 operating points of the illustrative
scenario in ``config.DEMO_SCENARIO``. The trend is therefore an input of the
scenario; the pipeline's job is to measure it back from the individual
records, which are drawn stochastically from those distributions with a
fixed random seed.
"""
from __future__ import annotations

import calendar
import random
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------- #
# Text templates for realistic unstructured ECM narratives
# --------------------------------------------------------------------------- #

SUBSYSTEMS = [
    "hydraulic pump", "engine control module", "final drive assembly",
    "boom cylinder", "cooling fan clutch", "fuel injection system",
    "track undercarriage", "cab HVAC unit", "transmission control unit",
    "exhaust aftertreatment system", "steering valve block", "alternator",
]

TEAMS = [
    "Powertrain Engineering", "Hydraulics Engineering", "Quality Assurance",
    "Supplier Quality", "Field Service", "Manufacturing Engineering",
    "Regulatory Affairs", "Structural Engineering", "Electrical Systems",
]

# Placeholder supplier names (fictional).
SUPPLIERS = [
    "Supplier A", "Supplier B", "Supplier C", "Supplier D", "Supplier E",
]

TEXT_TEMPLATES = {
    "Safety-Critical": [
        "Field service report flags a potential failure mode in the {subsystem} "
        "(assembly {assembly}, part {part}) that could compromise operator safety "
        "under high-load conditions; {team} has requested an urgent design review.",
        "Non-conformance report: {part} on the {subsystem} exhibited a crack "
        "during fatigue testing. {team} recommends immediate containment and a "
        "safety-critical engineering change to assembly {assembly}.",
        "Maintenance log entry: repeated overheating observed on {subsystem} "
        "(part {part}) across three units in the field; escalated to {team} as a "
        "safety-critical change request.",
    ],
    "Performance-Related": [
        "Engineering comment: {subsystem} (part {part}, assembly {assembly}) is "
        "underperforming relative to spec by ~8%; {team} proposes a performance "
        "tuning change to improve efficiency.",
        "Field service report notes reduced output from the {subsystem}; "
        "{team} traces the root cause to part {part} tolerance drift and requests "
        "a performance-related revision to assembly {assembly}.",
        "Test bench data shows the {subsystem} missing target cycle times; "
        "{team} is submitting a change request against part {part} to restore "
        "rated performance.",
    ],
    "Regulatory Compliance": [
        "Regulatory bulletin requires updated emissions documentation for the "
        "{subsystem} (part {part}); {team} is initiating a compliance-driven "
        "engineering change to assembly {assembly}.",
        "Non-conformance report: {part} on the {subsystem} does not meet the "
        "revised safety standard; {team} has opened a regulatory compliance "
        "change request for assembly {assembly}.",
        "Supplier communication from {supplier} indicates a material "
        "certification gap on {part}; {team} requests a compliance change to "
        "the {subsystem}.",
    ],
    "Cost-Driven": [
        "Supplier communication from {supplier} proposes an alternate source "
        "for {part} on the {subsystem} at a 12% cost reduction; {team} is "
        "evaluating a cost-driven engineering change to assembly {assembly}.",
        "Engineering comment: value-engineering review of the {subsystem} "
        "identifies part {part} as a candidate for redesign to reduce unit "
        "cost; {team} to submit change request.",
        "Procurement flags rising cost of {part} used in the {subsystem}; "
        "{team} requests a cost-driven change to qualify a second source for "
        "assembly {assembly}.",
    ],
    "Customer-Requested": [
        "Customer feedback log: fleet operator requests an enhancement to the "
        "{subsystem} (part {part}) for improved serviceability; {team} is "
        "drafting a customer-requested change to assembly {assembly}.",
        "Field service report: customer has asked for a configuration change "
        "to the {subsystem} to support a new attachment; {team} opens a "
        "customer-requested engineering change against part {part}.",
        "Supplier communication forwards a customer complaint about {part} "
        "wear rate on the {subsystem}; {team} is preparing a customer-driven "
        "revision to assembly {assembly}.",
    ],
}

STATUS_PREFIXES = [
    "Maintenance Log #{n}", "Non-Conformance Report NCR-{n}",
    "Field Service Report FSR-{n}", "Supplier Communication SC-{n}",
    "Engineering Comment EC-{n}",
]

# Generic, class-agnostic boilerplate that real ECM narratives are full of —
# injected so classes aren't trivially separable by boilerplate alone.
GENERIC_NOISE_SENTENCES = [
    "This item is currently pending review by the change control board.",
    "Additional supporting documentation has been attached to the ECN record.",
    "Cross-functional stakeholders have been notified for input.",
    "The change request has been logged in the PLM system for tracking.",
    "A follow-up review is scheduled for the next engineering change meeting.",
    "Impact assessment on related assemblies is still in progress.",
]

# Change-request text often touches more than one category — a supplier cost
# swap can carry performance implications, a safety fix can also be
# regulatory. This map lists, for each class, the classes whose vocabulary
# it plausibly overlaps with; a sentence from one of them is sometimes
# blended in so the classes are not trivially separable.
CONFUSION_MAP = {
    "Safety-Critical": ["Regulatory Compliance"],
    "Regulatory Compliance": ["Safety-Critical"],
    "Performance-Related": ["Cost-Driven", "Customer-Requested"],
    "Cost-Driven": ["Performance-Related", "Customer-Requested"],
    "Customer-Requested": ["Performance-Related", "Cost-Driven"],
}

# Probability that a second, overlapping-class sentence is blended into a
# narrative. The same value is used for every class, so per-class
# differences in the classification report come only from which classes
# share vocabulary in CONFUSION_MAP.
CONFUSION_PROB_ALL_CLASSES = 0.25
CONFUSION_PROB = {c: CONFUSION_PROB_ALL_CLASSES for c in CONFUSION_MAP}


def _scenario(metric: str, point: str, scale: float = 1.0) -> float:
    return C.DEMO_SCENARIO[metric][point] / scale


@dataclass
class GenerationParams:
    """Month-1 ('before') and month-6 ('after') operating points of the
    demonstration scenario, linearly interpolated across the 6-month window."""
    mean_cct_before: float = _scenario("Change Cycle Time (days)", "month_1")
    mean_cct_after: float = _scenario("Change Cycle Time (days)", "month_6")
    fpar_before: float = _scenario("First-Pass Approval Rate (%)", "month_1", 100)
    fpar_after: float = _scenario("First-Pass Approval Rate (%)", "month_6", 100)
    pending_before: float = _scenario("Approval Pending Rate (%)", "month_1", 100)
    pending_after: float = _scenario("Approval Pending Rate (%)", "month_6", 100)
    tci_before: float = _scenario("Traceability Coverage Index (%)", "month_1", 100)
    tci_after: float = _scenario("Traceability Coverage Index (%)", "month_6", 100)
    doc_err_before: float = _scenario("Documentation Errors (%)", "month_1", 100)
    doc_err_after: float = _scenario("Documentation Errors (%)", "month_6", 100)

    # Stage-specific congestion multipliers on cycle time / queue depth in
    # the demonstration scenario: uneven load across stages in month 1,
    # closer to even in month 6.
    stage_multiplier_before: dict = field(default_factory=lambda: {
        "Design Review": 1.20, "Quality Approval": 1.10,
        "Manufacturing": 0.95, "Final Release": 0.80,
    })
    stage_multiplier_after: dict = field(default_factory=lambda: {
        "Design Review": 1.05, "Quality Approval": 1.00,
        "Manufacturing": 1.00, "Final Release": 0.95,
    })


def _interp(before: float, after: float, month: int, n_months: int = C.N_MONTHS) -> float:
    """Linear interpolation across months 1..n_months (month 1 == 'before')."""
    t = (month - 1) / (n_months - 1)
    return before + t * (after - before)


def _draw_error_rate(base_rate: float, rng: np.random.Generator) -> float:
    """Record-level documentation-error probability: the month's base rate
    plus small Gaussian jitter (same for every class)."""
    return float(np.clip(base_rate + rng.normal(0, 0.01), 0.0, 0.95))


def generate_dataset(n_records: int = C.N_RECORDS, seed: int = C.RANDOM_SEED,
                      params: GenerationParams | None = None) -> pd.DataFrame:
    """Generate the ECM demonstration dataset as a pandas DataFrame."""
    params = params or GenerationParams()
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    classes = list(C.ECM_CLASS_PRIOR.keys())
    class_p = np.array([C.ECM_CLASS_PRIOR[c] for c in classes])
    class_p = class_p / class_p.sum()

    records = []
    per_month = n_records // C.N_MONTHS
    counts = [per_month] * C.N_MONTHS
    counts[-1] += n_records - sum(counts)  # remainder into last month

    row_id = 1
    for month, n_this_month in enumerate(counts, start=1):
        mean_cct = _interp(params.mean_cct_before, params.mean_cct_after, month)
        fpar = _interp(params.fpar_before, params.fpar_after, month)
        pending = _interp(params.pending_before, params.pending_after, month)
        tci = _interp(params.tci_before, params.tci_after, month)
        doc_err = _interp(params.doc_err_before, params.doc_err_after, month)

        cal_year, cal_month = C.month_index_to_year_month(month)
        days_in_month = calendar.monthrange(cal_year, cal_month)[1]

        ecm_classes = rng.choice(classes, size=n_this_month, p=class_p)
        stages = rng.choice(C.LIFECYCLE_STAGES, size=n_this_month)

        for i in range(n_this_month):
            ecm_class = ecm_classes[i]
            stage = stages[i]

            stage_mult_before = params.stage_multiplier_before[stage]
            stage_mult_after = params.stage_multiplier_after[stage]
            stage_mult = _interp(stage_mult_before, stage_mult_after, month)

            cct = max(1.0, rng.gamma(shape=6.0, scale=(mean_cct * stage_mult) / 6.0))
            queue_depth = max(0, int(rng.poisson(lam=max(0.5, stage_mult * 4))))

            fpar_this = np.clip(fpar + rng.normal(0, 0.03), 0.02, 0.99)
            first_pass_approval = rng.random() < fpar_this

            pending_this = np.clip(pending + rng.normal(0, 0.03), 0.01, 0.9)
            approval_pending = rng.random() < pending_this

            # Draw "fully linked across all 4 lifecycle phases" directly at
            # probability tci_this so the aggregate Traceability Coverage
            # Index (Eq. 1, strict all-phases definition) actually tracks
            # the interpolated before->after trajectory. Partially-linked
            # records (the common real-world case pre-Digital-Thread) get
            # independent, lower-probability per-phase flags instead of
            # compounding tci_this four times (which would crash toward 0).
            tci_this = float(np.clip(tci + rng.normal(0, 0.03), 0.05, 1.0))
            is_fully_linked = rng.random() < tci_this
            if is_fully_linked:
                linked_design = linked_manufacturing = linked_quality = linked_maintenance = True
            else:
                linked_design = rng.random() < 0.55
                linked_manufacturing = rng.random() < 0.50
                linked_quality = rng.random() < 0.45
                linked_maintenance = rng.random() < 0.35

            doc_err_this = _draw_error_rate(doc_err, rng)
            documentation_error = rng.random() < doc_err_this

            subsystem = py_rng.choice(SUBSYSTEMS)
            team = py_rng.choice(TEAMS)
            supplier = py_rng.choice(SUPPLIERS)
            part_number = f"PN-{py_rng.randint(10000, 99999)}"
            assembly_id = f"ASM-{py_rng.randint(1000, 9999)}"

            template = py_rng.choice(TEXT_TEMPLATES[ecm_class])
            sentence = template.format(
                subsystem=subsystem, part=part_number, assembly=assembly_id,
                team=team, supplier=supplier,
            )
            sentence_parts = [sentence]

            # Blend in a confusable-class sentence some of the time so the
            # label isn't trivially recoverable from template vocabulary
            # alone (see CONFUSION_MAP / CONFUSION_PROB above).
            if py_rng.random() < CONFUSION_PROB[ecm_class]:
                other_class = py_rng.choice(CONFUSION_MAP[ecm_class])
                other_template = py_rng.choice(TEXT_TEMPLATES[other_class])
                other_sentence = other_template.format(
                    subsystem=py_rng.choice(SUBSYSTEMS), part=part_number,
                    assembly=assembly_id, team=py_rng.choice(TEAMS),
                    supplier=py_rng.choice(SUPPLIERS),
                )
                sentence_parts.append(other_sentence)

            # Generic boilerplate, present across all classes.
            if py_rng.random() < 0.5:
                sentence_parts.append(py_rng.choice(GENERIC_NOISE_SENTENCES))

            raw_text = " ".join(sentence_parts)
            prefix = py_rng.choice(STATUS_PREFIXES).format(n=f"{row_id:05d}")
            raw_text = f"{prefix}: {raw_text}"

            request_date = date(cal_year, cal_month, py_rng.randint(1, days_in_month))

            records.append({
                "change_order_id": f"ECN-{row_id:06d}",
                "request_date": request_date.isoformat(),
                "month": month,
                "month_label": C.MONTH_LABELS[month - 1],
                "source_system": py_rng.choice(C.SOURCE_SYSTEMS),
                "ecm_class": ecm_class,
                "workflow_stage": stage,
                "cycle_time_days": round(float(cct), 2),
                "queue_depth": queue_depth,
                "first_pass_approval": bool(first_pass_approval),
                "approval_pending": bool(approval_pending),
                "documentation_error": bool(documentation_error),
                "linked_design": bool(linked_design),
                "linked_manufacturing": bool(linked_manufacturing),
                "linked_quality": bool(linked_quality),
                "linked_maintenance": bool(linked_maintenance),
                "part_number": part_number,
                "assembly_id": assembly_id,
                "responsible_team": team,
                "affected_subsystem": subsystem,
                "raw_text": raw_text,
            })
            row_id += 1

    df = pd.DataFrame.from_records(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    return df


def save_dataset(df: pd.DataFrame, path=None) -> None:
    path = path or C.RAW_DATASET_CSV
    df.to_csv(path, index=False)


if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset)
    print(f"Generated {len(dataset)} records -> {C.RAW_DATASET_CSV}")
    print(dataset["ecm_class"].value_counts())
