# Methodology

This document maps each part of the system architecture in the paper's
Figure 1 to the module, function, or script that implements it, and records
the settings of the synthetic demonstration dataset.

## System Architecture (Fig. 1, five layers)

| Layer | Implementation |
|---|---|
| Layer 1 — Data Source Layer (PLM/ERP/CAD/MMS/Supplier/ECN) | `data_generation.py` produces synthetic records tagged with a `source_system` field drawn from `config.SOURCE_SYSTEMS` |
| Layer 2 — Digital Thread Integration Layer (UDM, bidirectional traceability, TCI) | `digital_thread.py` |
| Layer 3 — NLP Processing Layer (preprocessing, TF-IDF, classification, NER) | `nlp_pipeline.py` |
| Layer 4 — Analytics & Visualization (KPI dashboards, BSS, heatmaps, trends) | `kpi_analytics.py` (measures) + `dashboard.py` (analytics dashboard) |
| Layer 5 — Decision & Feedback Layer | Monthly KPI and BSS outputs (`kpi_analytics.run`) are the inputs a change control board would act on; the incremental-learning loop itself is not simulated |

## Traceability Coverage Index (Eq. 1)

```
TCI = (N_linked / N_total) * 100
```

Implemented in `digital_thread.traceability_coverage_index()`. A record is
"linked" only if **all four** lifecycle-phase flags (`linked_design`,
`linked_manufacturing`, `linked_quality`, `linked_maintenance`) are true —
the strict reading of full cross-lifecycle traceability. See
`digital_thread.compute_full_linkage()`.

## NLP-Based Change Request Processing

* **Preprocessing** (tokenize, remove stop-words, lemmatize) →
  `nlp_pipeline.preprocess_text()`. Implemented without NLTK/spaCy so the
  default pipeline has zero extra installs; the module docstring notes the
  rule-based lemmatizer's limitations and how to swap in a full lemmatizer.
* **TF-IDF feature extraction (Eq. 2)**:

  ```
  TF-IDF(t, d, D) = TF(t, d) * log(|D| / |{d ∈ D : t ∈ d}|)
  ```

  Implemented via `sklearn.feature_extraction.text.TfidfVectorizer` inside
  `nlp_pipeline.train_classifier()`.
* **Multi-class classification into the 5 ECM domains** (Safety-Critical,
  Performance-Related, Regulatory Compliance, Cost-Driven,
  Customer-Requested) → `nlp_pipeline.train_classifier()`, default backend
  `LinearSVC` over TF-IDF features. The paper's classifier is a fine-tuned
  BERT model; that procedure is available as `backend="transformer"` (see
  "BERT Fine-Tuning Procedure" below).
* **NER for structured metadata** (part numbers, assembly IDs, responsible
  teams, subsystems) → `nlp_pipeline.extract_entities()`, rule-based (regex
  + controlled vocabulary) by default; a spaCy statistical-NER alternative
  is noted in the same function's docstring. On the demonstration dataset,
  whose text is built from templates, the rule-based extractor recovers every
  entity; real free text would score lower.

## Power BI-Style Analytics and Bottleneck Detection

* **KPIs** (Change Cycle Time, Approval Pending Rate, First-Pass Approval
  Rate, Change Propagation Index, Documentation Error Rate) →
  `kpi_analytics.py`, one function per measure, each a pandas equivalent of
  a Power BI DAX measure.
* **Bottleneck Severity Score (Eq. 3)**:

  ```
  BSS_i = alpha * (CCT_i_bar / CCT_baseline_bar)
        + beta  * (Q_i / Q_avg)
        + gamma * (1 - FPAR_i)          s.t. alpha + beta + gamma = 1
  ```

  Implemented in `kpi_analytics.bottleneck_severity_score()`, computed per
  workflow stage (and optionally per month). The paper leaves the weights
  and the flag threshold to be calibrated to organizational priorities; the
  toolkit's defaults, `alpha=0.5, beta=0.3, gamma=0.2` and `BSS > 1.0`, are
  set in `config.py` and configurable. The baseline cycle time
  `config.CCT_BASELINE_DAYS` is the demonstration scenario's month-1 mean.

## Demonstration dataset

The demonstration dataset is synthetic: 1,200 engineering change records for a fictional heavy-machinery manufacturer, generated with a fixed random seed by scripts/01_generate_dataset.py from the illustrative scenario in config.DEMO_SCENARIO; it is not the paper's case-study data, and its before/after trend is an input of the scenario rather than a measured effect.

**Records.** Structured fields (change order id, a calendar `request_date`
within January–June 2025 by default, a relative `month` index, workflow
stage, cycle time, queue depth, approval outcomes, documentation-error and
lifecycle-linkage flags, part/assembly/team/subsystem metadata, source
system) and one unstructured narrative per record, built from class-specific
templates (maintenance logs, non-conformance reports, supplier
communications, field service reports, engineering comments). Supplier names
are placeholders.

**Scenario.** `config.DEMO_SCENARIO` sets the month-1 and month-6 operating
points below; `data_generation.GenerationParams` interpolates linearly
between them, and each record's outcomes are drawn around the month's value
with a fixed seed (`config.RANDOM_SEED = 42`). The values are round,
illustrative choices for this toolkit.

| Metric | Month 1 | Month 6 |
|---|---|---|
| Change Cycle Time (days) | 20.0 | 14.0 |
| Approval Pending Rate (%) | 30 | 20 |
| First-Pass Approval Rate (%) | 60 | 72 |
| Traceability Coverage Index (%) | 55 | 80 |
| Documentation Errors (%) | 15 | 10 |

Other scenario settings: equal class shares (20% each,
`config.ECM_CLASS_PRIOR`); stage congestion multipliers of 1.20 / 1.10 /
0.95 / 0.80 (Design Review / Quality Approval / Manufacturing / Final
Release) in month 1, moving to 1.05 / 1.00 / 1.00 / 0.95 by month 6; and a
single probability, 0.25 for every class, of blending in a sentence from an
overlapping class (`data_generation.CONFUSION_MAP`) so the classes are not
trivially separable.

## The paper's published results

`ecm_digital_thread.published` records the paper's Tables 1–4 as printed,
and `scripts/06_published_values.py` writes them to
`results/tables/*_as_reported.*`. Nothing in the demonstration pipeline reads
these values. The script also checks the published tables against
themselves:

* each Table 1 improvement percentage against its before/after values;
* each Table 2 F1 score against the harmonic mean of its precision and
  recall, and the "Overall Average" row against the mean of the five classes;
* the "Proposed Framework" row of Table 4 against the corresponding Table 1
  values.

The paper's Figures 2–5 are not included: apart from the start and end
values quoted in its text, their data are given graphically.

## BERT Fine-Tuning Procedure

70:15:15 train/val/test split, WordPiece embeddings, max sequence length
256, AdamW optimizer, lr=2e-5, batch size 16, 5 epochs, cross-entropy loss,
early stopping + dropout regularization. Implemented in
`nlp_pipeline._train_transformer()`, invoked via
`train_classifier(df, backend="transformer")` or
`python scripts/03_train_nlp_classifier.py --backend transformer`. Requires
`pip install torch transformers` (not a default dependency — see README).

## Output files

| Output | Generated by |
|---|---|
| Demonstration KPIs, month 1 vs month 6 | `scripts/05_generate_figures_tables.py` → `results/tables/table1_ecm_performance_before_after.{csv,md}` |
| Demonstration NLP classification report | `scripts/03_train_nlp_classifier.py` → `results/tables/table2_nlp_classification_report.{csv,md,json}` |
| Demonstration figures (cycle time, bottleneck severity by stage, first-pass approval, traceability coverage) | `scripts/05_generate_figures_tables.py` → `results/figures/*.png` |
| Analytics dashboard + SQLite export | `scripts/05_generate_figures_tables.py` → `results/dashboard/` |
| The paper's Tables 1–4 and consistency checks | `scripts/06_published_values.py` → `results/tables/*_as_reported.*`, `results/tables/published_consistency_checks.json` |
