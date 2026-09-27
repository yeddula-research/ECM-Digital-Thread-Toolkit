#!/usr/bin/env python3
"""Step 3 — NLP change-request classification + NER (Section III.C/III.G).

Usage:
    python scripts/03_train_nlp_classifier.py                # sklearn backend (default)
    python scripts/03_train_nlp_classifier.py --backend transformer   # Sec. III.G fine-tuning procedure
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecm_digital_thread import config as C
from ecm_digital_thread.nlp_pipeline import run, evaluate_ner
from ecm_digital_thread.data_generation import SUBSYSTEMS, TEAMS
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "transformer"], default="sklearn")
    args = parser.parse_args()

    df, result = run(backend=args.backend)
    print(f"[3/6] NLP-scored dataset written -> {C.NLP_SCORED_CSV}")

    report_dict = result.report_dict if hasattr(result, "report_dict") else result["report_dict"]
    (C.RESULTS_TABLES_DIR / "table2_nlp_classification_report.json").write_text(
        json.dumps(report_dict, indent=2))

    print("      Classification report (test set):")
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict) and "precision" in metrics:
            print(f"        {label:24s} P={metrics['precision']:.2f}  "
                  f"R={metrics['recall']:.2f}  F1={metrics['f1-score']:.2f}")

    raw_df = pd.read_csv(C.RAW_DATASET_CSV)
    ner_accuracy = evaluate_ner(raw_df, TEAMS, SUBSYSTEMS)
    (C.RESULTS_TABLES_DIR / "ner_extraction_accuracy.json").write_text(
        json.dumps(ner_accuracy, indent=2))
    print(f"      NER exact-match extraction accuracy: {ner_accuracy}")
