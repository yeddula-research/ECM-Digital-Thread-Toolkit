#!/usr/bin/env python3
"""Run the full pipeline end-to-end, in order:

    1. generate the ECM dataset
    2. Digital Thread integration + TCI
    3. NLP classification + NER
    4. KPI / Bottleneck Severity Score analytics
    5. Tables 1-4 + Figures 2-5 + dashboard

Usage:
    python scripts/run_all.py [--backend sklearn|transformer]
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

STEPS = [
    "01_generate_dataset.py",
    "02_run_digital_thread.py",
    "03_train_nlp_classifier.py",
    "04_compute_kpis_bottlenecks.py",
    "05_generate_figures_tables.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "transformer"], default="sklearn")
    args = parser.parse_args()

    for step in STEPS:
        cmd = [sys.executable, str(SCRIPTS_DIR / step)]
        if step == "03_train_nlp_classifier.py":
            cmd += ["--backend", args.backend]
        print(f"\n{'=' * 78}\n>>> {step}\n{'=' * 78}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Step {step} failed with exit code {result.returncode}.", file=sys.stderr)
            sys.exit(result.returncode)

    print("\nAll steps completed. See data/, results/tables/, results/figures/, "
          "and results/dashboard/index.html")


if __name__ == "__main__":
    main()
