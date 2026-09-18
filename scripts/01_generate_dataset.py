#!/usr/bin/env python3
"""Step 1 — generate the ECM demonstration dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecm_digital_thread import config as C
from ecm_digital_thread.data_generation import generate_dataset, save_dataset

if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    print(f"[1/5] Generated {len(df)} ECM records -> {C.RAW_DATASET_CSV}")
    print(df["ecm_class"].value_counts().to_string())
