"""Module 3 runner: Stratified Train/Val/Test Split and Overlap Audit."""

import json
from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, SEED
from src.preprocess import create_stratified_splits, save_splits


def main():
    print("=== Module 3: Train / Validation / Test Stratified Split ===")
    clean_path = PROCESSED_DATA_DIR / "clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {clean_path}. Run Module 2 first.")

    print(f"Loading cleaned dataset from {clean_path}...")
    df_clean = pd.read_csv(clean_path)
    print(f"Loaded {len(df_clean):,} records.")

    print(f"Generating stratified 70/15/15 splits with global SEED={SEED}...")
    df_train, df_val, df_test, summary = create_stratified_splits(
        df_clean,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=SEED,
    )

    print("\nSplit Summary:")
    print(json.dumps(summary, indent=2))

    # Save summary to results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save CSV files
    save_splits(df_train, df_val, df_test, PROCESSED_DATA_DIR)

    # Checkpoint Verification
    print("\n--- Checkpoint Verification ---")
    train_ratios = summary["class_balance"]["train"]
    val_ratios = summary["class_balance"]["val"]
    test_ratios = summary["class_balance"]["test"]
    overlaps = summary["text_overlaps"]

    print(f"Class ratio Train: Real={train_ratios.get('1', train_ratios.get(1, 0)):.4f}, Fake={train_ratios.get('0', train_ratios.get(0, 0)):.4f}")
    print(f"Class ratio Val:   Real={val_ratios.get('1', val_ratios.get(1, 0)):.4f}, Fake={val_ratios.get('0', val_ratios.get(0, 0)):.4f}")
    print(f"Class ratio Test:  Real={test_ratios.get('1', test_ratios.get(1, 0)):.4f}, Fake={test_ratios.get('0', test_ratios.get(0, 0)):.4f}")
    print(f"Text overlaps: Train-Val={overlaps['train_val']}, Train-Test={overlaps['train_test']}, Val-Test={overlaps['val_test']}")

    assert overlaps["train_val"] == 0, "Leakage detected: overlap between train and val!"
    assert overlaps["train_test"] == 0, "Leakage detected: overlap between train and test!"
    assert overlaps["val_test"] == 0, "Leakage detected: overlap between val and test!"

    print("CHECKPOINT PASSED: Class ratios are strictly preserved across splits and text overlap is exactly ZERO.")


if __name__ == "__main__":
    main()

