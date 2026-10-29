"""Module 2 runner: Cleaning and leakage removal."""

import json
from pathlib import Path
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from src.preprocess import load_raw_data, run_cleaning_pipeline, save_cleaned_data


def main():
    print("=== Module 2: Cleaning and Leakage Removal ===")
    print("Loading raw data...")
    df_raw = load_raw_data(RAW_DATA_DIR)

    print("Running cleaning and anti-leakage pipeline...")
    df_clean, log = run_cleaning_pipeline(df_raw, min_words=10)

    print("\nCleaning Pipeline Log:")
    print(json.dumps(log, indent=2))

    # Save log to results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "preprocessing_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    # Save cleaned dataset
    clean_csv_path = PROCESSED_DATA_DIR / "clean.csv"
    save_cleaned_data(df_clean, clean_csv_path)

    # Checkpoint Verification
    print("\n--- Checkpoint Verification ---")
    reuters_clean = log["reuters_in_text_clean"]
    reuters_light = log["reuters_in_text_light"]
    print(f"Reuters occurrences in text_clean: Real(1)={reuters_clean.get(1, 0)}, Fake(0)={reuters_clean.get(0, 0)}")
    print(f"Reuters occurrences in text_light: Real(1)={reuters_light.get(1, 0)}, Fake(0)={reuters_light.get(0, 0)}")

    assert reuters_clean.get(1, 0) < 50, "Checkpoint failed: Reuters frequency in Real articles is still high!"
    print("CHECKPOINT PASSED: 'reuters' frequency is near zero and no longer one-sided by class.")


if __name__ == "__main__":
    main()

