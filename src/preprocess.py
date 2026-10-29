"""Data loading, cleaning, and preprocessing functions for Fake News Detection."""

from pathlib import Path
from typing import Dict, Any, Tuple
import re
import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LABEL_REAL, LABEL_FAKE, SEED


def load_raw_data(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load True.csv and Fake.csv, assign labels, and concatenate.

    Returns:
        pd.DataFrame: Combined DataFrame with columns ['title', 'text', 'subject', 'date', 'label']
    """
    true_path = data_dir / "True.csv"
    fake_path = data_dir / "Fake.csv"

    if not true_path.exists() or not fake_path.exists():
        raise FileNotFoundError(
            f"Expected {true_path} and {fake_path} to exist in data/raw."
        )

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = LABEL_REAL
    df_fake["label"] = LABEL_FAKE

    df_combined = pd.concat([df_true, df_fake], ignore_index=True)
    return df_combined


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics, class balance, and text lengths for raw data inspection."""
    char_lens = df["text"].fillna("").str.len()
    word_lens = df["text"].fillna("").apply(lambda t: len(str(t).split()))
    empty_count = int((df["text"].fillna("").str.strip() == "").sum())

    summary = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "null_counts": df.isnull().sum().to_dict(),
        "class_counts": df["label"].value_counts().to_dict(),
        "class_proportions": df["label"].value_counts(normalize=True).to_dict(),
        "empty_text_count": empty_count,
        "word_count_stats": {
            "overall_mean": float(word_lens.mean()),
            "overall_median": float(word_lens.median()),
            "overall_std": float(word_lens.std()),
            "overall_min": int(word_lens.min()),
            "overall_max": int(word_lens.max()),
            "real_mean": float(word_lens[df["label"] == LABEL_REAL].mean()),
            "real_median": float(word_lens[df["label"] == LABEL_REAL].median()),
            "fake_mean": float(word_lens[df["label"] == LABEL_FAKE].mean()),
            "fake_median": float(word_lens[df["label"] == LABEL_FAKE].median()),
        },
    }
    return summary
