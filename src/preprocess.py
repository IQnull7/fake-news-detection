"""Data loading, cleaning, leakage removal, splitting, and preprocessing functions."""

from pathlib import Path
from typing import Dict, Any, Tuple
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


# Regex patterns for cleaning & leakage removal
_LEADING_AGENCY_PATTERN = re.compile(
    r"^[\s\w\.,\/\-\'\(\)]*?\((?:Reuters|AP|AFP)\)\s*[-–—]?\s*",
    flags=re.IGNORECASE,
)
_REUTERS_TOKEN_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]*reuters[a-zA-Z0-9.-]*\b",
    flags=re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text_light(text: str) -> str:
    """Clean text lightly for transformer / BERT models (preserves casing & grammar)."""
    if not isinstance(text, str):
        return ""
    t = _LEADING_AGENCY_PATTERN.sub("", text)
    t = _REUTERS_TOKEN_PATTERN.sub("", t)
    t = _URL_PATTERN.sub("", t)
    t = _WHITESPACE_PATTERN.sub(" ", t).strip()
    return t


def clean_text_classical(text: str) -> str:
    """Clean text thoroughly for TF-IDF / classical ML models."""
    t = clean_text_light(text)
    t = t.lower()
    t = _PUNCTUATION_PATTERN.sub(" ", t)
    t = _WHITESPACE_PATTERN.sub(" ", t).strip()
    return t


def run_cleaning_pipeline(
    df: pd.DataFrame,
    min_words: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete Module 2 cleaning and anti-leakage pipeline."""
    log: Dict[str, Any] = {}
    n_initial = len(df)
    log["initial_count"] = n_initial

    # Step 1: Deduplicate on (title, text)
    df_step1 = df.drop_duplicates(subset=["title", "text"]).copy()
    log["duplicates_title_text_removed"] = n_initial - len(df_step1)

    # Step 2: Deduplicate on text
    df_step2 = df_step1.drop_duplicates(subset=["text"]).copy()
    log["duplicates_text_removed"] = len(df_step1) - len(df_step2)

    # Step 3: Filter short/empty texts
    word_counts = df_step2["text"].fillna("").apply(lambda t: len(str(t).split()))
    df_step3 = df_step2[word_counts >= min_words].copy()
    log["short_texts_removed"] = len(df_step2) - len(df_step3)

    # Step 4: Drop 'subject' and 'date' to prevent shortcut leakage
    cols_to_drop = [c for c in ["subject", "date"] if c in df_step3.columns]
    df_clean = df_step3.drop(columns=cols_to_drop).copy()
    log["columns_dropped"] = cols_to_drop

    # Step 5: Clean texts
    df_clean["text_light"] = df_clean["text"].apply(clean_text_light)
    df_clean["text_clean"] = df_clean["text"].apply(clean_text_classical)

    df_clean["title_light"] = df_clean["title"].apply(clean_text_light)
    df_clean["title_clean"] = df_clean["title"].apply(clean_text_classical)

    # Re-verify that after cleaning, texts did not become empty
    valid_mask = (df_clean["text_clean"].str.len() > 0) & (df_clean["text_light"].str.len() > 0)
    empty_after_clean = int((~valid_mask).sum())
    if empty_after_clean > 0:
        df_clean = df_clean[valid_mask].copy()
    log["empty_after_cleaning_removed"] = empty_after_clean

    # Step 5b: Deduplicate on cleaned text (prevents leakage from near-identical raw texts differing only by wire tags)
    pre_dedup_clean = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["text_clean"]).copy()
    df_clean = df_clean.drop_duplicates(subset=["text_light"]).copy()
    log["duplicates_post_cleaning_removed"] = pre_dedup_clean - len(df_clean)

    log["final_count"] = len(df_clean)
    log["final_class_counts"] = df_clean["label"].value_counts().to_dict()
    log["final_class_proportions"] = df_clean["label"].value_counts(normalize=True).to_dict()

    # Step 6: Reuters leakage audit check
    reuters_in_clean = df_clean.groupby("label")["text_clean"].apply(
        lambda s: int(s.str.contains("reuters", case=False).sum())
    ).to_dict()
    reuters_in_light = df_clean.groupby("label")["text_light"].apply(
        lambda s: int(s.str.contains("reuters", case=False).sum())
    ).to_dict()
    log["reuters_in_text_clean"] = reuters_in_clean
    log["reuters_in_text_light"] = reuters_in_light

    return df_clean, log


def save_cleaned_data(
    df: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_DIR / "clean.csv",
) -> None:
    """Save cleaned DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to {output_path} ({len(df):,} rows)")


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create leak-free stratified train, val, and test splits (70/15/15).

    Returns:
        (df_train, df_val, df_test, split_summary)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Step 1: Split into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    # Step 2: Split temp into val and test
    relative_test_ratio = test_ratio / temp_ratio
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_ratio,
        stratify=df_temp["label"],
        random_state=seed,
    )

    # Check zero overlap on text
    train_texts = set(df_train["text_clean"])
    val_texts = set(df_val["text_clean"])
    test_texts = set(df_test["text_clean"])

    overlap_train_val = len(train_texts.intersection(val_texts))
    overlap_train_test = len(train_texts.intersection(test_texts))
    overlap_val_test = len(val_texts.intersection(test_texts))

    summary: Dict[str, Any] = {
        "seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
            "total": len(df),
        },
        "class_balance": {
            "train": df_train["label"].value_counts(normalize=True).to_dict(),
            "val": df_val["label"].value_counts(normalize=True).to_dict(),
            "test": df_test["label"].value_counts(normalize=True).to_dict(),
        },
        "text_overlaps": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test,
        },
    }

    return df_train, df_val, df_test, summary


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Save train.csv, val.csv, and test.csv to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)
    print(f"Splits saved to {output_dir}:")
    print(f"  train.csv: {len(df_train):,} rows")
    print(f"  val.csv:   {len(df_val):,} rows")
    print(f"  test.csv:  {len(df_test):,} rows")
