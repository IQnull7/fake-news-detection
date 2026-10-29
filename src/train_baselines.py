"""Module 4: Sanity baseline models to audit shortcut and leakage potential."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED, set_seed
from src.evaluate import evaluate


def run_sanity_baselines() -> pd.DataFrame:
    """Train and evaluate 3 sanity baselines:
    1. Majority-class predictor
    2. Title-only TF-IDF + Logistic Regression
    3. First-5-words TF-IDF + Logistic Regression
    """
    set_seed(SEED)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("train.csv or val.csv missing in data/processed/. Run Module 3 first.")

    print(f"Loading train ({train_path}) and val ({val_path})...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    y_train = df_train["label"].values
    y_val = df_val["label"].values

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_results = []

    # -------------------------------------------------------------
    # 1. Majority-Class Predictor
    # -------------------------------------------------------------
    print("\n--- 1. Majority-Class Baseline ---")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(df_train["text_clean"], y_train)
    y_pred_dummy = dummy.predict(df_val["text_clean"])
    y_score_dummy = dummy.predict_proba(df_val["text_clean"])

    res_dummy = evaluate(
        y_true=y_val,
        y_pred=y_pred_dummy,
        y_score=y_score_dummy,
        model_name="Majority baseline",
        split_name="val",
    )
    baseline_results.append(res_dummy)
    joblib.dump(dummy, MODELS_DIR / "baseline_majority.joblib")
    print("Majority baseline metrics:", res_dummy)

    # -------------------------------------------------------------
    # 2. Title-Only TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("\n--- 2. Title-Only TF-IDF + Logistic Regression ---")
    title_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ("clf", LogisticRegression(random_state=SEED, max_iter=1000, C=1.0)),
    ])
    title_pipe.fit(df_train["title_clean"].fillna(""), y_train)
    y_pred_title = title_pipe.predict(df_val["title_clean"].fillna(""))
    y_score_title = title_pipe.predict_proba(df_val["title_clean"].fillna(""))

    res_title = evaluate(
        y_true=y_val,
        y_pred=y_pred_title,
        y_score=y_score_title,
        model_name="Title-only LR",
        split_name="val",
    )
    baseline_results.append(res_title)
    joblib.dump(title_pipe, MODELS_DIR / "baseline_title_lr.joblib")
    print("Title-only LR metrics:", res_title)

    # -------------------------------------------------------------
    # 3. First-5-Words TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("\n--- 3. First-5-Words TF-IDF + Logistic Regression ---")
    first5_train = df_train["text_clean"].fillna("").apply(lambda t: " ".join(str(t).split()[:5]))
    first5_val = df_val["text_clean"].fillna("").apply(lambda t: " ".join(str(t).split()[:5]))

    first5_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(random_state=SEED, max_iter=1000, C=1.0)),
    ])
    first5_pipe.fit(first5_train, y_train)
    y_pred_first5 = first5_pipe.predict(first5_val)
    y_score_first5 = first5_pipe.predict_proba(first5_val)

    res_first5 = evaluate(
        y_true=y_val,
        y_pred=y_pred_first5,
        y_score=y_score_first5,
        model_name="First-5-words LR",
        split_name="val",
    )
    baseline_results.append(res_first5)
    joblib.dump(first5_pipe, MODELS_DIR / "baseline_first5_lr.joblib")
    print("First-5-words LR metrics:", res_first5)

    df_summary = pd.DataFrame(baseline_results)
    print("\n=== Summary Table: Sanity Baselines (Validation) ===")
    print(df_summary.to_string(index=False))

    # Save to dedicated CSV
    df_summary.to_csv(RESULTS_DIR / "sanity_baselines.csv", index=False)

    # Checkpoint evaluation:
    # If first-5-words scored ~99%, wire leakage would still exist.
    # We verify that first-5-words does NOT score ~99%.
    print("\n--- Checkpoint Audit ---")
    maj_acc = res_dummy["accuracy"]
    first5_acc = res_first5["accuracy"]
    print(f"Majority Baseline Accuracy:   {maj_acc:.4f}")
    print(f"First-5-Words Model Accuracy: {first5_acc:.4f}")

    if first5_acc > 0.90:
        print("WARNING: First-5-words model scored >90%. Potential residual header leakage!")
    else:
        print("CHECKPOINT PASSED: First-5-words model did not exploit trivial source headers (does not hit >90%).")

    return df_summary


if __name__ == "__main__":
    run_sanity_baselines()

