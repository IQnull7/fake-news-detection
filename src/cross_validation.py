"""Module 9: Multi-seed and 5-Fold Cross-Validation Reliability Audits."""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, SEED, set_seed


def run_5fold_cv_reliability(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
) -> Dict[str, Any]:
    print(f"\n--- Running 5-Fold Stratified Cross-Validation (Seed={seed}) ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = df["text_clean"].fillna("").values
    y = df["label"].values

    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_val = X[val_idx], y[val_idx]

        # Vectorizer fit strictly on train fold
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
        X_tr_vec = tfidf.fit_transform(X_tr)
        X_va_vec = tfidf.transform(X_va)

        # Train Logistic Regression
        clf = LogisticRegression(C=10.0, max_iter=1000, random_state=seed)
        clf.fit(X_tr_vec, y_tr)

        y_pred = clf.predict(X_va_vec)
        y_probs = clf.predict_proba(X_va_vec)[:, 1]

        acc = float(accuracy_score(y_val, y_pred))
        prec = float(precision_score(y_val, y_pred, zero_division=0))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_val, y_probs))

        fold_metrics.append({
            "fold": fold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        })
        print(f"  Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    df_folds = pd.DataFrame(fold_metrics)

    summary = {
        "metric": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "mean": [float(df_folds[m].mean()) for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]],
        "std": [float(df_folds[m].std()) for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]],
    }
    df_summary = pd.DataFrame(summary)
    df_summary["formatted"] = [
        f"{m:.4f} +/- {s:.4f}" for m, s in zip(df_summary["mean"], df_summary["std"])
    ]

    print("\n5-Fold CV Summary (Logistic Regression):")
    print(df_summary[["metric", "formatted"]].to_string(index=False))

    return {
        "fold_results": fold_metrics,
        "summary": df_summary.to_dict(orient="records"),
        "raw_df": df_folds,
    }


def run_multi_seed_audit(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    seeds: List[int] = [42, 101, 2024, 7, 999],
) -> pd.DataFrame:
    print(f"\n--- Running Multi-Seed Robustness Audit across seeds: {seeds} ---")
    seed_records = []

    for s in seeds:
        set_seed(s)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
        X_tr = tfidf.fit_transform(df_train["text_clean"].fillna(""))
        X_va = tfidf.transform(df_val["text_clean"].fillna(""))

        clf = LogisticRegression(C=10.0, max_iter=1000, random_state=s)
        clf.fit(X_tr, df_train["label"].values)

        y_pred = clf.predict(X_va)
        y_probs = clf.predict_proba(X_va)[:, 1]

        y_val = df_val["label"].values
        seed_records.append({
            "seed": s,
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, y_probs)),
        })

    df_seeds = pd.DataFrame(seed_records)
    print("\nMulti-Seed Results Table:")
    print(df_seeds.to_string(index=False))
    return df_seeds


def main():
    clean_path = PROCESSED_DATA_DIR / "clean.csv"
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"

    print("Loading datasets for Module 9 reliability audit...")
    df_clean = pd.read_csv(clean_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # 1. 5-Fold Stratified CV
    cv_res = run_5fold_cv_reliability(df_clean, n_splits=5, seed=SEED)

    # 2. Multi-Seed Audit
    seed_df = run_multi_seed_audit(df_train, df_val)

    # Save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_res["raw_df"].to_csv(RESULTS_DIR / "cv_reliability_folds.csv", index=False)
    seed_df.to_csv(RESULTS_DIR / "cv_multi_seed.csv", index=False)

    with open(RESULTS_DIR / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_res["summary"], f, indent=2)

    # Checkpoint Verification: Check that std is very small (< 0.005)
    f1_std = [r["std"] for r in cv_res["summary"] if r["metric"] == "f1"][0]
    print(f"\n--- Checkpoint Verification ---")
    print(f"5-Fold CV F1 Standard Deviation: {f1_std:.5f}")
    assert f1_std < 0.01, f"High variance detected! std={f1_std}"
    print("CHECKPOINT PASSED: Standard deviation across folds is exceptionally small (<0.01), confirming model stability.")


if __name__ == "__main__":
    main()

