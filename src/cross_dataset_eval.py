"""Module 10: Cross-Dataset Generalization Evaluation & In-Domain vs Out-of-Domain Benchmark."""

from pathlib import Path
from typing import Dict, Any, List
import urllib.request
import io
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, SEED, set_seed
from src.preprocess import clean_text_classical, clean_text_light
from src.evaluate import evaluate
from src.train_lstm import BiLSTMClassifier, Vocabulary


LIAR_TEST_URL = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv"


def load_or_fetch_liar_dataset(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Download or load the LIAR test benchmark dataset."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    liar_csv_path = raw_dir / "liar_test.csv"

    if liar_csv_path.exists():
        print(f"Loading cached LIAR dataset from {liar_csv_path}...")
        return pd.read_csv(liar_csv_path)

    print(f"Fetching LIAR test dataset from {LIAR_TEST_URL}...")
    req = urllib.request.Request(LIAR_TEST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    cols = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party_affiliation", "barely_true_cnt", "false_cnt",
        "half_true_cnt", "mostly_true_cnt", "pants_on_fire_cnt", "context"
    ]
    df_raw = pd.read_csv(io.BytesIO(data), sep="\t", header=None, names=cols)

    # Standard Binarization:
    # 1 (Real): true, mostly-true, half-true
    # 0 (Fake): false, barely-true, pants-fire
    label_map = {
        "true": 1, "mostly-true": 1, "half-true": 1,
        "barely-true": 0, "false": 0, "pants-fire": 0
    }
    df_raw["binary_label"] = df_raw["label"].map(label_map)
    df_raw["statement_clean"] = df_raw["statement"].apply(clean_text_classical)
    df_raw["statement_light"] = df_raw["statement"].apply(clean_text_light)

    df_raw.to_csv(liar_csv_path, index=False)
    print(f"LIAR test dataset saved to {liar_csv_path} ({len(df_raw)} statements).")
    return df_raw


def evaluate_cross_dataset() -> pd.DataFrame:
    set_seed(SEED)
    df_liar = load_or_fetch_liar_dataset()
    y_liar = df_liar["binary_label"].values

    # In-domain test set
    df_isot_test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    y_isot = df_isot_test["label"].values

    models_to_test = [
        ("Majority baseline", "baseline_majority.joblib", "text_clean", "statement_clean"),
        ("Title-only LR", "baseline_title_lr.joblib", "title_clean", "statement_clean"),
        ("TF-IDF + Naive Bayes", "model_naive_bayes.joblib", "text_clean", "statement_clean"),
        ("TF-IDF + Logistic Regression", "model_logistic_regression.joblib", "text_clean", "statement_clean"),
        ("TF-IDF + Linear SVM", "model_linear_svm.joblib", "text_clean", "statement_clean"),
        ("TF-IDF + Random Forest", "model_random_forest.joblib", "text_clean", "statement_clean"),
        ("TF-IDF + XGBoost", "model_xgboost.joblib", "text_clean", "statement_clean"),
    ]

    comparison_records = []

    print("\n=== Evaluating Models on In-Domain (ISOT) vs Cross-Domain (LIAR) ===")
    for model_name, filename, isot_col, liar_col in models_to_test:
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            print(f"Skipping {model_name} (file not found: {filename})")
            continue

        pipeline = joblib.load(model_path)

        # 1. In-Domain Evaluation
        X_isot = df_isot_test[isot_col].fillna("")
        preds_isot = pipeline.predict(X_isot)
        res_in = evaluate(
            y_true=y_isot,
            y_pred=preds_isot,
            model_name=model_name,
            split_name="test_in_domain",
            save_cm=False,
        )

        # 2. Cross-Domain Evaluation (Zero retraining or tuning!)
        X_liar = df_liar[liar_col].fillna("")
        preds_liar = pipeline.predict(X_liar)
        res_cross = evaluate(
            y_true=y_liar,
            y_pred=preds_liar,
            model_name=model_name,
            split_name="cross_dataset_liar",
            save_cm=False,
        )

        comparison_records.append({
            "Model": model_name,
            "In-Domain Acc": res_in["accuracy"],
            "In-Domain Prec": res_in["precision"],
            "In-Domain Rec": res_in["recall"],
            "In-Domain F1": res_in["f1"],
            "Cross-Dataset Acc": res_cross["accuracy"],
            "Cross-Dataset Prec": res_cross["precision"],
            "Cross-Dataset Rec": res_cross["recall"],
            "Cross-Dataset F1": res_cross["f1"],
            "F1 Drop": round(res_in["f1"] - res_cross["f1"], 4),
        })

    # Evaluate BiLSTM
    bilstm_weights = MODELS_DIR / "bilstm_best.pt"
    vocab_path = MODELS_DIR / "bilstm_vocab.joblib"
    if bilstm_weights.exists() and vocab_path.exists():
        print("Evaluating BiLSTM on In-Domain and Cross-Dataset...")
        vocab: Vocabulary = joblib.load(vocab_path)
        device = torch.device("cpu")
        bilstm = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_layers=2)
        bilstm.load_state_dict(torch.load(bilstm_weights, map_location=device))
        bilstm.eval()

        # In-domain
        isot_t = torch.tensor([vocab.encode(t, 150) for t in df_isot_test["text_light"].fillna("")], dtype=torch.long)
        with torch.no_grad():
            bilstm_isot_preds = torch.argmax(bilstm(isot_t), dim=1).numpy()
        res_lstm_in = evaluate(y_isot, bilstm_isot_preds, model_name="BiLSTM (PyTorch)", split_name="test_in_domain", save_cm=False)

        # Cross-domain
        liar_t = torch.tensor([vocab.encode(t, 150) for t in df_liar["statement_light"].fillna("")], dtype=torch.long)
        with torch.no_grad():
            bilstm_liar_preds = torch.argmax(bilstm(liar_t), dim=1).numpy()
        res_lstm_cross = evaluate(y_liar, bilstm_liar_preds, model_name="BiLSTM (PyTorch)", split_name="cross_dataset_liar", save_cm=False)

        comparison_records.append({
            "Model": "BiLSTM (PyTorch)",
            "In-Domain Acc": res_lstm_in["accuracy"],
            "In-Domain Prec": res_lstm_in["precision"],
            "In-Domain Rec": res_lstm_in["recall"],
            "In-Domain F1": res_lstm_in["f1"],
            "Cross-Dataset Acc": res_lstm_cross["accuracy"],
            "Cross-Dataset Prec": res_lstm_cross["precision"],
            "Cross-Dataset Rec": res_lstm_cross["recall"],
            "Cross-Dataset F1": res_lstm_cross["f1"],
            "F1 Drop": round(res_lstm_in["f1"] - res_lstm_cross["f1"], 4),
        })

    df_comp = pd.DataFrame(comparison_records)
    print("\n=== Comprehensive In-Domain vs. Cross-Domain Comparison Table ===")
    print(df_comp.to_string(index=False))

    # Save to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_comp.to_csv(RESULTS_DIR / "cross_dataset_comparison.csv", index=False)

    # Save target format table as requested in workflow
    target_table = df_comp[["Model", "In-Domain Acc", "In-Domain Prec", "In-Domain Rec", "In-Domain F1", "Cross-Dataset F1"]].rename(
        columns={
            "In-Domain Acc": "Acc",
            "In-Domain Prec": "Prec",
            "In-Domain Rec": "Rec",
            "In-Domain F1": "F1",
            "Cross-Dataset F1": "Cross-dataset F1",
        }
    )
    target_table.to_csv(RESULTS_DIR / "final_results_table.csv", index=False)

    # Plot Comparison Figure
    plot_cross_domain_drop(df_comp)

    return df_comp


def plot_cross_domain_drop(df_comp: pd.DataFrame, output_path: Path = FIGURES_DIR / "cross_dataset_comparison.png") -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = df_comp[df_comp["Model"] != "Majority baseline"].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_df))
    width = 0.35

    ax.bar(x - width/2, plot_df["In-Domain F1"], width, label="In-Domain (ISOT Test)", color="#2b8cbe")
    ax.bar(x + width/2, plot_df["Cross-Dataset F1"], width, label="Cross-Domain (LIAR Benchmark)", color="#e41a1c")

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("In-Domain vs. Cross-Dataset Generalization Stress Test (Module 10)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Model"], rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, loc="upper right")

    for i in range(len(plot_df)):
        in_val = plot_df["In-Domain F1"].iloc[i]
        out_val = plot_df["Cross-Dataset F1"].iloc[i]
        ax.annotate(f"{in_val:.2f}", (x[i] - width/2, in_val), ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")
        ax.annotate(f"{out_val:.2f}", (x[i] + width/2, out_val), ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Cross-dataset comparison figure saved to {output_path}")


if __name__ == "__main__":
    evaluate_cross_dataset()

