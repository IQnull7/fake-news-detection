"""Reusable evaluation metrics, confusion matrix plotting, and experiment logging."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import METRICS_CSV_PATH, FIGURES_DIR, LABEL_MAP


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    model_name: str = "Model",
    split_name: str = "val",
    save_cm: bool = True,
    cm_filename: Optional[str] = None,
    metrics_path: Path = METRICS_CSV_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> Dict[str, float]:
    """Compute standard classification metrics, plot confusion matrix, and append to metrics.csv.

    Args:
        y_true: Ground truth binary labels (1 = Real, 0 = Fake)
        y_pred: Predicted binary labels
        y_score: Predicted probabilities or decision scores for positive class (1)
        model_name: Identifier for model (e.g. 'Majority baseline', 'TF-IDF + Logistic Regression')
        split_name: Dataset split evaluated ('val', 'test', 'cross_dataset')
        save_cm: Whether to generate and save confusion matrix figure
        cm_filename: Custom filename for confusion matrix
        metrics_path: Path to CSV where metrics are logged
        figures_dir: Directory where confusion matrix plots are saved

    Returns:
        Dict[str, float] of computed metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc: Optional[float] = None
    if y_score is not None:
        try:
            # If 2D probability matrix, take column 1 (positive class)
            if hasattr(y_score, "ndim") and y_score.ndim == 2:
                score_to_use = y_score[:, 1]
            else:
                score_to_use = y_score
            roc_auc = float(roc_auc_score(y_true, score_to_use))
        except Exception as e:
            roc_auc = None

    metrics = {
        "model": model_name,
        "split": split_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
    }

    # Save confusion matrix plot
    if save_cm:
        figures_dir.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred Fake (0)", "Pred Real (1)"],
            yticklabels=["True Fake (0)", "True Real (1)"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix: {model_name} ({split_name})", fontsize=11, fontweight="bold")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
        filename = cm_filename or f"cm_{safe_name}_{split_name}.png"
        fig_path = figures_dir / filename
        plt.savefig(fig_path, dpi=300)
        plt.close()

    # Append to metrics.csv
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([metrics])
    if metrics_path.exists():
        row_df.to_csv(metrics_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(metrics_path, mode="w", header=True, index=False)

    return metrics

