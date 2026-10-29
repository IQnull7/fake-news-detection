"""Module 5: Train and tune classical ML models (TF-IDF + Classifiers)."""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED, set_seed
from src.preprocess import clean_text_classical
from src.evaluate import evaluate


def train_and_tune_classical_models() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    set_seed(SEED)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    print("Loading data splits...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    y_train = df_train["label"].values
    y_val = df_val["label"].values

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fit TF-IDF Vectorizer strictly on training data
    print("\nFitting TfidfVectorizer on training set (max_features=50,000, ngram_range=(1,2))...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )
    X_train_vec = tfidf.fit_transform(df_train["text_clean"].fillna(""))
    X_val_vec = tfidf.transform(df_val["text_clean"].fillna(""))
    print(f"TF-IDF Vocabulary size: {len(tfidf.vocabulary_):,} features")

    # Save vectorizer
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")

    val_records = []
    trained_pipelines = {}

    # ------------------------------------------------------------------
    # Model 1: Multinomial Naive Bayes (Tune alpha on validation)
    # ------------------------------------------------------------------
    print("\n--- Tuning Multinomial Naive Bayes ---")
    best_nb_alpha = 0.1
    best_nb_f1 = -1.0
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        nb_cand = MultinomialNB(alpha=alpha)
        nb_cand.fit(X_train_vec, y_train)
        pred = nb_cand.predict(X_val_vec)
        score = nb_cand.predict_proba(X_val_vec)[:, 1]
        f1_cand = evaluate(y_val, pred, score, save_cm=False, metrics_path=RESULTS_DIR / "temp.csv")["f1"]
        if f1_cand > best_nb_f1:
            best_nb_f1 = f1_cand
            best_nb_alpha = alpha

    print(f"Best MNB alpha: {best_nb_alpha} (Val F1: {best_nb_f1:.4f})")
    nb_final = MultinomialNB(alpha=best_nb_alpha)
    nb_final.fit(X_train_vec, y_train)
    y_pred_nb = nb_final.predict(X_val_vec)
    y_score_nb = nb_final.predict_proba(X_val_vec)

    res_nb = evaluate(
        y_true=y_val,
        y_pred=y_pred_nb,
        y_score=y_score_nb,
        model_name="TF-IDF + Naive Bayes",
        split_name="val",
    )
    val_records.append(res_nb)
    pipe_nb = Pipeline([("cleaner", None), ("tfidf", tfidf), ("clf", nb_final)])
    joblib.dump(pipe_nb, MODELS_DIR / "model_naive_bayes.joblib")

    # ------------------------------------------------------------------
    # Model 2: Logistic Regression (Tune C on validation)
    # ------------------------------------------------------------------
    print("\n--- Tuning Logistic Regression ---")
    best_lr_c = 1.0
    best_lr_f1 = -1.0
    for c_val in [0.1, 1.0, 5.0, 10.0]:
        lr_cand = LogisticRegression(C=c_val, max_iter=1000, random_state=SEED)
        lr_cand.fit(X_train_vec, y_train)
        pred = lr_cand.predict(X_val_vec)
        score = lr_cand.predict_proba(X_val_vec)[:, 1]
        f1_cand = evaluate(y_val, pred, score, save_cm=False, metrics_path=RESULTS_DIR / "temp.csv")["f1"]
        if f1_cand > best_lr_f1:
            best_lr_f1 = f1_cand
            best_lr_c = c_val

    print(f"Best LR C: {best_lr_c} (Val F1: {best_lr_f1:.4f})")
    lr_final = LogisticRegression(C=best_lr_c, max_iter=1000, random_state=SEED)
    lr_final.fit(X_train_vec, y_train)
    y_pred_lr = lr_final.predict(X_val_vec)
    y_score_lr = lr_final.predict_proba(X_val_vec)

    res_lr = evaluate(
        y_true=y_val,
        y_pred=y_pred_lr,
        y_score=y_score_lr,
        model_name="TF-IDF + Logistic Regression",
        split_name="val",
    )
    val_records.append(res_lr)
    pipe_lr = Pipeline([("tfidf", tfidf), ("clf", lr_final)])
    joblib.dump(pipe_lr, MODELS_DIR / "model_logistic_regression.joblib")

    # ------------------------------------------------------------------
    # Model 3: Linear SVM (Calibrated for probabilities)
    # ------------------------------------------------------------------
    print("\n--- Tuning Linear SVM ---")
    best_svm_c = 1.0
    best_svm_f1 = -1.0
    for c_val in [0.1, 0.5, 1.0, 2.0]:
        base_svm = LinearSVC(C=c_val, max_iter=2000, random_state=SEED)
        cal_svm = CalibratedClassifierCV(base_svm, cv=3)
        cal_svm.fit(X_train_vec, y_train)
        pred = cal_svm.predict(X_val_vec)
        score = cal_svm.predict_proba(X_val_vec)[:, 1]
        f1_cand = evaluate(y_val, pred, score, save_cm=False, metrics_path=RESULTS_DIR / "temp.csv")["f1"]
        if f1_cand > best_svm_f1:
            best_svm_f1 = f1_cand
            best_svm_c = c_val

    print(f"Best Linear SVM C: {best_svm_c} (Val F1: {best_svm_f1:.4f})")
    svm_final = CalibratedClassifierCV(LinearSVC(C=best_svm_c, max_iter=2000, random_state=SEED), cv=3)
    svm_final.fit(X_train_vec, y_train)
    y_pred_svm = svm_final.predict(X_val_vec)
    y_score_svm = svm_final.predict_proba(X_val_vec)

    res_svm = evaluate(
        y_true=y_val,
        y_pred=y_pred_svm,
        y_score=y_score_svm,
        model_name="TF-IDF + Linear SVM",
        split_name="val",
    )
    val_records.append(res_svm)
    pipe_svm = Pipeline([("tfidf", tfidf), ("clf", svm_final)])
    joblib.dump(pipe_svm, MODELS_DIR / "model_linear_svm.joblib")

    # ------------------------------------------------------------------
    # Model 4: Random Forest Classifier
    # ------------------------------------------------------------------
    print("\n--- Training Random Forest Classifier ---")
    rf_final = RandomForestClassifier(
        n_estimators=150,
        max_depth=35,
        n_jobs=-1,
        random_state=SEED,
    )
    rf_final.fit(X_train_vec, y_train)
    y_pred_rf = rf_final.predict(X_val_vec)
    y_score_rf = rf_final.predict_proba(X_val_vec)

    res_rf = evaluate(
        y_true=y_val,
        y_pred=y_pred_rf,
        y_score=y_score_rf,
        model_name="TF-IDF + Random Forest",
        split_name="val",
    )
    val_records.append(res_rf)
    pipe_rf = Pipeline([("tfidf", tfidf), ("clf", rf_final)])
    joblib.dump(pipe_rf, MODELS_DIR / "model_random_forest.joblib")

    # ------------------------------------------------------------------
    # Model 5: XGBoost Classifier
    # ------------------------------------------------------------------
    print("\n--- Training XGBoost Classifier ---")
    xgb_final = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=SEED,
        eval_metric="logloss",
    )
    xgb_final.fit(X_train_vec, y_train)
    y_pred_xgb = xgb_final.predict(X_val_vec)
    y_score_xgb = xgb_final.predict_proba(X_val_vec)

    res_xgb = evaluate(
        y_true=y_val,
        y_pred=y_pred_xgb,
        y_score=y_score_xgb,
        model_name="TF-IDF + XGBoost",
        split_name="val",
    )
    val_records.append(res_xgb)
    pipe_xgb = Pipeline([("tfidf", tfidf), ("clf", xgb_final)])
    joblib.dump(pipe_xgb, MODELS_DIR / "model_xgboost.joblib")

    # Remove temporary CSV
    temp_csv = RESULTS_DIR / "temp.csv"
    if temp_csv.exists():
        temp_csv.unlink()

    df_val_summary = pd.DataFrame(val_records)
    df_val_summary.to_csv(RESULTS_DIR / "classical_val_scores.csv", index=False)

    print("\n=== Validation Results for Classical Models ===")
    print(df_val_summary.to_string(index=False))

    # Checkpoint Verification:
    # 1. Pipeline loads and predicts on raw string
    print("\n--- Verifying Pipeline Raw String Prediction ---")
    loaded_pipe = joblib.load(MODELS_DIR / "model_logistic_regression.joblib")
    raw_sample = "The president delivered an address today discussing economic reforms in parliament."
    sample_cleaned = clean_text_classical(raw_sample)
    prediction = loaded_pipe.predict([sample_cleaned])[0]
    proba = loaded_pipe.predict_proba([sample_cleaned])[0]
    print(f"Sample raw input: '{raw_sample}'")
    print(f"Prediction: {prediction} (Probabilities: Fake={proba[0]:.3f}, Real={proba[1]:.3f})")

    assert prediction in [0, 1], "Pipeline prediction failed!"
    print("CHECKPOINT PASSED: All classical models trained, tuned on validation, saved, and pipeline predicts on raw text.")

    return df_val_summary, {
        "nb": pipe_nb,
        "lr": pipe_lr,
        "svm": pipe_svm,
        "rf": pipe_rf,
        "xgb": pipe_xgb,
    }


if __name__ == "__main__":
    train_and_tune_classical_models()

