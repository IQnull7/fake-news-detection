# Fake News Detection Using Machine Learning and Natural Language Processing

A modular, scientifically rigorous machine learning and NLP framework to detect misinformation in news articles while actively neutralizing dataset shortcuts (source tag leaks, metadata shortcuts, and duplicate contamination).

---

## 📌 Problem Overview & Objectives

Digital misinformation poses critical threats to public trust, democratic processes, and public health. This project implements an automated, leak-free classification pipeline that analyzes linguistic style, journalistic conventions, and rhetorical patterns to classify articles as **Real** or **Fake**.

### Key Scientific Principles
1. **Shortcut Neutralization:** Eliminating wire agency datelines (e.g. `CITY (Reuters) -`) and dropping perfectly separating metadata columns (`subject` and `date`) so models learn linguistic semantics rather than publisher fingerprints.
2. **Leak-Free Partitioning:** Strict 70/15/15 stratified train/val/test splits with verified zero text overlap. Preprocessing vectorizers and vocabularies are fitted **strictly** on training data.
3. **Multi-Tier Modeling:** Benchmarking trivial baselines against classical classifiers (Multinomial NB, Logistic Regression, Linear SVM, Random Forest, XGBoost) and deep neural architectures (PyTorch BiLSTM).
4. **Generalization Stress Testing:** Evaluating models trained on ISOT against an external, unseen benchmark (**LIAR dataset**) to measure real-world domain transfer.
5. **Zero External Hub Dependencies:** Self-contained architecture using local models, PyTorch, and scikit-learn without any Hugging Face hubs, spaces, or external services.

---

## 📊 Final Results Table

*(Evaluated on identical hold-out test splits; cross-dataset evaluated on the out-of-domain LIAR benchmark)*

| Model | Acc | Prec | Rec | F1 | Cross-dataset F1 (LIAR) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Majority baseline** | 0.5476 | 0.5476 | 1.0000 | 0.7077 | 0.7208 |
| **Title-only LR** | 0.9447 | 0.9328 | 0.9688 | 0.9505 | 0.2693 |
| **TF-IDF + Naive Bayes** | 0.9573 | 0.9656 | 0.9560 | 0.9608 | 0.5399 |
| **TF-IDF + Logistic Regression** | 0.9862 | 0.9842 | 0.9908 | 0.9875 | 0.3252 |
| **TF-IDF + Linear SVM (Calibrated)** | **0.9908** | **0.9905** | **0.9927** | **0.9916** | 0.2547 |
| **TF-IDF + Random Forest** | 0.9702 | 0.9592 | 0.9876 | 0.9732 | 0.5236 |
| **TF-IDF + XGBoost** | 0.9845 | 0.9820 | 0.9898 | 0.9859 | 0.1034 |
| **BiLSTM (PyTorch Deep Model)** | 0.9787 | 0.9830 | 0.9780 | 0.9805 | 0.1900 |

### 5-Fold Cross-Validation Reliability (Logistic Regression)
- **Accuracy:** $0.9889 \pm 0.0018$
- **Precision:** $0.9865 \pm 0.0038$
- **Recall:** $0.9933 \pm 0.0009$
- **F1 Score:** $0.9899 \pm 0.0016$
- **ROC-AUC:** $0.9990 \pm 0.0003$

---

## 🔬 Key Empirical Findings & Caveats

Detailed documentation is preserved in [`CAVEATS_AND_FINDINGS.md`](CAVEATS_AND_FINDINGS.md). Core insights include:

1. **The `subject` Column Shortcut (Caveat 2):**
   In the raw ISOT dataset, real articles belong strictly to `politicsNews` and `worldnews`, while fake articles belong strictly to categories like `News`, `politics`, and `left-news`. Retaining `subject` yielded a trivial 100% artificial accuracy. Dropping it was essential to force the model to analyze text content.
2. **Agency Dateline Leakage (Caveat 1):**
   Over 99.2% of real articles initially opened with wire agency datelines like `WASHINGTON (Reuters) -`. After regex stripping, occurrences of `reuters` in the cleaned corpus dropped to **0**, and the First-5-Words baseline dropped from >99% down to 88.15%.
3. **The Generalization Drop (Caveat 12 & 14):**
   When tested on LIAR (single-sentence political fact-checking statements), model performance dropped from ~98.7% F1 to 25%–54% F1. This highlights the sharp domain gap between full-length journalistic styling (ISOT) and isolated political claims (LIAR).
4. **Interpretability & Stylistic Markers (Module 7):**
   Top features for real news correspond to institutional attribution (`said`, `said on`, `president donald`, weekday datelines), whereas fake news features correspond to blog aggregator and viral formatting cues (`via`, `read more`, `featured image`, `this`).

---

## 📁 Repository Structure

```
fake-news-detection/
├── data/
│   ├── raw/                    # Original True.csv, Fake.csv, liar_test.csv
│   └── processed/              # clean.csv, train.csv, val.csv, test.csv
├── notebooks/
│   ├── 01_load_and_inspect.ipynb    # Data loading, EDA, and leakage analysis
│   └── 02_classical_models.ipynb    # Baselines, model benchmarks, explainability
├── src/
│   ├── __init__.py
│   ├── config.py               # Global SEED=42, file paths, labels
│   ├── preprocess.py           # Regex cleaning, deduplication, stratified splitting
│   ├── evaluate.py             # Reusable classification metrics and confusion matrix
│   ├── train_baselines.py      # Module 4 sanity baselines (Majority, Title, First-5)
│   ├── train_classical.py      # Module 5 classical classifiers & tuning
│   ├── train_lstm.py           # Module 8 PyTorch BiLSTM neural model
│   ├── cross_validation.py     # Module 9 5-fold CV & multi-seed audits
│   ├── cross_dataset_eval.py   # Module 10 generalization test on LIAR
│   └── explainability_audit.py # Module 7 LIME, linear coefficients, error analysis
├── app/
│   └── app.py                  # Interactive Streamlit Web Interface
├── models/                     # Saved joblib pipelines and PyTorch checkpoints
├── results/
│   ├── figures/                # EDA, confusion matrices, and feature importance plots
│   ├── metrics.csv             # Central experiment log
│   ├── final_results_table.csv # In-domain vs. cross-domain comparison
│   └── error_analysis.md       # Comprehensive failure modes report
├── tests/
│   └── test_demo.py            # Automated sanity test for demo interface
├── requirements.txt
├── CAVEATS_AND_FINDINGS.md     # Dynamic experiment log and 20 caveats
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/IQnull7/fake-news-detection.git
cd fake-news-detection

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
# Phase A: Preprocessing and Splits
python -m src.run_eda
python -m src.run_preprocessing
python -m src.run_splits

# Phase B: Baselines and Classical Models
python -m src.train_baselines
python -m src.train_classical
python -m src.explainability_audit

# Phase C: Deep Learning Model (PyTorch BiLSTM)
python -m src.train_lstm

# Phase D: Reliability & Cross-Dataset Stress Test
python -m src.cross_validation
python -m src.cross_dataset_eval
```

### 3. Launch the Interactive Web Application (Module 11)
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser to test custom news articles or pre-loaded samples with instant stylistic explanations.

---

## ⚖️ Ethical Framing & Limitations (Caveat 19)

- **Decision-Support Scope:** This tool is designed as an editorial aid to flag stylistic anomalies and sensationalist phrasing. It is **not** an automated arbiter of factual truth.
- **Publisher Proxy Labels:** Ground truth labels in the ISOT dataset represent the publishing domain's track record rather than independent sentence-by-sentence fact-checks.
- **Domain Shift:** High in-domain accuracy on 2016–2017 political articles does not generalize universally to modern 2026 reporting without domain adaptation.
