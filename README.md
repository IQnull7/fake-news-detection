# Fake News Detection Using Machine Learning and Natural Language Processing

A modular, leak-free machine learning and NLP framework designed to classify news articles as real or fake, following strict scientific evaluation and anti-leakage methodology.

---

## Project Overview

- **Objective:** Detect misinformation and stylistic signatures in news articles while eliminating dataset-specific shortcuts (source tags, metadata leakage, duplicate contamination).
- **Core Methodology:**
  1. Systematic data cleaning & regex-based agency tag stripping (e.g. Reuters headers).
  2. Baseline shortcut auditing (Module 4) to ensure the model learns linguistic style rather than metadata.
  3. Classical ML benchmarks (TF-IDF + Naive Bayes, Logistic Regression, Linear SVM, Random Forest / XGBoost).
  4. Deep Transformer modeling (DistilBERT fine-tuning).
  5. Explainability audits (LIME / feature coefficients).
  6. Cross-dataset generalization stress test (WELFake / LIAR).
  7. Interactive web demo (Streamlit / Gradio).

---

## Directory Structure

```
fake-news-detection/
├── data/
│   ├── raw/                # Original dataset CSVs (True.csv, Fake.csv)
│   └── processed/          # Cleaned, deduplicated, and split sets
├── notebooks/              # Jupyter notebooks for EDA and analysis
├── src/
│   ├── __init__.py
│   ├── config.py           # Global SEED=42 and paths
│   ├── preprocess.py       # Cleaning, regex stripping & dataset preparation
│   ├── evaluate.py         # Standardized evaluation & metrics logging
│   ├── train_classical.py  # Classical model training & tuning
│   └── train_bert.py       # Transformer fine-tuning pipeline
├── app/                    # Interactive web demo (Streamlit / Gradio)
├── models/                 # Serialized model artifacts
├── results/                # Evaluation metrics CSVs and figures
│   └── figures/
├── CAVEATS_AND_FINDINGS.md # Project caveats and empirical findings log
├── requirements.txt
└── README.md
```

---

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IQnull7/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Reproducible Pipeline:**
   Follow the modular workflow outlined in `CAVEATS_AND_FINDINGS.md`.

