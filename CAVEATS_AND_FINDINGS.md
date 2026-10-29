# Project Caveats, Guidelines, and Findings

This document tracks all project constraints, critical caveats (data, modeling, evaluation, practical), and dynamic findings discovered throughout the execution of the modular workflow.

---

## 1. Core Caveats & Principles

### Data Caveats
1. **Source Leakage (Critical):**
   - Real articles originate largely from Reuters and typically start with patterns like `CITY (Reuters) -` or `[LOCATION] (Reuters)`.
   - Fake articles often come from distinct flagged websites with signature framing/clickbait styles.
   - Without aggressive stripping, models easily learn these source fingerprints rather than text content or style, inflating scores artificially to near 99-100%.
2. **`subject` and `date` Column Leakage:**
   - In ISOT, subject labels barely overlap between classes (e.g., `politicsNews`, `worldnews` for real vs. `News`, `politics`, `Government News`, `left-news`, `US_News`, `Middle-east` for fake).
   - Date formats and distributions also create trivial decision boundaries.
   - **Resolution:** Drop `subject` and `date` entirely from the feature set.
3. **Exact Duplicates Across Splits:**
   - Duplicate articles across train/val/test falsely inflate generalization metrics.
   - **Resolution:** Deduplicate on raw text and `title + text` *before* splitting.
4. **Narrow Temporal and Topic Window:**
   - The ISOT dataset is heavily centered on US politics during the 2016–2017 election cycle.
   - Models risk learning entity/topic vocabulary (e.g., specific political figures) rather than stylistic cues of deception.
5. **Proxy Ground Truth ("Fake" vs. "Real"):**
   - Labels reflect publisher status (flagged domain vs. established wire service), not sentence-by-sentence fact-checking.
   - The system detects stylistic, rhetorical, and source patterns associated with misinformation, not absolute factual truth.

### Modeling Caveats
6. **No Vectorizer or Scaler Leakage:**
   - Preprocessing transformers (e.g. `TfidfVectorizer`) must be fitted **strictly on the training split**. Never fit on the full dataset or validation/test sets.
7. **Strict Split Discipline:**
   - The test set is locked and touched only once for the final evaluation. All hyperparameter tuning and model decisions happen on the validation set or via cross-validation.
8. **Text Preprocessing Strategy by Model Family:**
   - **Classical ML (TF-IDF):** Benefits from lowercasing, punctuation handling, URL stripping, and light lemmatization/stopword removal.
   - **Transformers (BERT / DistilBERT):** Must receive `text_light` (near-natural text with capitalization, grammar, and sentence structure preserved). Heavy stemming or stopword removal degrades contextual representations.
9. **BERT Sequence Length & Truncation:**
   - Articles often exceed the 512-token limit. A clear truncation policy (e.g., head vs. head+tail) must be documented and tested consistently.
10. **Comprehensive Metrics Reporting:**
    - Accuracy alone can mask asymmetric errors. Always report precision, recall, F1-score, and ROC-AUC.
11. **Reproducibility:**
    - Use a single global seed (`SEED = 42`) across all random generators, data splits, and model initializations.

### Evaluation and Reporting Caveats
12. **In-Domain vs. Out-of-Domain Reality:**
    - A 98-99% score on ISOT does not imply real-world efficacy. Cross-dataset testing (Module 10, e.g., on WELFake or LIAR) provides the true measure of domain transfer and generalization.
13. **Variance and Robustness:**
    - Single-run results can be sensitive to lucky random seeds. Multi-seed runs or cross-validation reporting (`mean ± std`) are required for robust conclusions.
14. **Negative Results are Scientific Findings:**
    - Performance drops during cross-dataset evaluation or ablation studies are expected and valuable. Explain domain shift, topic shift, and vocabulary drift rather than attempting to hide it.
15. **Systematic Experiment Tracking:**
    - All experiment runs, baseline scores, and model evaluations must append directly to `results/metrics.csv`.

### Practical Caveats
16. **Compute Boundaries:**
    - Classical pipelines run efficiently on CPU. Transformer fine-tuning requires GPU resources (e.g., Colab / Kaggle).
17. **Demo Robustness:**
    - Production app (`app/app.py`) must load models once at startup, cap input token lengths, and gracefully handle empty or degenerate user inputs.
18. **Reference Compliance:**
    - Citations and literature alignment follow IEEE style across documentation and report preparation.
19. **Ethical Framing:**
    - Misinformation detection tools serve as decision-support / flagging mechanisms for reviewers and readers, not definitive truth arbiters.
20. **Infrastructure & Platform Constraint (Zero Hugging Face Dependency):**
    - Under explicit instruction, Hugging Face Hub, datasets, spaces, and hosting will NOT be used. Models, datasets, demos, and evaluations will be managed strictly via local disk (`models/`), GitHub, Kaggle, and local app runtimes (Streamlit/Gradio).

---

## 2. Dynamic Findings & Experiment Log

*(This section will be updated incrementally at each module checkpoint)*

### Phase A: Data Preparation
- **Module 0 (Setup):**
  - Initialized modular architecture, standard directories, global configuration (`SEED = 42`), and isolated virtual environment.
  - Documented initial 19 caveats + Caveat 20 strictly barring Hugging Face services/uploads/hubs.
- **Module 1 (Load & Inspect):**
  - **Dataset Verified:** Combined 44,898 total articles (Real: 21,417 / 47.70%, Fake: 23,481 / 52.30%).
  - **Missing Values:** Zero null cells across all 4 raw columns (`title`, `text`, `subject`, `date`).
  - **Empty Text Bodies:** 631 articles contain empty strings or whitespace-only text bodies that require filtering in Module 2.
  - **Empirical Confirmation of Caveat 2 (`subject` shortcut):**
    - Real classes map strictly to `politicsNews` (11,272) and `worldnews` (10,145).
    - Fake classes map strictly to `News` (9,050), `politics` (6,841), `left-news` (4,459), `Government News` (1,570), `US_News` (783), and `Middle-east` (778).
    - **Cross-class overlap is strictly 0.0%.** Retaining `subject` would completely invalidate model authenticity.
  - **Empirical Confirmation of Caveat 1 (Source fingerprinting):**
    - >99% of Real articles begin with agency datelines (e.g. `CITY (Reuters) - `).
    - Failure to strip these patterns enables trivial classifier convergence on agency names rather than content semantics.
  - **Deliverables Generated:**
    - `notebooks/01_load_and_inspect.ipynb`
    - Publication-ready visualizations in `results/figures/`:
      - `class_balance.png`
      - `subject_leakage.png`
      - `text_length_distribution.png`
      - `reuters_leakage_rate.png`
- **Module 2 (Cleaning & Leakage Removal):**
  - **Initial Raw Rows:** 44,898
  - **Deduplication:**
    - Dropped 5,793 exact duplicates on `(title, text)`.
    - Dropped 459 additional duplicates on `text` alone.
    - Dropped 272 post-cleaning duplicates on `text_clean` / `text_light` (articles differing only by stripped agency headers).
  - **Short / Empty Text Removal:** Dropped 163 articles with fewer than 10 words.
  - **Total Rows Dropped:** 6,687 rows (14.89% of raw data).
  - **Metadata Pruning:** Permanently dropped `subject` and `date` columns to prevent 100% shortcut leakage.
  - **Agency Fingerprint Neutralization:**
    - Regex applied for leading wire datelines (`^[\\s\\w\\.,\\/\\-\\']+\\((?:Reuters|AP|AFP)\\)\\s*[-–—]?\\s*`) and standalone agency tokens.
    - Post-cleaning audit: **`reuters` count in `text_clean` is exactly 0 for both Real and Fake**; **`reuters` count in `text_light` is exactly 0 for both Real and Fake**.
  - **Final Cleaned Dataset:** 38,211 rows (Real: 20,924 [54.76%], Fake: 17,287 [45.24%]).
  - **Deliverable:** Serialized to `data/processed/clean.csv` with dual representations (`text_clean` for classical TF-IDF, `text_light` for contextual deep learning). Checkpoint passed.
- **Module 3 (Train/Val/Test Stratified Split):**
  - **Partitioning Strategy:** Stratified 70% Train / 15% Validation / 15% Test with global `SEED = 42`.
  - **Split Counts:**
    - Train: 26,747 rows (Real: 54.76%, Fake: 45.24%)
    - Val: 5,732 rows (Real: 54.76%, Fake: 45.24%)
    - Test: 5,732 rows (Real: 54.76%, Fake: 45.24%)
  - **Leakage & Overlap Audit:**
    - `Train ∩ Val`: 0 overlapping articles.
    - `Train ∩ Test`: 0 overlapping articles.
    - `Val ∩ Test`: 0 overlapping articles.
  - **Deliverables:** Saved `data/processed/train.csv`, `val.csv`, and `test.csv`; logged audit to `results/split_summary.json`. Checkpoint passed.

### Phase B: Classical Models & Audits
- **Module 4 (Sanity Baselines):**
  - **Majority Baseline:** Accuracy 54.76%, F1 0.7077, ROC-AUC 0.5000 (reflects 54.76% positive class balance).
  - **Title-Only LR:** Accuracy 94.66%, F1 0.9522, ROC-AUC 0.9871. Demonstrates that headline style (sensational clickbait vs. formal reporting) carries substantial discriminative signal on its own.
  - **First-5-Words LR:** Accuracy 88.15%, F1 0.8921, ROC-AUC 0.9501.
  - **Checkpoint Verification:** In raw data with intact wire headers, first-5-words scored >99%. After regex dateline neutralization, accuracy dropped significantly to 88.15%, and inspection confirmed top features reflect conversational lead pronouns (`you`, `we`, `our`) vs. formal journalistic leads (`president donald`, `minister`), rather than wire agency tags.
- **Module 5 & 6 (TF-IDF + Classical Classifiers & Reusable Evaluation):**
  - Feature extraction: `TfidfVectorizer(ngram_range=(1,2), max_features=50,000, sublinear_tf=True)` fit strictly on `train.csv`.
  - Hyperparameter tuning strictly on validation set:
    - **Multinomial Naive Bayes (alpha=0.01):** Val Acc 95.29%, Val F1 0.9572, ROC-AUC 0.9885
    - **Logistic Regression (C=10.0):** Val Acc 98.67%, Val F1 0.9879, ROC-AUC 0.9983
    - **Linear SVM (Calibrated, C=2.0):** Val Acc 98.94%, Val F1 0.9903, ROC-AUC 0.9990
    - **Random Forest (n=150, depth=35):** Val Acc 97.23%, Val F1 0.9751, ROC-AUC 0.9970
    - **XGBoost (n=150, depth=6):** Val Acc 98.74%, Val F1 0.9886, ROC-AUC 0.9988
  - All classical models significantly outperformed the Module 4 baselines. End-to-end inference pipelines verified on raw text strings.
- **Module 7 (Leakage & Explainability Audit):**
  - **Feature Weight Inspection:**
    - Top Fake News indicators: `via`, `read more`, `read`, `featured image`, `image via`, `this`, `gop`.
    - Top Real News indicators: `said`, `said on`, `president donald`, `on wednesday`, `on thursday`, `on tuesday`.
    - Zero agency tokens (`reuters`, wire codes) present in top weights.
  - **Hold-out Test Evaluation (Logistic Regression):**
    - Accuracy: 98.62%, Precision: 98.42%, Recall: 99.08%, F1: 0.9875, ROC-AUC: 0.9989.
  - **Error Analysis (79 misclassified test samples / 1.38% error rate):**
    - False Positives (50): Fake articles verbatim-quoting official statements or speeches.
    - False Negatives (29): Real news op-eds written in informal or first-person rhetoric.
  - **Explainability Artifacts:** Generated `results/figures/feature_importance_top20.png` and documented report in `results/error_analysis.md`. Checkpoint passed.

### Phase C: Deep Learning Model
- **Module 8 (Self-Contained PyTorch BiLSTM):**
  - **Architecture:** 2-layer Bidirectional LSTM with 128-dim embeddings, 128-dim hidden states, spatial dropout (0.3), dual pooling (Average Pooling + Max Pooling = 512 dims), and dense linear classification head.
  - **Zero Hugging Face Dependency:** Vocabulary (25,000 tokens) built purely on training text (`text_light`); model trained from scratch with PyTorch.
  - **Convergence:**
    - Epoch 1: Train Loss 0.2249 | Val Loss 0.0916 | Val Acc 96.89%
    - Epoch 2: Train Loss 0.0773 | Val Loss 0.0820 | Val Acc 97.26%
  - **Hold-out Test Performance:** Accuracy 97.87%, Precision 98.30%, Recall 97.80%, F1 0.9805, ROC-AUC 0.9976.
  - **Latency Checkpoint:** Single article inference executes in 12.73 ms (<2s limit). Checkpoint passed.

### Phase D: Generalization & Reliability Audits
- **Module 9 (Cross-Validation Reliability):**
  - 5-Fold Stratified Cross-Validation on Logistic Regression:
    - Accuracy:  `0.9889 ± 0.0018`
    - Precision: `0.9865 ± 0.0038`
    - Recall:    `0.9933 ± 0.0009`
    - F1 Score:  `0.9899 ± 0.0016`
    - ROC-AUC:   `0.9990 ± 0.0003`
  - Multi-seed check across seeds `[42, 101, 2024, 7, 999]` yielded zero variance on the convex linear boundary. Standard deviation is exceptionally small (`<0.002`), demonstrating that high performance is not a lucky seed artifact (Caveat 13).
- **Module 10 (Cross-Dataset Generalization Stress Test):**
  - Evaluated all trained models against the out-of-domain **LIAR Benchmark Dataset** (1,267 political statements) with zero retraining or fine-tuning.
  - **Empirical Findings (Caveats 12 & 14):**
    - Logistic Regression F1 dropped from 98.75% to 32.52% (Drop: 0.6623).
    - Linear SVM F1 dropped from 99.16% to 25.47% (Drop: 0.7369).
    - BiLSTM F1 dropped from 98.05% to 19.00% (Drop: 0.7905).
    - Naive Bayes exhibited the highest cross-domain resilience with F1 53.99% (Drop: 0.4209).
  - **Scientific Explanation:**
    1. *Format Shift:* Full-length 400-word journalistic articles vs. isolated 15-word debate soundbites.
    2. *Task Shift:* Stylistic fakeness (sensationalist blog formatting) vs. objective political fact-checking (PolitiFact verifications of actual statements by public officials).
    3. *Temporal & Source Shift:* 2016-2017 election cycle news vs. 2007-2016 multi-source political rhetoric.

### Phase E: Interactive Interface & Delivery
- **Module 11 (Interactive Streamlit Demo):**
  - Production application developed in `app/app.py`.
  - Single-load caching (`@st.cache_resource`) for fast sub-second predictions.
  - Input length validation (<10 words friendly warning, handling empty strings).
  - Local feature contribution visualizer highlighting top words driving Real vs. Fake probability.
  - Pre-loaded with 4 representative test articles (real political news, viral conspiracy, diplomatic wire, shocking clickbait) and custom user input. Checkpoint passed (5/5 tests successful with zero crashes).
- **Module 12 & 13 (Showcase & Documentation):**
  - Complete report and benchmark tables synchronized across `results/metrics.csv`, `results/final_results_table.csv`, and `README.md`.

