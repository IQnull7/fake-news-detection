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
  - *To be recorded upon regex source stripping and deduplication.*
- **Module 3 (Splits):**
  - *To be recorded upon stratified split generation.*

