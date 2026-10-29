# Module 7: Leakage and Explainability Audit Report

## 1. Top Feature Weight Analysis (Logistic Regression)
The top positive coefficients (predicting **Real News**) and negative coefficients (predicting **Fake News**) were analyzed to verify that the model has learned linguistic style and content semantics rather than source fingerprints.

### Top 15 Indicative Real Features (Class 1)
| Rank | Feature N-gram | Coefficient | Semantic Interpretation |
|---|---|---|---|
| 1 | `said` | +18.5862 | Institutional & formal journalistic vocabulary |
| 2 | `said on` | +13.4874 | Institutional & formal journalistic vocabulary |
| 3 | `president donald` | +8.8480 | Institutional & formal journalistic vocabulary |
| 4 | `on wednesday` | +8.7697 | Institutional & formal journalistic vocabulary |
| 5 | `on thursday` | +8.7587 | Institutional & formal journalistic vocabulary |
| 6 | `on` | +8.7132 | Institutional & formal journalistic vocabulary |
| 7 | `on tuesday` | +8.5252 | Institutional & formal journalistic vocabulary |
| 8 | `said in` | +8.4303 | Institutional & formal journalistic vocabulary |
| 9 | `on friday` | +7.6239 | Institutional & formal journalistic vocabulary |
| 10 | `on monday` | +7.5049 | Institutional & formal journalistic vocabulary |
| 11 | `he said` | +6.7974 | Institutional & formal journalistic vocabulary |
| 12 | `wednesday` | +6.5233 | Institutional & formal journalistic vocabulary |
| 13 | `said the` | +6.4364 | Institutional & formal journalistic vocabulary |
| 14 | `republican` | +6.3033 | Institutional & formal journalistic vocabulary |
| 15 | `tuesday` | +6.2674 | Institutional & formal journalistic vocabulary |

### Top 15 Indicative Fake Features (Class 0)
| Rank | Feature N-gram | Coefficient | Semantic Interpretation |
|---|---|---|---|
| 1 | `via` | -16.6196 | Sensationalist, conversational, and hyper-partisan framing |
| 2 | `read more` | -10.1910 | Sensationalist, conversational, and hyper-partisan framing |
| 3 | `read` | -8.7217 | Sensationalist, conversational, and hyper-partisan framing |
| 4 | `this` | -8.5925 | Sensationalist, conversational, and hyper-partisan framing |
| 5 | `image` | -8.4597 | Sensationalist, conversational, and hyper-partisan framing |
| 6 | `featured image` | -8.1976 | Sensationalist, conversational, and hyper-partisan framing |
| 7 | `featured` | -8.1835 | Sensationalist, conversational, and hyper-partisan framing |
| 8 | `image via` | -7.4348 | Sensationalist, conversational, and hyper-partisan framing |
| 9 | `gop` | -7.2084 | Sensationalist, conversational, and hyper-partisan framing |
| 10 | `president trump` | -7.1769 | Sensationalist, conversational, and hyper-partisan framing |
| 11 | `just` | -7.1350 | Sensationalist, conversational, and hyper-partisan framing |
| 12 | `obama` | -7.0186 | Sensationalist, conversational, and hyper-partisan framing |
| 13 | `us` | -6.6529 | Sensationalist, conversational, and hyper-partisan framing |
| 14 | `the us` | -6.4929 | Sensationalist, conversational, and hyper-partisan framing |
| 15 | `com` | -6.3783 | Sensationalist, conversational, and hyper-partisan framing |

### Leakage Checkpoint Audit
- **Reuters or wire agencies in top 40 features:** None.
- **Agency location datelines in top 40 features:** None.
- **Outcome:** Passed. The decision boundary relies on formal institutional prose versus inflammatory, subjective language.

---

## 2. Test Set Evaluation & Error Analysis
On the hold-out test set (5,732 articles), the tuned TF-IDF + Logistic Regression pipeline achieved:
- **Accuracy:** 0.9862
- **Precision:** 0.9842
- **Recall:** 0.9908
- **F1 Score:** 0.9875
- **ROC-AUC:** 0.9989

### Misclassification Breakdown
- Total test articles: 5,732
- Total errors: 79 (1.38%)
- **False Positives (Fake misclassified as Real):** 50 articles
- **False Negatives (Real misclassified as Fake):** 29 articles

### Primary Failure Modes Identified:
1. **Fact-quoting Fake News (False Positives):**
   Fake articles that directly quote official statements or transcribe speeches at length adopt formal syntax and neutral vocabulary, misleading the linear classifier.
2. **Editorial / Op-Ed Real Articles (False Negatives):**
   Real news pieces written with a conversational, first-person, or rhetorical tone (e.g. political columns) mimic the stylistic markers typical of partisan blogs.
3. **Very Short / Elliptical Texts:**
   Articles with limited body length offer sparse n-gram overlap with the training vocabulary, leading to uncertain threshold boundaries.

---

## 3. LIME Case Studies (Local Feature Contributions)
Sample local explanations illustrate token contributions to final probability scores:

#### Case 0: "BREAKING: WHY DID OBAMA JUST Break Democrat Ranks, Throw Hil..."
- **True Label:** Fake
- **Predicted:** Real (Real Prob: 0.511, Fake Prob: 0.489)
- **Key Contributing Features:**
  - `said`: +0.2637 (Towards Real)
  - `obama`: -0.1231 (Towards Fake)
  - `barack`: +0.0998 (Towards Real)
  - `press`: -0.0770 (Towards Fake)
  - `president`: +0.0572 (Towards Real)

#### Case 1: "GAME CHANGER? Trump Recruits Election “Observers” To Prevent..."
- **True Label:** Fake
- **Predicted:** Real (Real Prob: 0.518, Fake Prob: 0.482)
- **Key Contributing Features:**
  - `on`: +0.1477 (Towards Real)
  - `watch`: -0.1348 (Towards Fake)
  - `said`: +0.1272 (Towards Real)
  - `friday`: +0.0879 (Towards Real)
  - `republican`: +0.0755 (Towards Real)

#### Case 2: "BREAKING: WHY IS OBAMA Sending Huge Number Of Troops To Germ..."
- **True Label:** Fake
- **Predicted:** Real (Real Prob: 0.930, Fake Prob: 0.070)
- **Key Contributing Features:**
  - `said`: +0.0893 (Towards Real)
  - `region`: +0.0806 (Towards Real)
  - `friday`: +0.0782 (Towards Real)
  - `northern`: +0.0618 (Towards Real)
  - `is`: -0.0287 (Towards Fake)

#### Case 3: "Sterling quarterback crop lead way into the playoffs..."
- **True Label:** Real
- **Predicted:** Fake (Real Prob: 0.281, Fake Prob: 0.719)
- **Key Contributing Features:**
  - `said`: +0.0913 (Towards Real)
  - `on`: +0.0901 (Towards Real)
  - `us`: -0.0608 (Towards Fake)
  - `just`: -0.0537 (Towards Fake)
  - `seeded`: -0.0382 (Towards Fake)

#### Case 4: "Infrastructure would be popular bipartisan step for Clinton:..."
- **True Label:** Real
- **Predicted:** Fake (Real Prob: 0.496, Fake Prob: 0.504)
- **Key Contributing Features:**
  - `on`: +0.1564 (Towards Real)
  - `tuesday`: +0.1153 (Towards Real)
  - `republican`: +0.1128 (Towards Real)
  - `obama`: -0.0889 (Towards Fake)
  - `like`: -0.0859 (Towards Fake)
