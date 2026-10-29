"""Fake News Detection - Interactive Streamlit Web Application (Module 11)."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.preprocess import clean_text_classical
from src.config import MODELS_DIR, RESULTS_DIR


st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
)


@st.cache_resource
def load_production_pipeline():
    """Load trained Logistic Regression pipeline and TF-IDF weights once at startup."""
    model_path = MODELS_DIR / "model_logistic_regression.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run training first.")

    pipeline = joblib.load(model_path)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    # Pre-compute dictionary of feature coefficients for fast local word explanations
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    weights_dict = {feat: float(c) for feat, c in zip(feature_names, coefs)}

    return pipeline, weights_dict


def get_contributing_words(clean_text: str, weights_dict: dict, top_k: int = 6):
    """Identify top words in the input that contribute towards Real (+) or Fake (-)."""
    tokens = clean_text.split()
    matched = []
    seen = set()

    # Check unigrams and bigrams
    for tok in tokens:
        if tok in weights_dict and tok not in seen:
            matched.append((tok, weights_dict[tok]))
            seen.add(tok)

    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in weights_dict and bigram not in seen:
            matched.append((bigram, weights_dict[bigram]))
            seen.add(bigram)

    matched.sort(key=lambda x: abs(x[1]), reverse=True)

    real_contributors = [(w, c) for w, c in matched if c > 0][:top_k]
    fake_contributors = [(w, c) for w, c in matched if c < 0][:top_k]
    return real_contributors, fake_contributors


# ------------------ STREAMLIT UI ------------------

def main():
    pipeline, weights_dict = load_production_pipeline()

    st.title("📰 AI-Assisted Fake News Detection System")
    st.markdown(
        "A modular, leak-free machine learning system to analyze journalistic text and flag stylistic patterns of misinformation."
    )

    # Sidebar: Model Specs & Ethical Framing
    with st.sidebar:
        st.header("⚙️ System Architecture")
        st.info(
            "**Classifier:** TF-IDF (50,000 N-Grams) + Calibrated Regularized Logistic Regression\n\n"
            "**In-Domain Test F1:** 98.75%\n\n"
            "**In-Domain Accuracy:** 98.62%\n\n"
            "**Cross-Dataset F1 (LIAR):** 32.52%\n\n"
            "**Seed:** 42 (Reproducible)"
        )

        st.subheader("⚠️ Caveats & Ethical Framing")
        st.caption(
            "**Decision-Support Tool:** This system analyzes rhetorical style, vocabulary, and agency markers. "
            "It is designed to flag articles for human editorial review, NOT to deliver definitive verdicts on objective truth."
        )

        st.divider()
        st.subheader("🧪 Load Pre-Set Examples")
        sample_choice = st.selectbox(
            "Select a test sample:",
            [
                "Custom Input",
                "Sample 1: Real Political News (Senate Budget)",
                "Sample 2: Fake Sensationalist News (Viral Conspiracy)",
                "Sample 3: Real Diplomatic Report (Foreign Ministry)",
                "Sample 4: Fake Clickbait (Shocking Scandal)",
            ],
        )

    # Preset Text Samples
    preset_samples = {
        "Sample 1: Real Political News (Senate Budget)": (
            "The U.S. Senate passed a bipartisan budget resolution on Thursday to fund federal agencies "
            "and avoid a government shutdown. Leaders from both parties praised the fiscal agreement, noting "
            "that the legislation protects national defense priorities while addressing critical healthcare programs. "
            "The measure now heads to the president for signing into law."
        ),
        "Sample 2: Fake Sensationalist News (Viral Conspiracy)": (
            "BREAKING: Shocking secret documents expose corrupt establishment elites who were caught red-handed! "
            "You won't believe what the mainstream media is desperate to hide from the American public. "
            "Read more to discover the explosive truth they are terrified of exposing! Featured image via viral blog."
        ),
        "Sample 3: Real Diplomatic Report (Foreign Ministry)": (
            "Diplomats gathered in Geneva on Wednesday to resume bilateral talks on international trade agreements. "
            "The delegation announced that constructive progress had been achieved regarding agricultural tariffs, "
            "with further formal negotiations scheduled for next month."
        ),
        "Sample 4: Fake Clickbait (Shocking Scandal)": (
            "UNBELIEVABLE: Bombshell scandal erupts as whistleblowers blow the lid off government corruption! "
            "We have uncovered undeniable proof that the deep state planned this whole operation. "
            "Share this everywhere before it gets censored and deleted from the internet! Read more via PatriotNews."
        ),
    }

    default_text = preset_samples.get(sample_choice, "")

    # Input Box
    user_input = st.text_area(
        "Enter or paste the news article text below:",
        value=default_text,
        height=220,
        placeholder="Type or paste article body here...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

    if analyze_btn or (sample_choice != "Custom Input" and user_input):
        cleaned_text = clean_text_classical(user_input)
        word_count = len(cleaned_text.split())

        # Validation handling (Caveat 17: gracefully handle empty or very short inputs)
        if not user_input.strip():
            st.warning("⚠️ Please provide news article text to analyze.")
            return

        if word_count < 10:
            st.warning(
                f"⚠️ Input is too short ({word_count} words). "
                "Reliable classification requires at least 10 words of article body."
            )
            return

        # Inference
        probs = pipeline.predict_proba([cleaned_text])[0]
        prob_fake = float(probs[0])
        prob_real = float(probs[1])
        prediction = int(np.argmax(probs))

        st.divider()
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.success("### Prediction: REAL NEWS")
                st.metric("Confidence (Real)", f"{prob_real * 100:.1f}%")
            else:
                st.error("### Prediction: FAKE NEWS / MISINFORMATION")
                st.metric("Confidence (Fake)", f"{prob_fake * 100:.1f}%")

            # Confidence progress bar
            st.progress(prob_real)
            st.caption(f"Real Probability: {prob_real:.3f} | Fake Probability: {prob_fake:.3f}")

        with res_col2:
            st.subheader("💡 Key Linguistic Indicators")
            real_words, fake_words = get_contributing_words(cleaned_text, weights_dict)

            if prediction == 1 and real_words:
                st.write("**Top words supporting Real News classification:**")
                for w, c in real_words:
                    st.markdown(f"- `{w}` *(formal attribution / reporting weight: +{c:.2f})*")
            elif prediction == 0 and fake_words:
                st.write("**Top words supporting Fake News classification:**")
                for w, c in fake_words:
                    st.markdown(f"- `{w}` *(sensational / blog marker weight: {c:.2f})*")
            else:
                st.write("Balanced stylistic markers detected across vocabulary.")

        with st.expander("🔎 View Cleaned Text & Processing Details"):
            st.write(f"**Cleaned Tokens ({word_count} words):**")
            st.text(cleaned_text)


if __name__ == "__main__":
    main()
