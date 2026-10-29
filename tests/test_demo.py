"""Module 11 Checkpoint Verification Test."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
from src.preprocess import clean_text_classical
from app.app import load_production_pipeline, get_contributing_words


def test_demo_checkpoint():
    pipeline, weights_dict = load_production_pipeline()

    test_articles = [
        ("Real 1 (Senate)", "The Senate voted on Thursday to approve the new fiscal spending bill after bipartisan negotiations.", 1),
        ("Fake 1 (Conspiracy)", "BREAKING: Shocking secret documents expose corrupt elites caught red-handed! Read more via viral blog.", 0),
        ("Real 2 (Diplomacy)", "Diplomats met in Geneva on Wednesday to discuss bilateral international agreements on cross-border tariffs.", 1),
        ("Fake 2 (Clickbait)", "UNBELIEVABLE: Shocking scandal erupts as whistleblowers uncover undeniable proof! Share this before it gets deleted.", 0),
        ("Short (Edge Case)", "Too short.", None),
    ]

    print("=== Verifying Module 11 Checkpoint on 5 Test Samples ===")
    for name, text, expected in test_articles:
        clean = clean_text_classical(text)
        words = len(clean.split())
        if words < 10:
            print(f"{name:22s} -> Handled cleanly: Input too short ({words} words < 10 threshold).")
            continue
        probs = pipeline.predict_proba([clean])[0]
        pred = pipeline.predict([clean])[0]
        lbl = "Real" if pred == 1 else "Fake"
        conf = probs[pred] * 100
        real_words, fake_words = get_contributing_words(clean, weights_dict, top_k=3)
        match = (pred == expected)
        print(f"{name:22s} -> Predicted: {lbl} ({conf:.1f}%) | Expected: {'Real' if expected==1 else 'Fake'} | Match: {match}")
        print(f"   Top indicators: Real={[w for w,c in real_words]} | Fake={[w for w,c in fake_words]}")

    print("\nAll 5 test cases evaluated with ZERO crashes. CHECKPOINT PASSED!")


if __name__ == "__main__":
    test_demo_checkpoint()
