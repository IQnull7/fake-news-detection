"""Module 1: Load, inspect, and generate EDA plots for raw ISOT dataset."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import FIGURES_DIR, RESULTS_DIR
from src.preprocess import load_raw_data, get_data_summary

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_eda_figures(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class Balance Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().rename(index={1: "Real (1)", 0: "Fake (0)"})
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax, palette=["#2b8cbe", "#e41a1c"])
    ax.set_title("Class Balance in Raw ISOT Dataset", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height):,} ({height / len(df):.1%})",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            xytext=(0, 4),
            textcoords="offset points",
        )
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.savefig(output_dir / "class_balance.png", dpi=300)
    plt.close()

    # 2. Subject Column Distribution (Demonstrating Caveat 2 Leakage)
    fig, ax = plt.subplots(figsize=(10, 5))
    ct = pd.crosstab(df["subject"], df["label"]).rename(columns={1: "Real", 0: "Fake"})
    ct.plot(kind="barh", stacked=True, ax=ax, color=["#e41a1c", "#2b8cbe"])
    ax.set_title(
        "Subject Distribution by Class (Zero Overlap = Extreme Shortcut Leakage)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Number of Articles", fontsize=12)
    ax.set_ylabel("Subject Category", fontsize=12)
    ax.legend(title="Class", frameon=True)
    plt.savefig(output_dir / "subject_leakage.png", dpi=300)
    plt.close()

    # 3. Text Length Distribution (Word Counts, clipped at 99th percentile for clean plotting)
    df["word_count"] = df["text"].fillna("").apply(lambda t: len(str(t).split()))
    p99 = df["word_count"].quantile(0.99)
    df_filtered = df[df["word_count"] <= p99]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=df_filtered,
        x="word_count",
        hue="label",
        element="step",
        stat="density",
        common_norm=False,
        bins=50,
        palette=["#e41a1c", "#2b8cbe"],
        ax=ax,
    )
    ax.set_title(
        "Word Count Distribution by Class (Clipped at 99th Percentile)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Article Word Count", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(["Real", "Fake"], title="Class")
    plt.savefig(output_dir / "text_length_distribution.png", dpi=300)
    plt.close()

    # 4. Source Fingerprint Leakage: Reuters Appearance Rate
    reuters_pattern = r"reuters"
    df["has_reuters"] = df["text"].fillna("").str.contains(reuters_pattern, case=False, regex=True)
    leak_rate = df.groupby("label")["has_reuters"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    leak_rate_renamed = leak_rate.rename(index={1: "Real (1)", 0: "Fake (0)"})
    sns.barplot(
        x=leak_rate_renamed.index,
        y=leak_rate_renamed.values,
        hue=leak_rate_renamed.index,
        legend=False,
        ax=ax,
        palette=["#2b8cbe", "#e41a1c"],
    )
    ax.set_title(
        "Percentage of Articles Containing 'Reuters' (Source Leakage)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("% Containing 'Reuters'", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylim(0, 110)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            xytext=(0, 4),
            textcoords="offset points",
        )
    plt.savefig(output_dir / "reuters_leakage_rate.png", dpi=300)
    plt.close()

    print(f"EDA plots saved successfully to {output_dir}")


def main():
    print("Loading raw data...")
    df = load_raw_data()
    summary = get_data_summary(df)
    print("\nData Summary:")
    print(json.dumps(summary, indent=2))

    print("\nGenerating EDA figures...")
    generate_eda_figures(df)


if __name__ == "__main__":
    main()
