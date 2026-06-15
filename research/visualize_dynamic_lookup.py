import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Visualize Dynamic Lookup Table")
    parser.add_argument("--M_bins", type=int, default=5, help="Number of M bins")
    parser.add_argument("--m_bins", type=int, default=5, help="Number of m bins")
    args = parser.parse_args()

    csv_path = f"research/lookup_table_{args.M_bins}x{args.m_bins}.csv"
    if not os.path.exists(csv_path):
        print(
            f"Error: {csv_path} not found. Did you run the C++ generator with these dimensions?"
        )
        return

    df = pd.read_csv(csv_path)

    pivot_ef = df.pivot(index="M_bin", columns="m_bin", values="ef")
    pivot_recall = df.pivot(index="M_bin", columns="m_bin", values="actual_recall")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # --- Plot 1: The Target EF Heatmap ---
    sns.heatmap(
        pivot_ef,
        ax=axes[0],
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Adaptive ef Required"},
        annot_kws={"size": 11},
    )

    axes[0].set_title(
        f"Optimal 'ef' to Reach Target Recall ({args.M_bins}x{args.m_bins} Table)",
        fontsize=15,
    )
    axes[0].set_xlabel("Local Ambiguity ($m_{LID}$) Decile -> Harder", fontsize=12)
    axes[0].set_ylabel("Relative Contrast ($M_{RC}$) Decile -> Harder", fontsize=12)

    axes[0].set_yticklabels(
        [f"M{int(float(y.get_text()))}" for y in axes[0].get_yticklabels()], rotation=0
    )
    axes[0].set_xticklabels(
        [f"m{int(float(x.get_text()))}" for x in axes[0].get_xticklabels()]
    )

    # --- Plot 2: The Actual Recall Heatmap ---
    # Setting the min threshold dynamically so the colors highlight variations nicely
    min_recall = df["actual_recall"].min() - 0.005
    max_recall = 1.00

    # Notice fmt=".4f" for the high precision display you requested!
    sns.heatmap(
        pivot_recall,
        ax=axes[1],
        annot=True,
        fmt=".4f",
        cmap="crest",
        vmin=min_recall,
        vmax=max_recall,
        cbar_kws={"label": "Actual Recall Achieved"},
        annot_kws={"size": 11},
    )

    axes[1].set_title("Actual Recall Achieved (4-decimal precision)", fontsize=15)
    axes[1].set_xlabel("Local Ambiguity ($m_{LID}$) Decile -> Harder", fontsize=12)
    axes[1].set_ylabel("")

    axes[1].set_yticklabels(
        [f"M{int(float(y.get_text()))}" for y in axes[1].get_yticklabels()], rotation=0
    )
    axes[1].set_xticklabels(
        [f"m{int(float(x.get_text()))}" for x in axes[1].get_xticklabels()]
    )

    plt.tight_layout()
    out_path = f"research/figures/lookup_ef_heatmap_{args.M_bins}x{args.m_bins}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    print(f"Saved Visualization to: {out_path}")


if __name__ == "__main__":
    main()
