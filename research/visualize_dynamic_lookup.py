import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats


def parse_bin_index(label, prefix):
    """Extract numeric bin index from interval string like '(0.03,0.17]' or numeric label."""
    label = str(label).strip()
    # Handle interval strings from pd.cut: "(0.03,0.17]"
    if label.startswith("("):
        # Extract the bin position from the label
        # We need to map interval strings back to their bin index
        return label
    # Numeric label
    try:
        return int(float(label))
    except ValueError:
        return label


def main():
    parser = argparse.ArgumentParser(description="Visualize Dynamic Lookup Table")
    parser.add_argument("--RC_bins", type=int, default=20, help="Number of RC bins")
    parser.add_argument("--RV_bins", type=int, default=20, help="Number of RV bins")
    parser.add_argument("--dataset", type=str, default="glove-100-angular", help="Dataset name")
    args = parser.parse_args()

    csv_path = f"research/lookup_table_{args.RC_bins}x{args.RV_bins}.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Detect column naming convention: 10x10 uses M_bin/RV_bin (numeric), 20x20+ uses RC_bin/RV_bin (intervals)
    # 25x25 uses M_bin/m_bin (numeric)
    rc_col = "RC_bin" if "RC_bin" in df.columns else "M_bin"
    rv_col = "RV_bin" if "RV_bin" in df.columns else "m_bin"

    # Check if bins are interval strings (20x20 style) or numeric (10x10 style)
    first_rc_val = str(df[rc_col].iloc[0]).strip()
    is_interval = first_rc_val.startswith("(")

    # Compute correlations from raw data
    rho_RC_str = ""
    rho_RV_str = ""

    diag_path = f"research/hardness_diagnostic_{args.dataset}.csv"
    if os.path.exists(diag_path):
        diag_df = pd.read_csv(diag_path)
        log_ef = np.log(diag_df["ef_true"].replace(0, 1))
        if "RC" in diag_df.columns:
            rho_RC, _ = stats.spearmanr(diag_df["RC"], log_ef)
            rho_RC_str = f"  |  $\\rho = {rho_RC:+.3f}$"
        if "RV_rank" in diag_df.columns:
            rho_RV, _ = stats.spearmanr(diag_df["RV_rank"], log_ef)
            rho_RV_str = f"  |  $\\rho = {rho_RV:+.3f}$"

    if is_interval:
        # For interval-based tables (20x20, 25x25), assign numeric bin indices
        # by sorting unique intervals by their lower bound
        rc_unique = sorted(df[rc_col].unique(), key=lambda x: float(str(x).split(",")[0].lstrip("(")))
        rv_unique = sorted(df[rv_col].unique(), key=lambda x: float(str(x).split(",")[0].lstrip("(")))
        rc_map = {v: i + 1 for i, v in enumerate(rc_unique)}
        rv_map = {v: i + 1 for i, v in enumerate(rv_unique)}
        df = df.copy()
        df["_rc_idx"] = df[rc_col].map(rc_map)
        df["_rv_idx"] = df[rv_col].map(rv_map)
        pivot_ef = df.pivot(index="_rc_idx", columns="_rv_idx", values="ef")
        pivot_recall = df.pivot(index="_rc_idx", columns="_rv_idx", values="actual_recall")
        rc_labels = [f"RC{i+1}" for i in range(len(rc_unique))]
        rv_labels = [f"RV{i+1}" for i in range(len(rv_unique))]
    else:
        # For numeric bin tables (10x10), pivot directly
        pivot_ef = df.pivot(index=rc_col, columns=rv_col, values="ef")
        pivot_recall = df.pivot(index=rc_col, columns=rv_col, values="actual_recall")
        rc_labels = [f"RC{int(x)}" for x in pivot_ef.index]
        rv_labels = [f"RV{int(x)}" for x in pivot_ef.columns]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

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
        f"Optimal 'ef' to Reach Target Recall ({args.RC_bins}x{args.RV_bins} Table)",
        fontsize=15,
    )

    axes[0].set_xlabel(f"Topological Rank Entrapment ($RV_{{Rank}}$){rho_RV_str}", fontsize=13, fontweight="bold")
    axes[0].set_ylabel(f"Relative Contrast ($RC$){rho_RC_str}", fontsize=13, fontweight="bold")

    axes[0].set_yticks(np.arange(len(rc_labels)) + 0.5)
    axes[0].set_yticklabels(rc_labels, rotation=0)
    axes[0].set_xticks(np.arange(len(rv_labels)) + 0.5)
    axes[0].set_xticklabels(rv_labels)

    min_recall = df["actual_recall"].min() - 0.005
    sns.heatmap(
        pivot_recall,
        ax=axes[1],
        annot=True,
        fmt=".4f",
        cmap="crest",
        vmin=min_recall,
        vmax=1.00,
        cbar_kws={"label": "Actual Recall Achieved"},
        annot_kws={"size": 11},
    )

    axes[1].set_title("Actual Recall Achieved", fontsize=15)
    axes[1].set_xlabel(f"Topological Rank Entrapment ($RV_{{Rank}}$){rho_RV_str}", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")

    axes[1].set_yticks(np.arange(len(rc_labels)) + 0.5)
    axes[1].set_yticklabels(rc_labels, rotation=0)
    axes[1].set_xticks(np.arange(len(rv_labels)) + 0.5)
    axes[1].set_xticklabels(rv_labels)

    plt.tight_layout()
    out_path = f"research/figures/lookup_ef_heatmap_{args.RC_bins}x{args.RV_bins}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    print(f"Saved Visualization to: {out_path}")


if __name__ == "__main__":
    main()
