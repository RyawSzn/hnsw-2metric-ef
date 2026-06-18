import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

def main():
    parser = argparse.ArgumentParser(description="Visualize Dynamic Lookup Table")
    parser.add_argument("--RC_bins", type=int, default=10, help="Number of RC bins")
    parser.add_argument("--RV_bins", type=int, default=10, help="Number of RV bins")
    parser.add_argument("--dataset", type=str, default="glove-100-angular", help="Dataset name")
    args = parser.parse_args()

    csv_path = f"research/lookup_table_{args.RC_bins}x{args.RV_bins}.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Compute correlations from raw data
    rho_RC_str = ""
    rho_RV_str = ""
    
    diag_path = f"research/hardness_diagnostic_{args.dataset}.csv"
    if os.path.exists(diag_path):
        diag_df = pd.read_csv(diag_path)
        log_ef = np.log(diag_df['ef_true'].replace(0, 1))
        if 'RC' in diag_df.columns:
            rho_RC, _ = stats.spearmanr(diag_df['RC'], log_ef)
            rho_RC_str = f"  |  $\\rho = {rho_RC:+.3f}$"
        if 'RV_rank' in diag_df.columns:
            rho_RV, _ = stats.spearmanr(diag_df['RV_rank'], log_ef)
            rho_RV_str = f"  |  $\\rho = {rho_RV:+.3f}$"

    pivot_ef = df.pivot(index="RC_bin", columns="RV_bin", values="ef")
    pivot_recall = df.pivot(index="RC_bin", columns="RV_bin", values="actual_recall")

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
    
    axes[0].set_xlabel(f"Topological Rank Entrapment ($RV_{{Rank}}$){rho_RV_str}", fontsize=13, fontweight='bold')
    axes[0].set_ylabel(f"Relative Contrast ($RC$){rho_RC_str}", fontsize=13, fontweight='bold')

    axes[0].set_yticklabels(
        [f"RC{int(float(y.get_text()))}" for y in axes[0].get_yticklabels()], rotation=0
    )
    axes[0].set_xticklabels(
        [f"RV{int(float(x.get_text()))}" for x in axes[0].get_xticklabels()]
    )

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
    axes[1].set_xlabel(f"Topological Rank Entrapment ($RV_{{Rank}}$){rho_RV_str}", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("")

    axes[1].set_yticklabels(
        [f"RC{int(float(y.get_text()))}" for y in axes[1].get_yticklabels()], rotation=0
    )
    axes[1].set_xticklabels(
        [f"RV{int(float(x.get_text()))}" for x in axes[1].get_xticklabels()]
    )

    plt.tight_layout()
    out_path = f"research/figures/lookup_ef_heatmap_{args.RC_bins}x{args.RV_bins}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    print(f"Saved Visualization to: {out_path}")

if __name__ == "__main__":
    main()
