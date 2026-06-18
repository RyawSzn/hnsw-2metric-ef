import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats

def main():
    parser = argparse.ArgumentParser(description="Visualize Dynamic Lookup Table")
    parser.add_argument("--dataset", type=str, default="glove-100-angular", help="Dataset name")
    parser.add_argument("--RC_bins", type=int, default=20, help="Number of RC bins")
    parser.add_argument("--RV_bins", type=int, default=20, help="Number of RV bins")
    args = parser.parse_args()

    csv_path = f"/home/ryawszn/experiments/2metric/lookup/lookup_table_{args.dataset}_{args.RC_bins}x{args.RV_bins}.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Compute correlations from raw data
    rho_RC_str = ""
    rho_RV_str = ""
    
    diag_path = f"/home/ryawszn/experiments/2metric/lookup/diagnostic_{args.dataset}_{args.RC_bins}x{args.RV_bins}.csv"
    if os.path.exists(diag_path):
        diag_df = pd.read_csv(diag_path)
        log_ef = np.log(diag_df['ef_true'].replace(0, 1))
        if 'RC' in diag_df.columns:
            rho_RC, _ = stats.spearmanr(diag_df['RC'], log_ef)
            rho_RC_str = f"  |  $\\rho = {rho_RC:+.3f}$"
        if 'RV_rank' in diag_df.columns:
            rho_RV, _ = stats.spearmanr(diag_df['RV_rank'], log_ef)
            rho_RV_str = f"  |  $\\rho = {rho_RV:+.3f}$"

    # Function to extract the lower bound from the interval string `(lower,upper]`
    def extract_lower(interval_str):
        if pd.isna(interval_str):
            return -1.0
        s = str(interval_str)
        try:
            return float(s.split(',')[0].replace('(', ''))
        except:
            return -1.0

    df['RC_sort'] = df['RC_bin'].apply(extract_lower)
    df['RV_sort'] = df['RV_bin'].apply(extract_lower)

    # Sort the dataframe by the lower bounds to ensure correct order in heatmap
    df = df.sort_values(['RC_sort', 'RV_sort'])

    pivot_ef = df.pivot(index="RC_bin", columns="RV_bin", values="ef")
    pivot_recall = df.pivot(index="RC_bin", columns="RV_bin", values="actual_recall")

    # Re-order the pivot table rows and columns by the sorted numerical lower bounds
    sorted_rc = sorted(df['RC_bin'].unique(), key=extract_lower)
    sorted_rv = sorted(df['RV_bin'].unique(), key=extract_lower)
    
    pivot_ef = pivot_ef.reindex(index=sorted_rc, columns=sorted_rv)
    pivot_recall = pivot_recall.reindex(index=sorted_rc, columns=sorted_rv)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    sns.heatmap(
        pivot_ef,
        ax=axes[0],
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Adaptive ef Required"},
        annot_kws={"size": 10},
    )

    axes[0].set_title(
        f"Optimal 'ef' to Reach Target Recall ({args.RC_bins}x{args.RV_bins}) - {args.dataset}",
        fontsize=16,
    )
    
    axes[0].set_xlabel(f"Topological Rank Entrapment ($RV_{{Rank}}$) Interval{rho_RV_str}", fontsize=14, fontweight='bold')
    axes[0].set_ylabel(f"Relative Contrast ($RC$) Interval{rho_RC_str}", fontsize=14, fontweight='bold')

    min_recall = df["actual_recall"].min() - 0.005 if not df.empty else 0.90
    sns.heatmap(
        pivot_recall,
        ax=axes[1],
        annot=True,
        fmt=".4f",
        cmap="crest",
        vmin=min_recall,
        vmax=1.00,
        cbar_kws={"label": "Actual Recall Achieved"},
        annot_kws={"size": 10},
    )

    axes[1].set_title(f"Actual Recall Achieved - {args.dataset}", fontsize=16)
    axes[1].set_xlabel(f"Topological Rank Entrapment ($RV_{{Rank}}$) Interval{rho_RV_str}", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("")

    plt.tight_layout()
    out_path = f"/home/ryawszn/experiments/2metric/lookup/lookup_ef_heatmap_{args.dataset}_{args.RC_bins}x{args.RV_bins}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    print(f"Saved Visualization to: {out_path}")

if __name__ == "__main__":
    main()
