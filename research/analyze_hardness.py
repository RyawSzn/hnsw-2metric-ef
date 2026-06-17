#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from scipy import stats

def spearman_report(df, target_col="ef_true"):
    log_target = np.log(df[target_col].replace(0, 1))
    signals = ["M", "m", "RV"]
    labels = {
        "M": "M (relative contrast, distance-based global)",
        "m": "m (LID / Hill estimator, distance-based local)",
        "RV": "RV (Gaussian-Weighted Revisit Ratio, graph-topology local)"
    }

    print("=" * 78)
    print(f"Spearman rank correlation against {target_col} and log({target_col})")
    print("=" * 78)
    for s in signals:
        rho_raw, p_raw = stats.spearmanr(df[s], df[target_col])
        rho_log, p_log = stats.spearmanr(df[s], log_target)
        print(f"{labels[s]:60s}  rho(log)={rho_log:+.4f} (p={p_log:.2e})")
    print()

def regression_r2(df, target_col="ef_true"):
    y = np.log(df[target_col].replace(0, 1)).values

    def fit_r2(X):
        X = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("=" * 78)
    print("OLS R^2 for log(ef_true), 2D Combinations")
    print("=" * 78)
    r2_Mm = fit_r2(df[["M", "m"]].values)
    r2_MRV = fit_r2(df[["M", "RV"]].values)
    r2_mRV = fit_r2(df[["m", "RV"]].values)

    print(f"  Combo 1: M + m    : R^2 = {r2_Mm:.4f}")
    print(f"  Combo 2: M + RV   : R^2 = {r2_MRV:.4f}")
    print(f"  Combo 3: m + RV   : R^2 = {r2_mRV:.4f}")
    print()
    best = max((r2_Mm, "M + m"), (r2_MRV, "M + RV"), (r2_mRV, "m + RV"))
    print(f"Winner: {best[1]} with R^2 = {best[0]:.4f}")
    print("=" * 78)
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_hardness_diagnostic.py <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    n_total = len(df)
    n_capped = (df["ef_true"] >= df["ef_true"].max()).sum()
    print(f"Loaded {n_total} queries from {csv_path}")
    if n_capped > 0:
        print(f"WARNING: {n_capped} queries ({n_capped/n_total:.1%}) hit max_ef cap.\n")

    spearman_report(df)
    regression_r2(df)

if __name__ == "__main__":
    main()
