import pandas as pd
from scipy.stats import spearmanr, pearsonr
import os

def main():
    csv_path = "research/predictor_comparison.csv"
    if not os.path.exists(csv_path):
        print("Error: CSV not found.")
        return

    df = pd.read_csv(csv_path)

    print("==========================================================================")
    print("      Correlation of Micro Difficulty (m) to the True Required 'ef'")
    print("      (Note: Higher True 'ef' means the query was harder / lower recall)")
    print("==========================================================================")
    print(f"{'Metric':<25} | {'Spearman (Rank)':<20} | {'Pearson (Linear)'}")
    print("-" * 74)

    metrics = [
        ('m_Spear (Greedy Search)', 'm_Spear'),
        ('m_Sphere (2-Hop Probe)', 'm_Sphere')
    ]

    for name, col in metrics:
        s_corr, _ = spearmanr(df[col], df['true_ef'])
        p_corr, _ = pearsonr(df[col], df['true_ef'])
        print(f"{name:<25} | {s_corr:>10.4f}           | {p_corr:>10.4f}")

    print("==========================================================================\n")

if __name__ == "__main__":
    main()
