import pandas as pd
import numpy as np
from scipy import stats

datasets = [
    "deep-image-96-angular",
    "glove-100-angular"
]

print("================================================================================")
print(f"{'Dataset':<25} | {'Metric':<10} | {'Best Gamma':<10} | {'Max Spearman rho'}")
print("================================================================================")

for dataset in datasets:
    df = pd.read_csv(f"research/gamma_sweep_{dataset}.csv")
    log_target = np.log(df['ef_true'].replace(0, 1))
    
    cols = [c for c in df.columns if c.startswith('RV_gamma_')]
    best_gamma = None
    best_rho = 0.0
    for col in cols:
        gamma_val = col.split('_')[-1]
        rho, _ = stats.spearmanr(df[col], log_target)
        if abs(rho) > abs(best_rho):
            best_rho = rho
            best_gamma = gamma_val
            
    metric_type = "L2" if "euclidean" in dataset else "Cosine"
    print(f"{dataset:<25} | {metric_type:<10} | gamma={best_gamma:<5} | rho = {best_rho:+.4f}")

print("================================================================================")
