import subprocess
import pandas as pd
import numpy as np
from scipy import stats

datasets = ["deep-image-96-angular", "glove-100-angular", "sift-128-euclidean"]

print("================================================================================")
print(f"{'Dataset':<25} | {'Best Gamma (y)':<15} | {'Max Spearman rho'}")
print("================================================================================")

for dataset in datasets:
    subprocess.run(["./build/hardness", dataset], check=True)
    df = pd.read_csv(f"research/sweep_rank_{dataset}.csv")
    log_target = np.log(df['ef_true'].replace(0, 1))
    
    cols = [c for c in df.columns if c.startswith('RV_rank_')]
    
    best_gamma = None
    best_rho = 0.0
    
    for col in cols:
        gamma_val = col.split('_')[-1]
        rho, _ = stats.spearmanr(df[col], log_target)
        if abs(rho) > abs(best_rho):
            best_rho = rho
            best_gamma = gamma_val
            
    print(f"{dataset:<25} | gamma = {best_gamma:<8} | rho = {best_rho:+.4f}")

print("================================================================================")
