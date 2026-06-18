import subprocess
import pandas as pd
import numpy as np
from scipy import stats

datasets = ["deep-image-96-angular", "glove-100-angular", "sift-128-euclidean"]

print("================================================================================")
print(f"{'Dataset':<25} | {'Gaussian RV':<15} | {'Rank RV (y=10)':<15} | {'Hub RV (b=0.5)'}")
print("================================================================================")

for dataset in datasets:
    subprocess.run(["./build/test_all_rv", dataset], check=True)
    df = pd.read_csv(f"research/final_rv_{dataset}.csv")
    log_target = np.log(df['ef_true'].replace(0, 1))
    
    rho_g, _ = stats.spearmanr(df['RV_gauss'], log_target)
    rho_r, _ = stats.spearmanr(df['RV_rank'], log_target)
    rho_h, _ = stats.spearmanr(df['RV_hub'], log_target)
    
    print(f"{dataset:<25} | {rho_g:+.4f}         | {rho_r:+.4f}         | {rho_h:+.4f}")

print("================================================================================")
