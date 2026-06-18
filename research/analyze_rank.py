import subprocess
import pandas as pd
import numpy as np
from scipy import stats

datasets = ["deep-image-96-angular", "glove-100-angular", "sift-128-euclidean"]

print("================================================================================")
print(f"{'Dataset':<25} | {'Unweighted RV':<15} | {'Rank RV (y=5)':<15} | {'Rank RV (y=10)'}")
print("================================================================================")

for dataset in datasets:
    subprocess.run(["./build/test_rank_weight", dataset], check=True)
    df = pd.read_csv(f"research/rank_weight_{dataset}.csv")
    log_target = np.log(df['ef_true'].replace(0, 1))
    
    rho_unw, _ = stats.spearmanr(df['RV_unw'], log_target)
    rho_r5, _ = stats.spearmanr(df['RV_rank_5'], log_target)
    rho_r10, _ = stats.spearmanr(df['RV_rank_10'], log_target)
    
    print(f"{dataset:<25} | {rho_unw:+.4f}         | {rho_r5:+.4f}         | {rho_r10:+.4f}")

print("================================================================================")
