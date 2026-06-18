import subprocess
import pandas as pd
import numpy as np
from scipy import stats

datasets = ["deep-image-96-angular", "glove-100-angular", "sift-128-euclidean"]

print("================================================================================")
print(f"{'Dataset':<25} | {'m (LID)':<15} | {'Rank RV (y=10)':<15}")
print("================================================================================")

for dataset in datasets:
    subprocess.run(["./build/test_final_rank", dataset], check=True)
    df = pd.read_csv(f"research/final_rank_{dataset}.csv")
    log_target = np.log(df['ef_true'].replace(0, 1))
    
    rho_lid, _ = stats.spearmanr(df['m_LID'], log_target)
    rho_r, _ = stats.spearmanr(df['RV_rank'], log_target)
    
    print(f"{dataset:<25} | {rho_lid:+.4f}         | {rho_r:+.4f}")

print("================================================================================")
