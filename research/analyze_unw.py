import subprocess
import pandas as pd
import numpy as np
from scipy import stats

datasets = ["deep-image-96-angular", "glove-100-angular", "sift-128-euclidean"]

print("================================================================================")
print(f"{'Dataset':<25} | {'Unweighted RV':<15} | {'Log-Weighted RV'}")
print("================================================================================")

for dataset in datasets:
    subprocess.run(["./build/test_unweighted", dataset], check=True)
    df = pd.read_csv(f"research/unweighted_{dataset}.csv")
    log_target = np.log(df['ef_true'].replace(0, 1))
    
    rho_unw, _ = stats.spearmanr(df['RV_unw'], log_target)
    rho_log, _ = stats.spearmanr(df['RV_log'], log_target)
    
    print(f"{dataset:<25} | {rho_unw:+.4f}         | {rho_log:+.4f}")

print("================================================================================")
