import subprocess
import pandas as pd
import numpy as np
from scipy import stats
import os

datasets = [
    "deep-image-96-angular",
    "glove-100-angular",
    "sift-128-euclidean"
]

print("================================================================================")
print(f"{'Dataset':<25} | {'Metric':<10} | {'RV_log Correlation (rho)'}")
print("================================================================================")

for dataset in datasets:
    subprocess.run(["./build/test_log_datasets", dataset], check=True)
    
    df = pd.read_csv(f"research/log_sweep_{dataset}.csv")
    log_target = np.log(df['ef_true'].replace(0, 1))
    
    rho, _ = stats.spearmanr(df['RV_log'], log_target)
            
    metric_type = "L2" if "euclidean" in dataset else "Cosine"
    print(f"{dataset:<25} | {metric_type:<10} | rho = {rho:+.4f}")

print("================================================================================")
