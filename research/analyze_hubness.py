import subprocess
import pandas as pd
import numpy as np
from scipy import stats

print("================================================================================")
print(f"{'Dataset':<25} | {'Gaussian RV':<15} | {'Hub RV (b=0.5)':<15} | {'Hub RV (b=1.0)':<15}")
print("================================================================================")

dataset = "glove-100-angular"
subprocess.run(["./build/test_hubness", dataset], check=True)
df = pd.read_csv(f"research/hubness_weight_{dataset}.csv")
log_target = np.log(df['ef_true'].replace(0, 1))

rho_g, _ = stats.spearmanr(df['RV_gauss'], log_target)
rho_h05, _ = stats.spearmanr(df['RV_hub_05'], log_target)
rho_h10, _ = stats.spearmanr(df['RV_hub_10'], log_target)
rho_h20, _ = stats.spearmanr(df['RV_hub_20'], log_target)

print(f"{dataset:<25} | {rho_g:+.4f}         | {rho_h05:+.4f}         | {rho_h10:+.4f}")
print(f"                                                                 | Hub RV (b=2.0): {rho_h20:+.4f}")
print("================================================================================")
