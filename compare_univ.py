import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('research/univ_vs_raw.csv')
log_target = np.log(df['ef_true'].replace(0, 1))

print("============================================================")
print("Spearman Correlation vs log(ef_true)")
print("------------------------------------------------------------")

for col, desc in [("MR_raw_gauss", "Raw Gaussian (gamma=5.0)"), ("MR_univ_gauss", "Universal Gaussian (Queue-Normalized)")]:
    rho_log, p = stats.spearmanr(df[col], log_target)
    print(f"{desc:<40} | rho = {rho_log:+.4f}")
print("============================================================")
