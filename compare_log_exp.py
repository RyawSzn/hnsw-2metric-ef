import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('research/log_vs_exp.csv')
log_target = np.log(df['ef_true'].replace(0, 1))

print("============================================================")
print("Spearman Correlation vs log(ef_true)")
print("------------------------------------------------------------")

for col, desc in [("MR_gauss_5", "Gaussian / Heat Kernel (gamma=5.0)"), ("MR_log", "Logarithmic (Queue-Bound Log)")]:
    rho_log, p = stats.spearmanr(df[col], log_target)
    print(f"{desc:<35} | rho = {rho_log:+.4f}")
print("============================================================")
