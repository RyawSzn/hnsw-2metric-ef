import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('research/gamma_tuning_glove-100-angular.csv')
log_target = np.log(df['ef_true'].replace(0, 1))

cols = [c for c in df.columns if c.startswith('RV_gamma_')]

print("=" * 60)
print(f"{'Gamma (gamma)':<15} | {'Spearman Correlation vs log(ef_true)'}")
print("-" * 60)

best_gamma = None
best_rho = 0.0

for col in cols:
    gamma_val = col.split('_')[-1]
    rho, p = stats.spearmanr(df[col], log_target)
    print(f"gamma = {gamma_val:<8} | rho = {rho:+.4f}")
    if abs(rho) > abs(best_rho):
        best_rho = rho
        best_gamma = gamma_val

print("=" * 60)
print(f"Optimal Gamma: {best_gamma} (rho = {best_rho:+.4f})")
