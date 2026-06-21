import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/ryawszn/experiments/2metric/lookup/research_categorizers_glove-100-angular.csv')

y = df['ef_true'].values

signals = {
    'RC': df['RC'].values,
    'd_mean': df['d_mean'].values,
    'd_ep': df['d_ep'].values,
    'm_LID': df['m_LID'].values,
    'RV_rank': df['RV_rank'].values
}

print("=== 1. Individual Spearman Correlations with ef_true ===")
for name, val in signals.items():
    rho, _ = stats.spearmanr(val, y)
    print(f"{name:>10}: {rho:+.4f}")

print("\n=== 2. Cross-Correlations with RV_rank (Lower magnitude = more independent) ===")
for name, val in signals.items():
    if name == 'RV_rank': continue
    rho, _ = stats.spearmanr(val, signals['RV_rank'])
    print(f"{name:>10} vs RV_rank: {rho:+.4f}")

def partial_spearman(x, y_tgt, z):
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y_tgt), stats.rankdata(z)
    def resid(a, b):
        lr = LinearRegression().fit(b.reshape(-1,1), a)
        return a - lr.predict(b.reshape(-1,1))
    return stats.pearsonr(resid(rx, rz), resid(ry, rz))[0]

print("\n=== 3. Partial Correlations (What the signal adds BEYOND RV_rank) ===")
for name, val in signals.items():
    if name == 'RV_rank': continue
    pc = partial_spearman(val, y, signals['RV_rank'])
    print(f"{name:>10} | RV_rank: {pc:+.4f}")

print("\n=== 4. 2D Joint Explanatory Power (R² via GBM) ===")
for name in ['RC', 'd_mean', 'd_ep', 'm_LID']:
    X = np.column_stack([signals[name], signals['RV_rank']])
    r2 = cross_val_score(GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42), X, y, cv=5, scoring='r2').mean()
    print(f"{name:>10} + RV_rank: R² = {r2:.4f}")

