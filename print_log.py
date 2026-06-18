import pandas as pd
import numpy as np
from scipy import stats

try:
    df = pd.read_csv("research/log_sweep_deep-image-96-angular.csv")
    rho, _ = stats.spearmanr(df['RV_log'], np.log(df['ef_true'].replace(0, 1)))
    print(f"deep-image-96-angular  | Cosine | rho = {rho:+.4f}")
except Exception:
    pass

try:
    df = pd.read_csv("research/log_sweep_sift-128-euclidean.csv")
    rho, _ = stats.spearmanr(df['RV_log'], np.log(df['ef_true'].replace(0, 1)))
    print(f"sift-128-euclidean     | L2     | rho = {rho:+.4f}")
except Exception:
    pass
