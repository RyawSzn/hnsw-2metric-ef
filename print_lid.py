import pandas as pd
import numpy as np
from scipy import stats

print("======================================================================")
print(f"{'Dataset':<25} | {'Metric':<10} | {'m (LID)':<10} | {'RV_log'}")
print("======================================================================")

for dataset in ["glove-100-angular", "deep-image-96-angular", "sift-128-euclidean"]:
    try:
        df = pd.read_csv(f"research/log_sweep_{dataset}.csv")
        rho_rv, _ = stats.spearmanr(df['RV_log'], np.log(df['ef_true'].replace(0, 1)))
        rho_m, _  = stats.spearmanr(df['m_LID'], np.log(df['ef_true'].replace(0, 1)))
        metric = "L2" if "euclidean" in dataset else "Cosine"
        print(f"{dataset:<25} | {metric:<10} | {rho_m:+.4f}    | {rho_rv:+.4f}")
    except Exception as e:
        pass
print("======================================================================")
