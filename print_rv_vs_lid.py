import pandas as pd
import numpy as np
from scipy import stats

print("================================================================================")
print(f"{'Dataset':<25} | {'m (LID)':<15} | {'Unweighted RV'}")
print("================================================================================")

for dataset in ["deep-image-96-angular", "glove-100-angular", "sift-128-euclidean"]:
    try:
        df = pd.read_csv(f"research/rv_vs_lid_{dataset}.csv")
        log_target = np.log(df['ef_true'].replace(0, 1))
        
        rho_lid, _ = stats.spearmanr(df['m_LID'], log_target)
        rho_rv, _  = stats.spearmanr(df['RV_unw'], log_target)
        
        print(f"{dataset:<25} | {rho_lid:+.4f}         | {rho_rv:+.4f}")
    except Exception as e:
        print(f"Failed {dataset}: {e}")
print("================================================================================")
