import pandas as pd
import numpy as np
from scipy import stats
for dataset in ["deep-image-96-angular", "glove-100-angular", "sift-128-euclidean"]:
    try:
        df = pd.read_csv(f"research/final_rv_{dataset}.csv")
        log_target = np.log(df['ef_true'].replace(0, 1))
        rho_g, _ = stats.spearmanr(df['RV_gauss'], log_target)
        rho_r, _ = stats.spearmanr(df['RV_rank'], log_target)
        rho_h, _ = stats.spearmanr(df['RV_hub'], log_target)
        print(f"{dataset:<25} | {rho_g:+.4f}         | {rho_r:+.4f}         | {rho_h:+.4f}")
    except:
        pass
