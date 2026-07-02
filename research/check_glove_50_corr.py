import pandas as pd
import numpy as np
import h5py
from scipy.stats import pearsonr, spearmanr

def compute_lid_from_dists(all_query_distances, k, num_queries, epsilon=1e-1):
    lids = np.zeros(num_queries)
    for i in range(num_queries):
        sorted_distances = np.sort(all_query_distances[i])[0:k]  
        r_k = sorted_distances[-1]
        if r_k == 0:
            lids[i] = np.nan
        else:
            safe_distances = np.maximum(sorted_distances / r_k, epsilon)
            log_values = np.log(safe_distances)
            lid = -1 / np.mean(log_values)
            lids[i] = np.nan if np.isnan(lid) or np.isinf(lid) else lid
    return lids

df = pd.read_csv('glove_score_recall.csv')
with h5py.File('/home/ryawszn/experiments/2metric/data/glove-100-angular.hdf5', 'r') as f:
    distances = np.array(f['distances'])

df['LID'] = compute_lid_from_dists(distances, 50, distances.shape[0])
df = df.dropna(subset=['LID', 'score', 'recall'])

p_rv, _ = pearsonr(df['score'], df['recall'])
s_rv, _ = spearmanr(df['score'], df['recall'])

p_lid, _ = pearsonr(df['LID'], df['recall'])
s_lid, _ = spearmanr(df['LID'], df['recall'])

print(f"--- GLOVE (ef=50, k=50) GLOBAL CORRELATION ---")
print(f"RV vs Recall  -> Pearson: {p_rv:.4f} | Spearman: {s_rv:.4f}")
print(f"LID vs Recall -> Pearson: {p_lid:.4f} | Spearman: {s_lid:.4f}")
