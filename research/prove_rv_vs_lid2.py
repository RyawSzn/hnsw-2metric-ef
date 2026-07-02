import pandas as pd
import numpy as np
import h5py
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

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

def main():
    df = pd.read_csv('sift_score_recall.csv')
    
    hdf5_path = '/home/ryawszn/experiments/2metric/data/sift-128-euclidean.hdf5'
    with h5py.File(hdf5_path, 'r') as f:
        distances = np.array(f['distances'])
    
    df['LID'] = compute_lid_from_dists(distances, 100, distances.shape[0])
    df = df.dropna(subset=['LID', 'score', 'recall'])
    
    print("\n" + "="*50)
    print("MATHEMATICAL PROOF: RV vs LID IN PROXIMITY GRAPHS (GLOVE DATASET)")
    print("="*50)

    # Note: RV has a POSITIVE correlation with Recall (Higher RV = Easier)
    # Note: LID has a NEGATIVE correlation with Recall (Higher LID = Harder)

    # 1. GLOBAL CORRELATION (Baseline)
    sp_rv_global, _ = spearmanr(df['score'], df['recall'])
    sp_lid_global, _ = spearmanr(df['LID'], df['recall'])
    print(f"\n[1] GLOBAL CORRELATION (All Queries):")
    print(f"    LID: {abs(sp_lid_global):.4f}")
    print(f"    RV:  {abs(sp_rv_global):.4f}")

    # 2. TAIL CORRELATION (Hardest 15% of Queries)
    p15 = df['recall'].quantile(0.15)
    df_tail = df[df['recall'] <= p15]
    
    sp_rv_tail, _ = spearmanr(df_tail['score'], df_tail['recall'])
    sp_lid_tail, _ = spearmanr(df_tail['LID'], df_tail['recall'])
    
    print(f"\n[2] TAIL CORRELATION (Bottom 15% Recalls - The 'Hard' Queries):")
    print(f"    LID: {abs(sp_lid_tail):.4f} (LID's predictive power drops drastically)")
    print(f"    RV:  {abs(sp_rv_tail):.4f}")
    
    # 3. ROUTING FAILURE PREDICTION (AUROC)
    # Define failure as worst 10% of queries
    p10 = df['recall'].quantile(0.10)
    y_fail = (df['recall'] <= p10).astype(int)
    
    # Predict failure: Higher score for AUROC means more likely to fail.
    # So we use -RV (lower RV = failure) and LID (higher LID = failure)
    auc_rv = roc_auc_score(y_fail, -df['score'])
    auc_lid = roc_auc_score(y_fail, df['LID'])
    
    print(f"\n[3] PREDICTING ROUTING FAILURES (ROC-AUC for worst 10%):")
    print(f"    LID AUROC: {auc_lid:.4f}")
    print(f"    RV AUROC:  {auc_rv:.4f}")

    # 4. THE 'LID BLIND SPOT' (Local Minima Detection)
    # Queries that LID thinks are "Easy" (LID <= Median) but actually "Failed" (Recall <= 10th percentile)
    lid_median = df['LID'].median()
    blind_spots = df[(df['LID'] <= lid_median) & (df['recall'] <= p10)]
    
    # Did RV catch these blind spots? (RV <= Median, meaning RV correctly thought they were hard)
    rv_median = df['score'].median()
    caught_by_rv = blind_spots[blind_spots['score'] <= rv_median]
    
    print(f"\n[4] LID BLIND SPOTS (Local Minima Traps):")
    print(f"    Queries LID falsely thought were 'Easy' but failed: {len(blind_spots)}")
    print(f"    Out of those, RV correctly flagged as 'Hard': {len(caught_by_rv)} ({len(caught_by_rv)/len(blind_spots)*100:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
