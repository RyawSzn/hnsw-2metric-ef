import pandas as pd
import numpy as np
import h5py

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

def evaluate(dataset, csv_file, hdf5_path):
    df = pd.read_csv(csv_file)
    with h5py.File(hdf5_path, 'r') as f:
        distances = np.array(f['distances'])
    df['LID'] = compute_lid_from_dists(distances, 100, distances.shape[0])
    df = df.dropna(subset=['LID', 'score', 'recall'])
    
    # Define "Failed" queries as bottom 10%
    p10 = df['recall'].quantile(0.10)
    
    # 1. False Positive Rate (LID Blind Spots)
    lid_median = df['LID'].median()
    lid_blind_spots = df[(df['LID'] <= lid_median) & (df['recall'] <= p10)]
    
    # rv is score. In this setup, lower score = failure. 
    # RV "caught" it if RV is ALSO low (<= median)
    rv_median = df['score'].median()
    caught_by_rv = lid_blind_spots[lid_blind_spots['score'] <= rv_median]
    
    return len(lid_blind_spots), len(caught_by_rv)

print("Analyzing LID Blind Spots across datasets...")
sift_total, sift_caught = evaluate("SIFT", "research/sift_score_recall.csv", "/home/ryawszn/experiments/2metric/data/sift-128-euclidean.hdf5")
print(f"SIFT: Out of {sift_total} LID blind spots, RV caught {sift_caught} ({sift_caught/sift_total*100:.1f}%)")
