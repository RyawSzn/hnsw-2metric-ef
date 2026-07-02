import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def compute_lid_from_dists(all_query_distances, k, num_queries, epsilon=1e-1):
    lids = np.zeros(num_queries)
    for i in range(num_queries):
        sorted_distances = np.sort(all_query_distances[i])[0:k]  
        r_k = sorted_distances[-1]
        
        if r_k == 0:
            lids[i] = np.nan
            continue
        else:
            safe_distances = np.maximum(sorted_distances / r_k, epsilon)
            log_values = np.log(safe_distances)
            lid = -1 / np.mean(log_values)
            if np.isnan(lid) or np.isinf(lid):
                lids[i] = np.nan
            else:
                lids[i] = lid
    return lids

def main():
    print("Loading score and recall data...")
    df = pd.read_csv('sift_score_recall.csv')
    
    print("Loading SIFT distances from HDF5 to compute LID...")
    hdf5_path = '/home/ryawszn/experiments/2metric/data/sift-128-euclidean.hdf5'
    with h5py.File(hdf5_path, 'r') as f:
        distances = np.array(f['distances'])
    
    num_queries = distances.shape[0]
    k_lid = 100 # Compute LID based on 100 nearest neighbors (from HDF5 dimensions)
    
    print("Computing LIDs...")
    lids = compute_lid_from_dists(distances, k_lid, num_queries)
    df['LID'] = lids
    
    # Drop rows where LID is NaN
    df = df.dropna(subset=['LID', 'score', 'recall'])
    
    # Calculate Correlations
    pearson_score, _ = pearsonr(df['score'], df['recall'])
    spearman_score, _ = spearmanr(df['score'], df['recall'])
    
    pearson_lid, _ = pearsonr(df['LID'], df['recall'])
    spearman_lid, _ = spearmanr(df['LID'], df['recall'])
    
    print(f"RV vs Recall -> Pearson: {pearson_score:.4f}, Spearman: {spearman_score:.4f}")
    print(f"LID vs Recall -> Pearson: {pearson_lid:.4f}, Spearman: {spearman_lid:.4f}")
    
    print("Plotting RV vs Recall...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='score', y='recall', alpha=0.3, color='blue', edgecolor=None)
    plt.title(f'RV vs Recall (SIFT Dataset | ef=10)\\nPearson: {pearson_score:.3f} | Spearman: {spearman_score:.3f}')
    plt.xlabel('RV')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig('sift_RV_vs_recall.png', dpi=300)
    plt.close()
    
    print("Plotting LID vs Recall...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='LID', y='recall', alpha=0.3, color='orange', edgecolor=None)
    plt.title(f'LID vs Recall (SIFT Dataset | ef=10)\\nPearson: {pearson_lid:.3f} | Spearman: {spearman_lid:.3f}')
    plt.xlabel('Local Intrinsic Dimensionality (LID)')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig('sift_LID_vs_recall.png', dpi=300)
    plt.close()
    
    print("Plots saved: sift_RV_vs_recall.png, sift_LID_vs_recall.png")

if __name__ == "__main__":
    main()
