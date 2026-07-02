import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

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
    print("Loading GloVe data...")
    df = pd.read_csv('glove_score_recall.csv')
    with h5py.File('/home/ryawszn/experiments/2metric/data/glove-100-angular.hdf5', 'r') as f:
        distances = np.array(f['distances'])
    
    df['LID'] = compute_lid_from_dists(distances, 50, distances.shape[0])
    df = df.dropna(subset=['LID', 'score', 'recall'])
    
    # Ranks (0=Easy, 100=Hard)
    df['LID_Hardness'] = df['LID'].rank(pct=True) * 100
    df['RV_Hardness'] = df['score'].rank(pct=True, ascending=False) * 100
    
    # Isolate Failed Queries (Worst 10%)
    p10_recall = df['recall'].quantile(0.10)
    failed_queries = df[df['recall'] <= p10_recall].copy()
    
    # Categorize Quadrants
    conditions = [
        (failed_queries['LID_Hardness'] > 50) & (failed_queries['RV_Hardness'] > 50),
        (failed_queries['LID_Hardness'] <= 50) & (failed_queries['RV_Hardness'] > 50),
        (failed_queries['LID_Hardness'] > 50) & (failed_queries['RV_Hardness'] <= 50),
        (failed_queries['LID_Hardness'] <= 50) & (failed_queries['RV_Hardness'] <= 50)
    ]
    choices = [
        'Both Caught It', 
        'LID Blind Spot (RV Caught It)', 
        'RV Blind Spot (LID Caught It)', 
        'Both Missed It'
    ]
    failed_queries['Quadrant'] = np.select(conditions, choices, default='Other')
    
    counts = failed_queries['Quadrant'].value_counts()
    
    # Plotting
    plt.figure(figsize=(11, 9))
    sns.set_style("whitegrid")
    
    palette = {
        'Both Caught It': '#95a5a6',                 # Grey
        'LID Blind Spot (RV Caught It)': '#2ecc71',  # Green
        'RV Blind Spot (LID Caught It)': '#3498db',  # Blue
        'Both Missed It': '#e74c3c'                  # Red
    }
    
    ax = sns.scatterplot(
        data=failed_queries, 
        x='LID_Hardness', 
        y='RV_Hardness', 
        hue='Quadrant',
        palette=palette,
        s=90, alpha=0.8, edgecolor='w', linewidth=0.5
    )
    
    # Quadrant Dividers
    plt.axvline(50, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8)
    plt.axhline(50, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8)
    
    # Custom Annotations with Counts
    def add_annotation(x, y, text, count, color):
        plt.text(x, y, f"{text}\\n(n={count})", ha='center', va='center', 
                 fontsize=13, fontweight='bold', color=color, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

    add_annotation(75, 75, "Both Caught It", counts.get('Both Caught It', 0), '#7f8c8d')
    add_annotation(25, 75, "LID Blind Spot\\n(Caught by RV)", counts.get('LID Blind Spot (RV Caught It)', 0), '#27ae60')
    add_annotation(75, 25, "RV Blind Spot\\n(Caught by LID)", counts.get('RV Blind Spot (LID Caught It)', 0), '#2980b9')
    add_annotation(25, 25, "Both Missed It", counts.get('Both Missed It', 0), '#c0392b')
    
    plt.title('Orthogonality Proof: RV vs LID on Failed Queries\\nGloVe Dataset (ef=50, k=50)', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('LID Hardness Percentile (0=Easy, 100=Hard)\\n<-- RV Blind Spots caught here     |     LID Normal operating zone -->', fontsize=12, labelpad=15)
    plt.ylabel('RV Hardness Percentile (0=Easy, 100=Hard)\\n<-- LID Blind Spots caught here     |     RV Normal operating zone -->', fontsize=12, labelpad=15)
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), frameon=True)
    plt.tight_layout()
    
    filename = 'Orthogonal_Proof_RV_vs_LID.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to research/{filename}")

if __name__ == "__main__":
    main()
