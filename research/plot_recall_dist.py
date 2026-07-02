import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Read the data
    df = pd.read_csv('deepIm_recalls.csv')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Create histogram and KDE
    sns.histplot(df['recall'], bins=20, kde=True, color='skyblue', edgecolor='black')
    
    plt.title('Recall Distribution of Queries\\nDataset: deep-image-96-angular | ef=250', fontsize=16)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Number of Queries', fontsize=14)
    
    # Add vertical line for average recall
    avg_recall = df['recall'].mean()
    plt.axvline(avg_recall, color='red', linestyle='dashed', linewidth=2, label=f'Average Recall: {avg_recall:.4f}')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('deepIm_recall_distribution.png', dpi=300)
    print(f"Plot saved to research/deepIm_recall_distribution.png")
    print(f"Average Recall: {avg_recall:.4f}")

if __name__ == "__main__":
    main()
