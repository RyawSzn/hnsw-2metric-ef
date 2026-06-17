import sys
import pandas as pd

def simulate_2d_table(df, f1, f2, bins=20, safety_percentile=0.90):
    df = df.copy()
    try:
        df['b1'] = pd.qcut(df[f1], q=bins, labels=False, duplicates='drop')
        df['b2'] = pd.qcut(df[f2], q=bins, labels=False, duplicates='drop')
    except ValueError:
        df['b1'] = pd.cut(df[f1], bins=bins, labels=False)
        df['b2'] = pd.cut(df[f2], bins=bins, labels=False)

    table = df.groupby(['b1', 'b2'])['ef_true'].quantile(safety_percentile).fillna(df['ef_true'].max()).to_dict()
    df['ef_pred'] = df.apply(lambda r: table.get((r['b1'], r['b2']), df['ef_true'].max()), axis=1)

    return df['ef_pred'].mean(), (df['ef_pred'] >= df['ef_true']).mean()

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
        
    df = pd.read_csv(sys.argv[1])
    print("Simulating 2D Routing Tables (20x20 grid, 90th percentile safety margin)")
    print("=" * 60)
    print(f"{'2D Features':<20} | {'Average EF':<15} | {'Success Rate'}")
    print("-" * 60)

    combos = [("M", "m"), ("M", "M_R"), ("m", "M_R")]
    for f1, f2 in combos:
        avg_ef, success_rate = simulate_2d_table(df, f1, f2)
        print(f"{f1} + {f2:<14} | {avg_ef:<15.1f} | {success_rate:.2%}")

if __name__ == '__main__':
    main()
