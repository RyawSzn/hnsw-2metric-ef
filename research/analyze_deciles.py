import pandas as pd
import numpy as np

df = pd.read_csv('/home/ryawszn/experiments/2metric/lookup/research_categorizers_glove-100-angular.csv')

df['rv_decile'] = pd.qcut(df['RV_rank'], 10, labels=False)
df['d_ep_decile'] = pd.qcut(df['d_ep'], 10, labels=False)
df['rc_decile'] = pd.qcut(df['RC'], 10, labels=False)

print("Variance inside the hardest RV bin (decile 0):", df[df.rv_decile == 0]['ef_true'].std())
hard_rv = df[df.rv_decile == 0]

print("Splitting the hardest RV bin by RC deciles:")
print(hard_rv.groupby(pd.qcut(hard_rv['RC'], 5, duplicates='drop'))['ef_true'].mean())

print("\nSplitting the hardest RV bin by d_ep deciles:")
print(hard_rv.groupby(pd.qcut(hard_rv['d_ep'], 5, duplicates='drop'))['ef_true'].mean())

print("\nSplitting the hardest RV bin by d_mean deciles:")
print(hard_rv.groupby(pd.qcut(hard_rv['d_mean'], 5, duplicates='drop'))['ef_true'].mean())

print("\nSplitting the hardest RV bin by m_LID deciles:")
print(hard_rv.groupby(pd.qcut(hard_rv['m_LID'], 5, duplicates='drop'))['ef_true'].mean())
