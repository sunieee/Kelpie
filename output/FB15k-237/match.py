import pandas as pd

f1 = 'ConvE-epoch=8/score_df.csv'
f2 = 'ConvE-0513/score_df.csv'

df1 = pd.read_csv(f1, index_col=0)
df2 = pd.read_csv(f2, index_col=0)


# merge df1 and df2 by columns  'explain' and 'facts', rename the columns
df = pd.merge(df1, df2, on=['explain', 'facts'], how='inner', suffixes=('_1', '_2'))
# save
df.to_csv('merged.csv', index=False)