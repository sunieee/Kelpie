
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('ConvE/score_df.csv', index_col=0)

# create a new dataframe by calculating the std group by columns 'to_explain' and 'explanation'
df_std = df.groupby(['to_explain', 'explanation']).std().reset_index()

print(df_std.head())
print(df_std.describe())

# make a distribution plot on column 'A', 'B', 'C' and 'T' for df_std
df_std[['A', 'T']].plot(kind='density', subplots=True, layout=(2, 1), sharex=False, figsize=(10, 10))
plt.savefig('ConvE/distribution.png')
