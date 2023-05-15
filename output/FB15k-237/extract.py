import os
import pandas as pd
import json

folder = 'ConvE'
# readlines from out.log
with open(f'{folder}/out.log', 'r') as f:
    contend = f.read()

df = pd.DataFrame(columns=[f'{x}_{y}' for x in ['AA', 'BB', 'AB', 'BA', 'AC', 'CA', 'BB', 'CC'] for y in ['score', 'rank','best_score']])

for line in contend.split('\n'):
    if line.startswith('init scores: '):
        df.loc[len(df)] = json.loads(line.split('init scores: ')[-1].replace('\'', '\"'))

df.to_csv(f'{folder}/scores.csv')