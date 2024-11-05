import json
import os
from collections import defaultdict
import pandas as pd


folders = []
for model in ['complex', 'conve', 'transe']:
    for dataset in ['FB15k', 'WN18', 'FB15k-237', 'WN18RR', 'YAGO3-10']:
        folder = f'{model}_{dataset}'
        folders.append(folder)
        
dx = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head'])
rx = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head'])


def process(folder):
    for setting in ['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head']:
        suffix = setting + '4' if setting in ['data_poisoning', 'criage', 'k1', 'kelpie'] else setting
        file = f'output_end_to_end_{suffix}.json'
        if not os.path.exists(f'out/{folder}/{file}'):
            print(f'out/{folder}/{file} not exists')
            continue
        
        print('processing', folder, setting)
        with open(f'out/{folder}/{file}', 'r') as f:
            data = json.load(f)
            prediction2data = {','.join(t['prediction']) :t for t in data if 'prediction' in t}

        MRR = 0
        for k, v in prediction2data.items():
            if 'prediction' not in v:
                continue
            if 'original' not in v:
                continue
            MRR += v['original']['MRR']
            dx.loc[setting, folder] += v['dMRR']

        MRR /= len(prediction2data)
        dx.loc[setting, folder] = round(dx.loc[setting, folder] / len(prediction2data), 3)
        rx.loc[setting, folder] = round(dx.loc[setting, folder] / MRR, 3)

for folder in folders:
    process(folder)

dx.to_csv('out/dx.csv')
rx.to_csv('out/rx.csv')