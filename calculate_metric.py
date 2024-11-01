import json
import os
from collections import defaultdict
import pandas as pd


folders = []
for model in ['complex', 'conve', 'transe']:
    for dataset in ['FB15k', 'WN18', 'FB15k-237', 'WN18RR', 'YAGO3-10']:
        folder = f'{model}_{dataset}'
        folders.append(folder)
        
dx = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'R4', 'GA4', 'R4.head', 'R5'])
rx = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'R4', 'GA4', 'R4.head', 'R5'])
fx = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'R4', 'GA4', 'R4.head', 'R5'])
gx = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'R4', 'GA4', 'R4.head', 'R5'])


def process(folder):
    if not os.path.exists(f'out/{folder}/output_end_to_end_kelpie4.json'):
        prediction2data_base = None
        return
    else:
        with open(f'out/{folder}/output_end_to_end_kelpie4.json', 'r') as f:
            data = json.load(f)
            prediction2data_base = {','.join(t['prediction']) :t for t in data if 'prediction' in t}

    for setting in ['data_poisoning', 'criage', 'k1', 'kelpie', 'R4', 'GA4', 'R4.head', 'R5']:
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
        valid_count = 0
        for k, v in prediction2data.items():
            if 'prediction' not in v:
                continue
            if k not in prediction2data_base:
                continue
            MRR += 1/v['old_rank']
            dMRR = 1/v['old_rank'] - 1/v['new_rank']
            dx.loc[setting, folder] += dMRR
            base = prediction2data_base[k]
            dMRR_base = 1/base['old_rank'] - 1/base['new_rank']
            if dMRR > dMRR_base:
                fx.loc[setting, folder] += 1
            if dMRR >0 or dMRR_base > 0:
                valid_count += 1

            gx.loc[setting, folder] += max(dMRR, dMRR_base) - dMRR_base

        MRR /= len(prediction2data)
        dx.loc[setting, folder] = round(dx.loc[setting, folder] / len(prediction2data), 3)
        rx.loc[setting, folder] = round(dx.loc[setting, folder] / MRR, 3)
        fx.loc[setting, folder] = round(fx.loc[setting, folder] / valid_count, 3)
        gx.loc[setting, folder] = round(gx.loc[setting, folder] / len(prediction2data) / MRR, 3)

for folder in folders:
    process(folder)

dx.to_csv('out/dx.csv')
rx.to_csv('out/rx.csv')
fx.to_csv('out/fx.csv')
gx.to_csv('out/gx.csv')