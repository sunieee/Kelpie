import json
import os
from collections import defaultdict
import pandas as pd


folders = []
for model in ['complex', 'conve', 'transe']:
    for dataset in ['FB15k', 'WN18', 'FB15k-237', 'WN18RR']:    # , 'YAGO3-10'
        folder = f'{model}_{dataset}'
        folders.append(folder)
        
rx = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head'])
rx_h = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head'])
rx_t = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head'])
rx_subset = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head'])
rx_subset_h = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head'])
rx_subset_t = pd.DataFrame(0, columns=folders, index=['data_poisoning', 'criage', 'k1', 'kelpie', 'score4', 'score4.head'])


def process(folder):
    file = f'output_end_to_end_score4.json'
    valid_predictions = []
    if os.path.exists(f'out/{folder}/{file}'):
        with open(f'out/{folder}/{file}', 'r') as f:
            data = json.load(f)
            base_prediction2data = {','.join(t['prediction']) :t for t in data if 'prediction' in t}
            valid_predictions = [k for k, v in base_prediction2data.items() if len(v['explanation'][0]['triples']) > 3]

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

        dMRR = 0
        dMRR_subset = 0
        dMRR_h = 0
        dMRR_subset_h = 0
        dMRR_t = 0
        dMRR_subset_t = 0
        MRR = 0
        MRR_subset = 0
        MRR_h = 0
        MRR_subset_h = 0
        MRR_t = 0
        MRR_subset_t = 0
        for k, v in prediction2data.items():
            if 'prediction' not in v:
                continue
            if 'original' not in v:
                continue
            dMRR += v['dMRR']
            MRR += v['original']['MRR']
            dMRR_h += v['original']['MRR_head'] - v['new']['MRR_head']
            MRR_h += v['original']['MRR_head']
            dMRR_t += v['original']['MRR_tail'] - v['new']['MRR_tail']
            MRR_t += v['original']['MRR_tail']
            if k in valid_predictions:
                MRR_subset += v['original']['MRR']
                dMRR_subset += v['dMRR']
                MRR_subset_h += v['original']['MRR_head']
                dMRR_subset_h += v['original']['MRR_head'] - v['new']['MRR_head']
                MRR_subset_t += v['original']['MRR_tail']
                dMRR_subset_t += v['original']['MRR_tail'] - v['new']['MRR_tail']
                
        # MRR /= len(prediction2data)
        # if len(valid_predictions):
        #     MRR_subset /= len(valid_predictions)

        rx.loc[setting, folder] = round(dMRR / MRR, 3)
        rx_h.loc[setting, folder] = round(dMRR_h / MRR_h, 3)
        rx_t.loc[setting, folder] = round(dMRR_t / MRR_t, 3)
        
        if len(valid_predictions):
           rx_subset.loc[setting, folder] = round(dMRR_subset / MRR_subset, 3)
           rx_subset_h.loc[setting, folder] = round(dMRR_subset_h / MRR_subset_h, 3)
           rx_subset_t.loc[setting, folder] = round(dMRR_subset_t / MRR_subset_t, 3)

for folder in folders:
    process(folder)

rx.to_csv('out/rx.csv')
rx_subset.to_csv('out/rx_subset.csv')
rx_h.to_csv('out/rx_h.csv')
rx_subset_h.to_csv('out/rx_subset_h.csv')
rx_t.to_csv('out/rx_t.csv')
rx_subset_t.to_csv('out/rx_subset_t.csv')