import json
import os
from collections import defaultdict
import pandas as pd


# base_dir = 'out(alpha=0.05)'
# base_dir = 'outV7-T1'
base_dir = 'out'
folders = []
for model in ['complex', 'conve', 'transe']:
    for dataset in ['FB15k', 'WN18', 'FB15k-237', 'WN18RR']:    # , 'YAGO3-10'
        folder = f'{model}_{dataset}'
        folders.append(folder)
        
rx = pd.DataFrame(0, columns=folders, index=['criage', 'data_poisoning', 'k1', 'kelpie'])
rx_h = pd.DataFrame(0, columns=folders, index=['criage', 'data_poisoning', 'k1', 'kelpie'])
rx_t = pd.DataFrame(0, columns=folders, index=['criage', 'data_poisoning', 'k1', 'kelpie'])
rx_subset = pd.DataFrame(0, columns=folders, index=['criage', 'data_poisoning', 'k1', 'kelpie'])
rx_subset_h = pd.DataFrame(0, columns=folders, index=['criage', 'data_poisoning', 'k1', 'kelpie'])
rx_subset_t = pd.DataFrame(0, columns=folders, index=['criage', 'data_poisoning', 'k1', 'kelpie'])
fx = pd.DataFrame(0, columns=folders, index=['criage', 'data_poisoning', 'k1', 'kelpie'])
gx = pd.DataFrame(0, columns=folders, index=['criage', 'data_poisoning', 'k1', 'kelpie'])


def setting2path(folder, setting):
    suffix = setting
    if 'WN18' in folder:
        if setting in ['eXpath(0111)', 'eXpath(1011)', 'eXpath(1101)', 'eXpath(1110)', 'eXpath(1000)']:
            suffix = setting.replace('(', '(h')
        if setting == 'eXpath':
            suffix = 'eXpath(h)'
    else:
        if setting == 'eXpath':
            suffix = 'eXpath()'
    
    file = f'output_end_to_end_{suffix}4.json'
    if setting == 'eXpath1':
        if 'WN18' in folder:
            file = f'output_end_to_end_eXpath(h)1.json'
        else:
            file = f'output_end_to_end_eXpath()1.json'
    return f'{base_dir}/{folder}/{file}'


def process(folder):
    file = f'output_end_to_end_score4.json'
    valid_predictions = []
    if os.path.exists(f'{base_dir}/{folder}/{file}'):
        with open(f'{base_dir}/{folder}/{file}', 'r') as f:
            data = json.load(f)
            base_prediction2data = {','.join(t['prediction']) :t for t in data if 'prediction' in t}
            valid_predictions = [k for k, v in base_prediction2data.items() if v['dMRR'] > 0.1]

    for setting in ['criage', 'data_poisoning', 'k1', 'AnyBurlAttack', 'kelpie', \
                    'eXpath()', 'eXpath(h)', 'eXpath(t)', 'eXpath', \
                    'eXpath(0111)', 'eXpath(1011)', 'eXpath(1101)', 'eXpath(1110)', 'eXpath(1000)', 'eXpath1',
                    'eXpath1+criage', 'eXpath1+data_poisoning', 'eXpath1+k1', 'eXpath+kelpie',
                    'k1+criage', 'k1+data_poisoning', 'criage+data_poisoning']:
        print('processing', folder, setting)
        if '+' in setting:
            file1 = setting2path(folder, setting.split('+')[0])
            file2 = setting2path(folder, setting.split('+')[1])
            if not os.path.exists(file1) or not os.path.exists(file2):
                print(file1, file2, 'not exists')
                continue
            with open(file1, 'r') as f:
                data1 = json.load(f)
            with open(file2, 'r') as f:
                data2 = json.load(f)
            prediction2data1 = {','.join(t['prediction']) :t for t in data1 if 'prediction' in t}
            prediction2data2 = {','.join(t['prediction']) :t for t in data2 if 'prediction' in t}
            prediction2data = {}
            for k in set(prediction2data1.keys()) | set(prediction2data2.keys()):
                if prediction2data2.get(k, {'dMRR': 0})['dMRR'] > prediction2data1.get(k, {'dMRR': 0})['dMRR']:
                    prediction2data[k] = prediction2data2[k]
                else:
                    prediction2data[k] = prediction2data1[k]
        else:
            file = setting2path(folder, setting)
            if not os.path.exists(file):
                print(file, 'not exists')
                continue
            with open(file, 'r') as f:
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
            dMRR += v['dMRR'] # max(v['dMRR'], 0)
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

    for pair in [('eXpath1', 'criage'), ('eXpath1', 'data_poisoning'), ('eXpath1', 'k1'), ('eXpath', 'kelpie')]:
        setting1, setting2 = pair
        file1 = setting2path(folder, setting1)
        file2 = setting2path(folder, setting2)
        if not os.path.exists(file1) or not os.path.exists(file2):
            print(file1, file2, 'not exists')
            continue
        with open(file1, 'r') as f:
            data1 = json.load(f)
        with open(file2, 'r') as f:
            data2 = json.load(f)
        prediction2data1 = {','.join(t['prediction']) :t for t in data1 if 'prediction' in t}
        prediction2data2 = {','.join(t['prediction']) :t for t in data2 if 'prediction' in t}
        
        vxy = 0
        sxy = 0
        gxy_m = 0
        gxy_n = 0
        for k in prediction2data2:
            if prediction2data1[k]['dMRR'] > 0 or prediction2data2[k]['dMRR'] > 0:
                vxy += 1
                if prediction2data1[k]['dMRR'] > prediction2data2[k]['dMRR']:
                    sxy += 1
                    gxy_m += prediction2data1[k]['dMRR'] - prediction2data2[k]['dMRR']
                    gxy_n += prediction2data2[k]['dMRR']
        # fx.loc[setting2, folder] = round(sxy / vxy, 3)
        # gx.loc[setting2, folder] = round(gxy_m / gxy_n, 3)
        fx.loc[setting2, folder] = str(round(sxy / vxy * 1000) / 10) + '%'
        gx.loc[setting2, folder] = str(round(gxy_m / gxy_n * 1000) / 10) + '%'


for folder in folders:
    process(folder)

rx.to_csv(f'{base_dir}/rx.csv')
rx_subset.to_csv(f'{base_dir}/rx_subset.csv')
rx_h.to_csv(f'{base_dir}/rx_h.csv')
rx_subset_h.to_csv(f'{base_dir}/rx_subset_h.csv')
rx_t.to_csv(f'{base_dir}/rx_t.csv')
rx_subset_t.to_csv(f'{base_dir}/rx_subset_t.csv')
fx.to_csv(f'{base_dir}/fx.csv')
gx.to_csv(f'{base_dir}/gx.csv')