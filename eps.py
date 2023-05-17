import os
import argparse
import random
import time
import numpy
import torch

from dataset import Dataset
from kelpie import Kelpie
from data_poisoning import DataPoisoning
from criage import Criage
import yaml
import click
import pandas as pd
import numpy as np
import math
from link_prediction.models.model import *
from utils import *


# if os.path.exists(args.model_path):
#     ech(f'loading models from path: {args.model_path}')
#     model.load_state_dict(torch.load(args.model_path))
# else:
#     ech(f'model does not exists! {args.model_path}')

# ---------------------train---------------------
if int(args.run[0]):
    ech("Training model...")
    t = time.time()
    optimizer = Optimizer(model=model, hyperparameters=hyperparameters)
    optimizer.train(train_samples=dataset.train_samples, evaluate_every=10, #10 if args.method == "ConvE" else -1,
                    save_path=args.model_path,
                    valid_samples=dataset.valid_samples)
    print(f"Train time: {time.time() - t}")

# ---------------------test---------------------
model.eval()
if int(args.run[1]):
    ech("Evaluating model...")
    Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=True, folder=args.output_folder)

    ech("making facts to explain...")
    lis = []
    print("{:^15}\t{:^15}\t{:^15}\t{:^15}".format('relation', '#targets', '#triples', '#top_triples'))

    folders = [os.path.join(args.output_folder, d) for d in os.listdir(args.output_folder) if os.path.isdir(d)] + [args.output_folder]
    print(folders)
    for d in folders:
        f = os.path.join(d, 'filtered_ranks.csv')
        if os.path.exists(f):
            df = pd.read_csv(f, sep=';', header=None)
            df.columns = ['h', 'r', 't', 'hr', 'tr']
            try:
                size = len(dataset.rid2target[dataset.relation_name_2_id[d.split('/')[-1]]])
            except:
                size = len(dataset.entities)

            top_count = 0
            for i in range(len(df)):
                # if df.loc[i, 'tr'] <= math.ceil(size*0.05):
                if df.loc[i, 'tr'] == 1:
                    top_count += 1
                    lis.append('\t'.join([df.loc[i, 'h'], df.loc[i, 'r'], df.loc[i, 't']]))
            print("{:^15}\t{:^15}\t{:^15}\t{:^15}".format(d, size, len(df), top_count))

    with open(args.explain_path, 'w') as f:
        f.write('\n'.join(lis))
    # print(lis)
            

# ---------------------explain---------------------
if not int(args.run[2]):
    os.abort()
start_time = time.time()

ech("Reading facts to explain...")
with open(args.explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

# get the ids of the elements of the fact to explain and the perspective entity

prefilter = args.prefilter
relevance_threshold = args.relevance_threshold

if args.baseline is None:
    kelpie = Kelpie(model, dataset, hyperparameters, prefilter)
elif args.baseline == "data_poisoning":
    kelpie = DataPoisoning(model, dataset, hyperparameters, prefilter)
elif args.baseline == "criage":
    kelpie = Criage(model, dataset, hyperparameters)
elif args.baseline == "k1":
    kelpie = Kelpie(model, dataset, hyperparameters, prefilter, max_explanation_length=1)
else:
    kelpie = Kelpie(model, dataset, hyperparameters, prefilter)


fact = ('/m/01mvth',  '/people/person/nationality', '/m/09c7w0')
 # 'base', 
ix = 0
times = 3

exp_df = pd.DataFrame(columns=['to_explain', 'paths', 'length', 'AA', 'AB', 'BA', 'BB', 'CA', 'AC', 'CC', 'head', 'tail', 'path'])
rel_df = pd.DataFrame(columns=['to_explain', 'triples', 'length', 'A', 'T', 'B', 'C', 'truth', 'approx']) 

for fact in testing_facts:
    triple = Triple.from_fact(fact)
    origin_score = triple.origin_score()
    if base_rank[str(triple)] > 1:
        print('target fact is not remarkable, next...')
        continue
    ix += 1
    if ix > 100:
        break
    print('=' * 50)
    print(f"Explaining fact {ix} on {len(testing_facts)}: {str(triple)}")
    print('origin score', origin_score)

    explanations = kelpie.explain_necessary(sample_to_explain=triple,
                                            perspective="head",
                                            num_promising_samples=args.prefilter_threshold,
                                            l_max=1)
    print('=====================explanations=====================')
    for ix, exp in enumerate(explanations):
        print(ix, exp)

    for exp in explanations:
        exp_df.loc[len(exp_df)] = exp.metric
        rel_df.loc[len(rel_df)] = exp.head.metric
        rel_df.loc[len(rel_df)] = exp.tail.metric
        rel_df.loc[len(rel_df)] = exp.path.metric
        
        for i in range(times):
            exp.calculate_score()
            
            exp_df.loc[len(exp_df)] = exp.metric
            rel_df.loc[len(rel_df)] = exp.head.metric
            rel_df.loc[len(rel_df)] = exp.tail.metric
            rel_df.loc[len(rel_df)] = exp.path.metric

    if ix % 10 == 0:
        ech('explaination output:')
        end_time = time.time()
        print("Explain time: " + str(end_time - start_time) + " seconds")
        print('count_dic', count_dic)
        print('count_dic_mean', {k: np.mean(v) for k, v in count_dic.items()})

