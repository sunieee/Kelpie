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
score_df = pd.DataFrame(columns=['explain', 'facts', 'length', 'origin', 'retrain', 'pt'])  # 'base', 

def get_max_explaination(fact_to_explain):
    head, relation, tail = fact_to_explain
    origin_score = get_origin_score(fact_to_explain)
    print(f"Explaining fact {i} on {len(testing_facts)}: {fact_to_explain}")
    head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                    dataset.get_id_for_relation_name(relation), \
                                    dataset.get_id_for_entity_name(tail)
    sample_to_explain = (head_id, relation_id, tail_id)

    rule_samples_with_relevance = kelpie.explain_necessary(sample_to_explain=sample_to_explain,
                                                            perspective="head",
                                                            num_promising_samples=args.prefilter_threshold,
                                                            l_max = 1)
    print('rule_samples_with_relevance', rule_samples_with_relevance)
    paths, score = rule_samples_with_relevance[0]

    p = Path(paths)
    p.get_retrain_score()
    relevance = p.relevance
    new_score_df = pd.DataFrame([[fact_to_explain, p.head, len(p.head), origin_score, p.retrain_head_score, score[0]],
                                [fact_to_explain, p.tail, len(p.tail), origin_score, p.retrain_tail_score, score[1]],
                                [fact_to_explain, p.triples, len(p.triples), origin_score, p.retrain_path_score, score[2]]],
                                        columns=score_df.columns)
    score_df = pd.concat([score_df, new_score_df])
    score_df.to_csv('score_df.csv')


for fact in testing_facts:
    get_max_explaination(fact)

ech('explaination output:')
end_time = time.time()
print("Explain time: " + str(end_time - start_time) + " seconds")
print('count_dic', count_dic)
print('count_dic_mean', {k: np.mean(v) for k, v in count_dic.items()})
relevance_df.to_csv('relevance.csv')
prelimentary_df.to_csv('prelimentary.csv')