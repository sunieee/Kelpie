import os
import argparse
import random
import time
import numpy
import torch
import yaml
import click
import pandas as pd
import numpy as np
import math
from datetime import datetime

from kelpie import Kelpie
from data_poisoning import DataPoisoning
from criage import Criage

from link_prediction.models.model import *
from link_prediction.evaluation.evaluation import Evaluator
from utils import *

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
def ignore_triple(triple):
    h, r, t = triple
    if r in ['hasmethod', 'hasoperation']:
        return True
    if t in ['p1962']:
        return True
    return False


model.eval()
if int(args.run[1]):
    ech("Evaluating model...")
    Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=True, folder=args.output_folder)

    ech("making facts to explain...")
    lis = []
    print("{:^15}\t{:^15}\t{:^15}\t{:^15}".format('relation', '#targets', '#triples', '#top_triples'))
    df = pd.read_csv(os.path.join(args.output_folder, 'filtered_ranks.csv'), sep=';', header=None)
    df.columns = ['h', 'r', 't', 'hr', 'tr']

    for d in set(df['r']):
        rel_df = df[df['r'] == d]
        size = len(dataset.rid2target[dataset.relation_name_2_id[d]])
        top_count = 0
        for i in range(len(df)):
            # if df.loc[i, 'tr'] <= math.ceil(size*0.05):
            if df.loc[i, 'tr'] <= 5 and not ignore_triple(df.loc[i, ['h', 'r', 't']]):
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

ech(f"Reading facts to explain... from {args.explain_path}")
with open(args.explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]
print("len(testing_facts):", len(testing_facts))

output_lines = []
def triple2str(triple):
    return '<' +','.join(triple) + '>'

def print_line(line):
    print(line)
    output_lines.append(line)

# get the ids of the elements of the fact to explain and the perspective entity

prefilter = args.prefilter
relevance_threshold = args.relevance_threshold

if args.baseline is None:
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                    relevance_threshold=relevance_threshold)
elif args.baseline == "data_poisoning":
    kelpie = DataPoisoning(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter)
elif args.baseline == "criage":
    kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters)
elif args.baseline == "k1":
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                    relevance_threshold=relevance_threshold, max_explanation_length=1)
else:
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                    relevance_threshold=relevance_threshold)

xrule = Xrule(model, dataset)


print('dataset size:', dataset.train_samples.shape, len(dataset.entity_id_2_name), len(dataset.relation_id_2_name))

def retrain_without_samples(remove_triples, sample_to_explain):
    model = TargetModel(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
    model.to('cuda')
    logger.info("[retrain_without_samples] Re-Training model...")
    st = time.time()
    samples = dataset.train_samples.copy()
    ids = []
    
    for t in remove_triples:
        # print('filtering tail', dataset.train_to_filter[(sample[0], sample[1])])
        triple = list(dataset.forward_triple(t))
        try:
            ids.append(samples.tolist().index(triple))
        except:
            logger.warning(f"triple {triple}({t}) not found in samples")
            return 0

    print('delete rows:', ids, [samples[i] for i in ids])
    np.delete(samples, ids, axis=0)
    print('retrain with samples:', samples.shape, samples[:5])
    optimizer = Optimizer(model=model, hyperparameters=hyperparameters)
    optimizer.epochs = 2
    optimizer.train(train_samples=samples, evaluate_every=10, #10 if args.method == "ConvE" else -1,
                    save_path=args.model_path,
                    valid_samples=dataset.valid_samples)

    logger.info(f"[retrain_without_samples] Re-Train time: {time.time() - st}")
    original_target_entity_score, \
    original_best_entity_score, \
    original_target_entity_rank = kelpie.engine.original_results_for(original_sample_to_predict=sample_to_explain)
    
    rt_target_entity_score, \
    rt_best_entity_score, \
    rt_target_entity_rank = extract_detailed_performances(model, sample_to_explain)

    rank_worsening = rt_target_entity_rank - original_target_entity_rank
    score_worsening = original_target_entity_score - rt_target_entity_score

    if dataset.args.relevance_method == 'kelpie':
        relevance = float(rank_worsening + 1 / (1 + math.exp(-score_worsening)))
    elif dataset.args.relevance_method == 'rank':
        relevance = float(rank_worsening)
    elif dataset.args.relevance_method == 'score':
        relevance = float(score_worsening * 10)

    cur_line = ";".join(dataset.sample_to_fact(sample_to_explain)) + ";" + \
                ";".join([";".join(dataset.sample_to_fact(x)) for x in remove_triples]) + ";" + \
                str(original_best_entity_score) + ";" + \
                str(original_target_entity_score) + ";" + \
                str(original_target_entity_rank) + ";" + \
                str(rt_best_entity_score) + ";" + \
                str(rt_target_entity_score) + ";" + \
                str(rt_target_entity_rank) + ";" + \
                str(relevance)

    with open(os.path.join(args.output_folder, f"retrain_details_{len(remove_triples)}.csv"), "a") as output_file:
        output_file.writelines([cur_line + "\n"])
    logger.info(f"[retrain_without_samples] Re-Train Completed. Relevance: {relevance}")

    return round(relevance, 4)


def print_facts(rule_samples_with_relevance, sample_to_explain):
    # print(rule_samples_with_relevance)
    rule_facts_with_relevance = []
    for cur_rule_with_relevance in rule_samples_with_relevance:
        cur_rule_samples, cur_relevance = cur_rule_with_relevance

        cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
        cur_rule_facts = ";".join([triple2str(x) for x in cur_rule_facts])
        rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))
        print_line('\t' + rule_facts_with_relevance[-1])

        # if cur_relevance > 1:
        #     for i in range(10):
        #         retrain_without_samples(remove_triples=cur_rule_samples, sample_to_explain=sample_to_explain)

path_dic = {}
cnt_df = pd.DataFrame(columns=['path', 'head', 'tail'])

for i, fact in enumerate(testing_facts):
    if ignore_triple(fact):
        continue
    head, relation, tail = fact
    ech(f"Explaining fact {i} on {len(testing_facts)}: + {triple2str(fact)}")
    logger.info(f"Explaining fact {i} on {len(testing_facts)}: + {triple2str(fact)}")
    head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                    dataset.get_id_for_relation_name(relation), \
                                    dataset.get_id_for_entity_name(tail)
    sample_to_explain = (head_id, relation_id, tail_id)
    
    paths = dataset.find_all_path_within_k_hop(head_id, tail_id, 3)
    heads = set([x[0] for x in paths])
    tails = set([x[-1] for x in paths])
    cnt_df.loc[len(cnt_df)] = [len(paths), len(heads), len(tails)]
    print('path:', len(paths), 'head:', len(heads), 'tail:', len(tails))
    
    # randomly select 10 paths from paths
    selected_paths = random.sample(paths, 10)
    for p in selected_paths:
        Path.build(sample_to_explain, [tuple(t) for t in p])
    if i < 20:
        path_dic[str(sample_to_explain)] = paths
    continue

    score, best, rank = extract_detailed_performances(model, sample_to_explain)
    if rank > 10:
        logger.info(f'{dataset.sample_to_fact(sample_to_explain, True)} is not a valid prediction (rank={rank}, score={score}). Skip')
        continue
    
    ech(f'input of fact {dataset.sample_to_fact(sample_to_explain, True)} {rank}')
    # rule_samples_with_relevance = kelpie.explain_necessary(sample_to_explain=sample_to_explain,
    #                                                         perspective="head",
    #                                                         num_promising_samples=args.prefilter_threshold)
    # print_line(f'output of fact {triple2str(fact)}')
    # print_facts(rule_samples_with_relevance, sample_to_explain)

    # for cur_rule_with_relevance in rule_samples_with_relevance:
    #     cur_rule_samples, cur_relevance = cur_rule_with_relevance
    #     if len(cur_rule_samples) == 1:
    #         print('finding all path for', cur_rule_samples[0])
    #         target = cur_rule_samples[0][-1] if head_id == cur_rule_samples[0][0] else cur_rule_samples[0][0]
    #         path = dataset.find_all_path(sample_to_explain, target)
    explanations = xrule.explain_necessary(sample_to_explain)

print(cnt_df.mean())
cnt_df.to_csv(f'{args.output_folder}/path_cnt.csv', index=False)
with open(f'{args.output_folder}/all_paths.json', 'w') as f:
    json.dump(path_dic, f, indent=4, cls=NumpyEncoder)

# ech('explaination output:')
# end_time = time.time()
# print("Explain time: " + str(end_time - start_time) + " seconds")
# with open(os.path.join(args.output_folder, f"{args.mode}.txt"), "w") as output:
#     output.writelines(output_lines)

# print('count_dic', count_dic)
# print('count_dic_mean', {k: np.mean(v) for k, v in count_dic.items()})