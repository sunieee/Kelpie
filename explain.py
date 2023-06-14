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
    print(os.path.join(args.output_folder, 'filtered_ranks.csv'))
    df = pd.read_csv(os.path.join(args.output_folder, 'filtered_ranks.csv'), sep=';', header=None, dtype=str)
    df.columns = ['h', 'r', 't', 'hr', 'tr']
    df['hr'] = df['hr'].astype(int)
    df['tr'] = df['tr'].astype(int)

    for d in set(df['r']):
        rel_df = df[df['r'] == d]
        rel_df.reset_index(inplace=True)
        size = len(dataset.rid2target[dataset.relation_name_2_id[d]])
        top_count = 0
        for i in range(len(rel_df)):
            # if df.loc[i, 'tr'] <= math.ceil(size*0.05):
            if rel_df.loc[i, 'tr'] == 1 and not ignore_triple(rel_df.loc[i, ['h', 'r', 't']]):
                top_count += 1
                lis.append('\t'.join([str(rel_df.loc[i, 'h']), rel_df.loc[i, 'r'], str(rel_df.loc[i, 't'])]))
        print("{:^15}\t{:^15}\t{:^15}\t{:^15}".format(d, size, len(rel_df), top_count))

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

def retrain_without_samples(remove_triples, prediction):
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
    original_target_entity_rank = kelpie.engine.original_results_for(original_sample_to_predict=prediction)
    
    rt_target_entity_score, \
    rt_best_entity_score, \
    rt_target_entity_rank = extract_performances(model, prediction)

    rank_worsening = rt_target_entity_rank - original_target_entity_rank
    score_worsening = original_target_entity_score - rt_target_entity_score

    if dataset.args.relevance_method == 'kelpie':
        relevance = float(rank_worsening + 1 / (1 + math.exp(-score_worsening)))
    elif dataset.args.relevance_method == 'rank':
        relevance = float(rank_worsening)
    elif dataset.args.relevance_method == 'score':
        relevance = float(score_worsening * 10)

    cur_line = ";".join(dataset.sample_to_fact(prediction)) + ";" + \
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


def print_facts(rule_samples_with_relevance, prediction):
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
        #         retrain_without_samples(remove_triples=cur_rule_samples, prediction=prediction)

path_dic = {}
cnt_df = pd.DataFrame(columns=['prediction', 'fact','#paths', '#heads', '#hop2', '#tails', '#meta_path', 'max_meta', '#super_path', 'max_super', '#ph', '#p2', '#pt', '#meta_h', '#meta_t'])
valid_hops_df = pd.DataFrame()
valid_exp_df = pd.DataFrame()
exp_info_df = pd.DataFrame()


def path_statistic(prediction):
    head_id, relation_id, tail_id = prediction
    # make a statistic of path on the prediction
    paths = dataset.find_all_path_within_k_hop(head_id, tail_id, 3)
    try:
        heads = set([x[0] for x in paths])
        hop2 = set([x[1] for x in paths])
        tails = set([x[-1] for x in paths])
    except Exception as e:
        print(e)
        print(paths)
        return

    ph = set([x[0][-1] for x in paths])
    p2 = set([x[1][-1] for x in paths])
    pt = set([x[-1][0] for x in paths])
    meta_path = defaultdict(int)
    super_path = defaultdict(int)
    meta_h = set([x[0][1] for x in paths])
    meta_t = set([x[-1][1] for x in paths])
    for p in paths:
        meta = [t[1] for t in p]
        meta_path[tuple(meta)] += 1
        super = [t[0] for t in p]
        super_path[tuple(super)] += 1
    max_meta = max(list(meta_path.values()) + [0])
    max_super = max(list(super_path.values()) + [0])
    cnt_df.loc[len(cnt_df)] = [str(prediction), str(fact) , len(paths), len(heads), len(hop2), len(tails), len(meta_path), max_meta, len(super_path), max_super, len(ph), len(p2), len(pt), len(meta_h), len(meta_t)]
    print('path:', len(paths), 'head:', len(heads),'hop2', len(hop2), 'tail:', len(tails), 'rel_typ:', len(meta_path), 'max_rel_typ:', max_meta, 'ph:', len(ph), 'p2:', len(p2), 'pt:', len(pt), 'meta_h:', len(meta_h), 'meta_t:', len(meta_t))
    cnt_df.to_csv(f'{args.output_folder}/path_cnt.csv', index=False)
    if i < 20:
        path_dic[str(prediction)] = paths


def random_explain_path(super_paths):
    # randomly select 10 paths from paths
    # selected_paths = random.sample(paths, min(10, len(paths)))
    # for p in selected_paths:
    #     Path(prediction, [tuple(t) for t in p])
    
    # randomly select 10 super paths from super paths
    selected_super_paths = random.sample(super_paths, min(10, len(super_paths)))
    for p in selected_super_paths:
        print('prediction', prediction, 'construct super path', p)
        SuperPath(prediction, list(p))

hop_df = pd.DataFrame()


def random_explain_group(phs, pts, prediction):
    head_id, relation_id, tail_id = prediction
    hop_dic = {
        'head': phs,
        'tail': pts
    }
    target_id = {
        'head': head_id,
        'tail': tail_id
    }
    for target in ['head', 'tail']:
        hops = hop_dic[target]
        selected_hops = random.sample(hops, min(10, len(hops)))
        all_samples_to_remove = set()
        logger.info(f'hops on {target}: {hops}')
        logger.info(f'selected hops: {selected_hops}')

        for hop in selected_hops:
            samples_to_remove = args.available_samples[target_id[target]] & args.available_samples[hop]
            all_samples_to_remove |= samples_to_remove
        all_exp = Explanation.build(prediction, list(all_samples_to_remove), [target_id[target]])

        for hop in selected_hops:
            ech(f'explaining hop {hop} on sample {prediction}')
            samples_to_remove = args.available_samples[target_id[target]] & args.available_samples[hop]
            exp = Explanation.build(prediction, list(samples_to_remove), [target_id[target]])
            ret = {
                'prediction': dataset.sample_to_fact(prediction, True), 
                'perspective': target, 
                'hop': hop, 
                'rel': exp.relevance, 
                '#remove': len(samples_to_remove), 
                '#hops': len(selected_hops), 
                'all_rel': all_exp.relevance,
            }
            for p in [1, 2, float('inf')]:
                ret.update({
                    f'delta_{p}': exp.ret[f'delta_{p}'],
                    f'delta_all_{p}': all_exp.ret[f'delta_{p}'],
                    f'partial_t_{p}': exp.ret[f'partial_t_{p}'],
                    f'partial_h_{p}': exp.ret[f'partial_h_{p}'],
                    f'partial_h_all_{p}': all_exp.ret[f'partial_h_{p}'],
                    f'partial_t_all_{p}': all_exp.ret[f'partial_t_{p}'],
                })

            update_df(hop_df, ret, 'hops.csv')


def explain_all_path(super_paths, prediction, phs, pts):
    head, relation, tail = prediction
    first_hop_exps = {}
    for hop in phs:
        hop_samples = list(args.available_samples[head] & args.available_samples[hop])
        exp = Explanation.build(prediction, hop_samples, [head])
        if exp.relevance > DEFAULT_VALID_THRESHOLD - 0.05:
            first_hop_exps[hop] = exp
    logger.info(f'valid first hops ({len(first_hop_exps)}/{len(phs)}): {first_hop_exps.keys()}')

    last_hop_exps = {}
    for hop in pts:
        hop_samples = list(args.available_samples[tail] & args.available_samples[hop])
        exp = Explanation.build(prediction, hop_samples, [tail])
        if exp.relevance > DEFAULT_VALID_THRESHOLD - 0.05:
            last_hop_exps[hop] = exp
    logger.info(f'valid last hops ({len(last_hop_exps)}/{len(pts)}): {last_hop_exps.keys()}')

    # search hyper-path between first and last hop
    valid_super_paths = [(head, p, tail) for p in set(first_hop_exps.keys()) & set(last_hop_exps.keys())]
    for super in super_paths:
        if super[1] in first_hop_exps and super[-2] in last_hop_exps:
            valid_super_paths.append(super)
    logger.info(f'valid super paths ({len(valid_super_paths)}/{len(super_paths)}): {valid_super_paths}')

    filtered_exps = {}
    valid_exps = []
    for super in valid_super_paths:
        head_exp = first_hop_exps[super[1]]
        tail_exp = last_hop_exps[super[-2]]

        update_df(exp_info_df, {
            'head_partial': head_exp.ret['partial_inf'],
            'head_delta': head_exp.ret['partial_t_inf'] * tail_exp.ret['delta_2'],
            'head_partial_t': head_exp.ret['partial_t_inf'],
            'head_partial_h': head_exp.ret['partial_h_inf'],
            'tail_partial_t': tail_exp.ret['partial_t_inf'],
            'tail_partial_h': tail_exp.ret['partial_h_inf'],
            'tail_partial': tail_exp.ret['partial_inf'],
            'tail_delta': tail_exp.ret['partial_h_inf'] * head_exp.ret['delta_2'],
        }, 'exp_info.csv')
        
        delta_h = head_exp.ret['partial_t_inf'] * tail_exp.ret['delta_2']
        if head_exp.relevance + delta_h * coef['k_h'] < DEFAULT_VALID_THRESHOLD:
            logger.info(f'Rule1: {hop} is not relevant: {head_exp.relevance} + {delta_h} * {coef["k_h"]}')
            continue

        delta_t = tail_exp.ret['partial_h_inf'] * head_exp.ret['delta_2']
        if tail_exp.relevance + delta_t * coef['k_t'] < DEFAULT_VALID_THRESHOLD:
            logger.info(f'Rule2: {hop} is not relevant: {tail_exp.relevance} + {delta_t} * {coef["k_t"]}')
            continue
        
        partial = (head_exp.ret['partial_inf'] + tail_exp.ret['partial_inf']) / 2
        delta = partial * head_exp.ret['delta_2'] * tail_exp.ret['delta_2']
        if head_exp.relevance + tail_exp.relevance + delta * coef['k'] < DEFAULT_VALID_THRESHOLD:
            logger.info(f'Rule3: {super} is not relevant: {head_exp.relevance} + {tail_exp.relevance} + {delta} * {coef["k"]}')
            continue

        exp = Explanation.build(prediction, head_exp.samples_to_remove + tail_exp.samples_to_remove, [head, tail])
        filtered_exps[super] = exp
        if exp.relevance > DEFAULT_VALID_THRESHOLD:
            valid_exps.append((super, exp))
            update_df(valid_exp_df, {
                'prediction': prediction,
                'super_path': super,
                'relevance': exp.relevance,
                'head_rel': head_exp.relevance,
                'tail_rel': tail_exp.relevance,
                'triples': head_exp.samples_to_remove + tail_exp.samples_to_remove,
                'partial': partial,
                'delta_h': head_exp.ret['delta_2'],
                'delta_t': tail_exp.ret['delta_2'],
                'delta': delta,
            }, 'valid_exp.csv')


    logger.info(f'exp ({len(filtered_exps)}/{len(valid_super_paths)}): {filtered_exps.keys()}')
    valid_exps.sort(key=lambda x: x[1].relevance, reverse=True)
    logger.info(f'valid exp ({len(valid_exps)}/{len(filtered_exps)}): {valid_exps}')

    update_df(valid_hops_df, {
            'prediction': prediction,
            '#pts': len(pts),
            '#valid_last_hops': len(last_hop_exps),
            '#phs': len(phs),
            '#valid_first_hops': len(first_hop_exps),
            '#super_paths': len(super_paths),
            '#valid_super_paths': len(valid_super_paths),
            '#exps': len(filtered_exps),
            '#valid_exps': len(valid_exps),
        }, 'valid_hops.csv')



for i, fact in enumerate(testing_facts[:2000]):
    if ignore_triple(fact):
        continue
    h, r, t = fact
    ech(f"Explaining fact {i} on {len(testing_facts)}: + {triple2str(fact)}")
    # logger.info(dataset.entity_name_2_id)
    head, relation, tail = dataset.get_id_for_entity_name(h), \
                                    dataset.get_id_for_relation_name(r), \
                                    dataset.get_id_for_entity_name(t)
    prediction = (head, relation, tail)
    logger.info(f"Explaining fact {i} on {len(testing_facts)}: + {dataset.sample_to_fact(prediction, True)}")
    
    score, best, rank = extract_performances(model, prediction)
    if rank > 10:
        logger.info(f'{dataset.sample_to_fact(prediction, True)} is not a valid prediction (rank={rank}, score={score}). Skip')
        continue
    ech(f'input of fact {dataset.sample_to_fact(prediction, True)} {rank}')
    
    
    # path_statistic(prediction)
    paths = dataset.find_all_path_within_k_hop(head, tail, 3)
    super_paths = set()
    available_samples = defaultdict(set)
    phs = set()
    pts = set()
    for p in paths:
        super = get_path_entities(prediction, p)
        phs.add(super[1])
        pts.add(super[-2])
        super_paths.add(tuple(super))
        for t in p:
            # print(t, t[0], t[2])
            available_samples[t[0]].add(t)
            available_samples[t[2]].add(t)
    args.available_samples = available_samples
    print('global_avilable_samples on path', len(args.available_samples))
    # random_explain_path(super_paths)
    # random_explain_group(phs, pts, prediction)
    # explain_all_path(super_paths, prediction, phs, pts)
    
    
    # rule_samples_with_relevance = kelpie.explain_necessary(prediction=prediction,
    #                                                         perspective="head",
    #                                                         num_promising_samples=args.prefilter_threshold)
    # print_line(f'output of fact {triple2str(fact)}')
    # print_facts(rule_samples_with_relevance, prediction)

    # for cur_rule_with_relevance in rule_samples_with_relevance:
    #     cur_rule_samples, cur_relevance = cur_rule_with_relevance
    #     if len(cur_rule_samples) == 1:
    #         print('finding all path for', cur_rule_samples[0])
    #         target = cur_rule_samples[0][-1] if head_id == cur_rule_samples[0][0] else cur_rule_samples[0][0]
    #         path = dataset.find_all_path(prediction, target)
    # explanations = xrule.explain_necessary(prediction)

    # if len(phs) >= 20 and len(pts) >= 200:
    #     print('too many hops, skip')
    #     continue

    head_generator = OneHopGenerator('head', prediction, phs)
    tail_generator = OneHopGenerator('tail', prediction, pts)
    path_generator = PathGenerator(prediction, super_paths)

    

print(cnt_df.describe())
cnt_df.describe().to_csv(f'{args.output_folder}/describe.csv', index=False)
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