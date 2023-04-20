import os
import torch
import click
import yaml
import numpy as np
from typing import Any, Tuple
import argparse
from typing import Any, Tuple
import numpy
import torch
from torch import nn
from dataset import Dataset
import os
import pandas as pd
from collections import defaultdict
import re
import argparse
import yaml
import time
import random

from link_prediction.models.model import *
from link_prediction.models.transe import TransE
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.gcn import CompGCN
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from link_prediction.evaluation.evaluation import Evaluator
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER

def parse_args():
    parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")
    parser.add_argument("--dataset",
                        type=str,
                        help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")

    parser.add_argument("--method",
                        type=str,
                        help="The method to use: ComplEx, ConvE, TransE")

    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="path of the model to explain the predictions of.")

    parser.add_argument("--explain_path",
                        type=str,
                        required=True,
                        help="path of the file with the facts to explain the predictions of.")

    parser.add_argument("--coverage",
                        type=int,
                        default=10,
                        help="Number of random entities to extract and convert")

    parser.add_argument("--baseline",
                        type=str,
                        default=None,
                        choices=[None, "k1", "data_poisoning", "criage"],
                        help="attribute to use when we want to use a baseline rather than the Kelpie engine")

    parser.add_argument("--entities_to_convert",
                        type=str,
                        help="path of the file with the entities to convert (only used by baselines)")

    parser.add_argument("--relevance_threshold",
                        type=float,
                        default=None,
                        help="The relevance acceptance threshold to use")

    prefilters = [TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER]
    parser.add_argument('--prefilter',
                        choices=prefilters,
                        default='graph-based',
                        help="Prefilter type in {} to use in pre-filtering".format(prefilters))

    parser.add_argument("--prefilter_threshold",
                        type=int,
                        default=20,
                        help="The number of promising training facts to keep after prefiltering")

    parser.add_argument("--run",
                        type=str,
                        default='111',
                        help="whether train, test or explain")

    parser.add_argument("--output_folder",
                        type=str,
                        default='.')

    parser.add_argument("--embedding_model",
                        type=str,
                        default=None,
                        help="embedding model before LP model header")

    parser.add_argument('--ignore_inverse', dest='ignore_inverse', default=False, action='store_true',
                        help="whether ignore inverse relation when evaluate")

    parser.add_argument('--train_restrain', dest='train_restrain', default=False, action='store_true',
                        help="whether apply tail restrain when training")

    parser.add_argument('--specify_relation', dest='specify_relation', default=False, action='store_true',
                        help="whether specify relation when evaluate")

    parser.add_argument('--relation_path', default=False, action='store_true',
                        help="whether generate relation path instead of triples")
    
    return parser.parse_args()


def rd(x):
    return round(x, 6)

def path2str(dataset, path):
    if not args.relation_path:
        return ",".join(dataset.sample_to_fact(path))
    s = ''
    for t in path:
        f = dataset.sample_to_fact(t)
        s += f'{f[0]}-{f[1]}->'
    return s + f[2]


def paths2str(dataset, paths):
    return "|".join([path2str(dataset, x) for x in paths])


def strfy(entity_ids):
    if hasattr(entity_ids, '__iter__'):
        lis = [str(x) for x in entity_ids]
        return ','.join(lis)
    return entity_ids

def get_entity_embeddings(entity_embeddings, kelpie_entity_embedding):
    if kelpie_entity_embedding is None:
        return entity_embeddings
    
    # print(len(entity_embeddings), len(kelpie_entity_embedding), entity_embeddings.shape)

    return torch.cat([entity_embeddings, kelpie_entity_embedding], 0)
    # 
    # print(kelpie_entity_embedding.shape, type(kelpie_entity_embedding))
    # return torch.cat(entity_embeddings + [kelpie_entity_embedding], 0)

def terminate_at(length, count):
    '''记录长度为length的解释有多少个'''
    count_dic[length].append(count)
    print(f'\tnumber of rules with length {length}: {count}')


def get_first(x):
    if hasattr(x, "__iter__"):
        return x[0]
    return x

def prefilter_negative(all_rules, top_k=None):
        if type(all_rules) == dict:
            all_rules = all_rules.items()
        all_rules = sorted(all_rules, key=lambda x: get_first(x[1]), reverse=True)
        if top_k is None or top_k > len(all_rules):
            top_k = len(all_rules)
        for i in range(top_k):
            if get_first(all_rules[i][1]) < 0:
                break
        i += 1
        print(f'\tpositive top {top_k} rules: {i}/{len(all_rules)}')
        return all_rules[:i]


def reverse_sample(t: Tuple[Any, Any, Any], num_direct_relations: int):
    if t[1] < num_direct_relations:
        reverse_rel = t[1] + num_direct_relations
    else:
        reverse_rel = t[1] - num_direct_relations
    return (t[2], reverse_rel, t[0])


def get_forward_sample(t: Tuple[Any, Any, Any], num_direct_relations: int):
    if t[1] < num_direct_relations:
        return t
    return (t[2], t[1] - num_direct_relations, t[0])


def plot_dic(dic, path='data/statistic.png', size=(15, 6), label=True, rotation=30, limit=10, hspace=0.4):
    '''
    limit 代表 此值以下不绘制
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(dic), figsize=(size[0], size[1] * len(dic)))
    plt.xticks(rotation=270)
    fig.subplots_adjust(hspace=hspace, wspace=0.2)
    
    for ix, t in enumerate(list(dic.keys())):
        d = {k: v for k, v in dic[t].items() if v >= limit}

        axis = ax if len(dic) == 1 else ax[ix]
        axis.title.set_text(f'{t}: {len(d)}/{len(dic[t])} (limit:{limit})')
        rects = axis.bar([str(x) for x in d.keys()], list(d.values()))
        
        if label:
            for rect in rects:  #rects 是柱子的集合
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=10, ha='center', va='bottom')

        for tick in axis.get_xticklabels():
            tick.set_rotation(rotation)
    
    plt.savefig(path)


def plot_dics(dics, folder):
    os.makedirs(folder, exist_ok=True)
    for k, dic in dics.items():
        plot_dic({k: dic}, os.path.join(folder, k + '.png'), limit=0, size=(8, 8), hspace=0, rotation=0)


def ech(s, fg='yellow'):
    click.echo(click.style('='*10 + s + '='*10, fg))

def get_yaml_data(yaml_file):
    # 打开yaml文件
    print("***获取yaml文件数据***")
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    print("类型：", type(file_data))

    # 将字符串转化为字典或列表
    print("***转化yaml数据为字典或列表***")
    data = yaml.safe_load(file_data)
    print(data)
    print("类型：", type(data))
    return data



# global variable!
count_dic = defaultdict(list)
relevance_df = pd.DataFrame(columns=['triple', 'explanation', 'relevance', 'head_relevance', 'tail_relevance'])
addition_df = pd.DataFrame(columns=['triple', 'origin', 'addition', 'relevance', 'origin_relevance', 'addition_relevance'])
prelimentary_df = pd.DataFrame(columns=['explanation', 'prelimentary', 'true', 'type_ix'])
config = get_yaml_data('config.yaml')
args = parse_args()

print('relation_path', args.relation_path)
cfg = config[args.dataset][args.method]
args.restrain_dic = config[args.dataset].get('tail_restrain', None)

# deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

# load the dataset and its training samples
ech(f"Loading dataset {args.dataset}...")
dataset = Dataset(name=args.dataset, separator="\t", load=True, args=args)
try:
    tail_restrain = dataset.tail_restrain
except:
    tail_restrain = None
args.tail_restrain = tail_restrain

ech("Initializing LP model...")
hyperparameters = {
    DIMENSION: cfg['D'],
    EPOCHS: cfg['Ep'],
    RETRAIN_EPOCHS: cfg['REp'] if 'REp' in cfg else cfg['Ep'],
    BATCH_SIZE: cfg['B'],
    LEARNING_RATE: cfg['LR']
}
if args.method == "ConvE":
    hyperparameters = {**hyperparameters,
                    INPUT_DROPOUT: cfg['Drop']['in'],
                    FEATURE_MAP_DROPOUT: cfg['Drop']['feat'],
                    HIDDEN_DROPOUT: cfg['Drop']['h'],
                    HIDDEN_LAYER_SIZE: 9728,
                    DECAY: cfg['Decay'],
                    LABEL_SMOOTHING: 0.1}
    TargetModel = ConvE
    Optimizer = BCEOptimizer
elif args.method == "ComplEx":
    hyperparameters = {**hyperparameters,
                    INIT_SCALE: 1e-3,
                    OPTIMIZER_NAME: 'Adagrad',  # 'Adagrad', 'Adam', 'SGD'
                    DECAY_1: 0.9,
                    DECAY_2: 0.999,
                    REGULARIZER_WEIGHT: cfg['Reg'],
                    REGULARIZER_NAME: "N3"}
    TargetModel = ComplEx
    Optimizer = MultiClassNLLOptimizer
elif args.method == "TransE":
    hyperparameters = {**hyperparameters,
                    MARGIN: 5,
                    NEGATIVE_SAMPLES_RATIO: cfg['N'],
                    REGULARIZER_WEIGHT: cfg['Reg'],}
    TargetModel = TransE
    Optimizer = PairwiseRankingOptimizer
print('LP hyperparameters:', hyperparameters)

if args.embedding_model and args.embedding_model != 'none':
    cf = config[args.dataset][args.embedding_model]
    print('embedding_model config:', cf)
    args.embedding_model = CompGCN(
        num_bases=cf['num_bases'],
        num_rel=dataset.num_relations,
        num_ent=dataset.num_entities,
        in_dim=cf['in_dim'],
        layer_size=cf['layer_size'],
        comp_fn=cf['comp_fn'],
        batchnorm=cf['batchnorm'],
        dropout=cf['dropout']
    )
else:
    args.embedding_model = None

model = TargetModel(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
model.to('cuda')


def get_origin_score(fact):
    sample_to_explain = dataset.fact_to_sample(fact)
    all_scores = model.all_scores(numpy.array([sample_to_explain])).detach().cpu().numpy()[0]
    return all_scores[sample_to_explain[-1]]

class Scores:
    def __init__(self) -> None:
        pass

# triple 为 3个元素的list
class Path:
    def __init__(self, triples, fact_to_explain) -> None:
        self.triples = triples
        self.head = triples[0]
        self.tail = triples[-1]
        self.fact_to_explain = fact_to_explain

    @staticmethod
    def from_str(s):
        # the explanation consists of a single path
        nodes = re.split('->|-', s)
        triples = [nodes[i:i+3] for i in range(0, len(nodes), 2)][:-1]
        return Path(triples)
    
    def __str__(self) -> str:
        return ''.join([triple[0] + '-' + triple[1] + '->' for triple in self.triples] + [self.tail[-1]])

    def get_retrain_score(self):
        self.retrain_head_score = Explanation([self.head], self.fact_to_explain)
        self.retrain_tail_score = Explanation([self.tail], self.fact_to_explain)
        self.retrain_path_score = Explanation(self.triples, self.fact_to_explain)


class Paths:
    def __init__(self, paths, fact_to_explain) -> None:
        self.paths = paths
        self.fact_to_explain = fact_to_explain
        self.head = [path.head for path in self.paths]
        self.tail = [path.tail for path in self.paths]
        self.triples = []
        for path in self.paths:
            self.triples.extend(path.triples)

    @staticmethod
    def from_str(s):
        return Paths([Path(path_str) for path_str in s.split('|')])
    
    def __str__(self) -> str:
        return '|'.join([str(path) for path in self.paths])
    
    def get_retrain_score(self):
        self.retrain_head_score = Explanation(self.head, self.fact_to_explain)
        self.retrain_tail_score = Explanation(self.tail, self.fact_to_explain)
        self.retrain_path_score = Explanation(self.triples, self.fact_to_explain)


class Explanation:
    def __init__(self, triples, fact_to_explain) -> None:
        self.triples = triples
        self.fact_to_explain = fact_to_explain

    def __str__(self) -> str:
        return str(self.triples)

    def get_retrain_score(self):
        model = TargetModel(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
        model.to('cuda')
        ech("Re-Training model...")
        t = time.time()
        samples = dataset.train_samples.copy()
        ids = []
        for triple in self.triples:
            sample = dataset.fact_to_sample(triple)
            sample = dataset.original_sample(sample)
            # print('filtering tail', dataset.train_to_filter[(sample[0], sample[1])])
            ids.append(samples.tolist().index(list(sample)))

        print('delete rows:', ids)
        np.delete(samples, ids, axis=0)
        optimizer = Optimizer(model=model, hyperparameters=hyperparameters)
        optimizer.train(train_samples=samples, evaluate_every=10, #10 if args.method == "ConvE" else -1,
                        save_path=args.model_path,
                        valid_samples=dataset.valid_samples)
        print(f"Train time: {time.time() - t}")

        sample_to_explain = dataset.fact_to_sample(self.fact_to_explain)
        model.eval()
        all_scores = model.all_scores(numpy.array([sample_to_explain])).detach().cpu().numpy()[0]
        return all_scores[sample_to_explain[-1]]