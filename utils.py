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
# import List, Dict
from typing import List, Dict

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
from link_prediction.optimization.bce_optimizer import KelpieBCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import KelpieMultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import KelpiePairwiseRankingOptimizer
from collections import OrderedDict

from kelpie_dataset import KelpieDataset


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
    return round(x, 4)

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
print('model is_minimizer:', model.is_minimizer())


def get_origin_score(fact):
    sample_to_explain = dataset.fact_to_sample(fact)
    all_scores = model.all_scores(numpy.array([sample_to_explain])).detach().cpu().numpy()[0]
    return all_scores[sample_to_explain[-1]]

base_score = {}

if isinstance(model, ComplEx):
    kelpie_optimizer_class = KelpieMultiClassNLLOptimizer
elif isinstance(model, ConvE):
    kelpie_optimizer_class = KelpieBCEOptimizer
elif isinstance(model, TransE):
    kelpie_optimizer_class = KelpiePairwiseRankingOptimizer
else:
    kelpie_optimizer_class = KelpieMultiClassNLLOptimizer

class Triple:
    def __init__(self, triple) -> None:
        self.h = triple[0]
        self.r = triple[1]
        self.t = triple[2]
        self.triple = triple

    def reverse(self):
        if self.r < dataset.num_direct_relations:
            rr = self.r + dataset.num_direct_relations
        else:
            rr = self.r - dataset.num_direct_relations
        return Triple((self.t, rr, self.h))
    
    def __str__(self) -> str:
        return "<" + ", ".join([x.split('/')[-1] for x in self.to_fact()]) + ">"

    def forward(self):
        if self.r < dataset.num_direct_relations:
            return self
        return Triple((self.t, self.r - dataset.num_direct_relations, self.h))
    
    def origin_score(self, remove_triples=[], retrain=False):
        if len(remove_triples) == 0:
            if str(self) in base_score and not retrain:
                return base_score[str(self)]

        self.model = TargetModel(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
        self.model.to('cuda')
        ech("Re-Training model...")
        t = time.time()
        samples = dataset.train_samples.copy()
        ids = []
        for triple in remove_triples:
            # print('filtering tail', dataset.train_to_filter[(sample[0], sample[1])])
            ids.append(samples.tolist().index(list(triple.forward().triple)))

        print('delete rows:', ids, [samples[i] for i in ids])
        np.delete(samples, ids, axis=0)
        optimizer = Optimizer(model=self.model, hyperparameters=hyperparameters)
        optimizer.train(train_samples=samples, evaluate_every=10, #10 if args.method == "ConvE" else -1,
                        save_path=args.model_path,
                        valid_samples=dataset.valid_samples)
        print(f"Train time: {time.time() - t}")
        ret = self.extract_detailed_performances(self.model, 'AA')
        if len(remove_triples) == 0:
            base_score[str(self)] = ret

        return ret
    
    def extract_detailed_performances(self, model: Model, name: str):
        # return model.predict_tail(sample)
        print('evaluating', self.triple)
        model.eval()
        # check how the model performs on the sample to explain   , sigmoid=False
        all_scores = model.all_scores(numpy.array([self.triple])).detach().cpu().numpy()[0]

        # print('original target score:', all_scores[[self.origin.h, self.origin.t]])
        # print('all score:', all_scores)
        target_score = all_scores[self.t] # todo: this only works in "head" perspective
        filter_out = model.dataset.to_filter[(self.h, self.r)] if (self.h, self.r) in model.dataset.to_filter else []

        if model.is_minimizer():
            all_scores[filter_out] = 1e6
            all_scores[self.t] = target_score
            best_score = numpy.min(all_scores)
            target_rank = numpy.sum(all_scores <= target_score)  # we use min policy here

        else:
            all_scores[filter_out] = -1e6
            all_scores[self.t] = target_score
            best_score = numpy.max(all_scores)
            target_rank = numpy.sum(all_scores >= target_score)  # we use min policy here

        ret = {
            f'{name}_score': rd(target_score), 
            f'{name}_rank': target_rank, 
            f'{name}_best_score': rd(best_score)
        }
        print('ret:', ret)
        return rd(target_score)

    @staticmethod
    def from_fact(fact):
        sample = dataset.fact_to_sample(fact)
        return Triple(sample)
    
    def to_fact(self):
        return dataset.sample_to_fact(self.triple)

    def replace_head(self, triple):
        return Triple((triple.h, self.r, self.t))
    
    def replace_tail(self, triple):
        return Triple((self.h, self.r, triple.t))

# triple/sample 为长度为3的 tuple
class Path:
    def __init__(self, triples: List[Triple], fact_to_explain: Triple) -> None:
        assert len(triples)
        self.triples = triples
        self.fact_to_explain = fact_to_explain

    def reverse(self):
        return Path([triple.reverse() for triple in self.triples[::-1]], self.fact_to_explain.reverse())
    
    @property
    def head(self):
        return self.triples[0]
    
    @property
    def tail(self):
        return self.triples[-1]
    
    @property
    def rel_path(self):
        return tuple([triple.r for triple in self.triples])

    @staticmethod
    def from_str(s: str):
        # the explanation consists of a single path
        nodes = re.split('->|-', s)
        triples = [nodes[i:i+3] for i in range(0, len(nodes), 2)][:-1]
        return Path(triples)
    
    def __len__(self):
        return len(self.triples)
    
    def __str__(self) -> str:
        return ''.join([str(triple.h) + '-' + str(triple.r) + '->' for triple in self.triples] + [str(self.tail.t)])

    def has_entity(self, ent):
        return ent in [t.h for t in self.triples] # + [t.t for t in self.triples]

    def extend(self, triple: Triple):  #  -> Path
        return Path(self.triples + [triple], self.fact_to_explain)
    
    # overload operator +
    def __add__(self, path):
        return Path(self.triples + path.triples, self.fact_to_explain)
    
    @property
    def inverse_rel(self):
        for i in range(len(self.triples) - 1):
            if abs(self.triples[i].r - self.triples[i+1].r) == dataset.num_direct_relations:
                return min(self.triples[i].r, self.triples[i+1].r)
        return -1
    
class Explanation:

    # The kelpie_cache is a simple LRU cache that allows reuse of KelpieDatasets and of base post-training results
    # without need to re-build them from scratch every time.
    _kelpie_dataset_cache_size = kelpie_dataset_cache_size
    _kelpie_dataset_cache = OrderedDict()

    _original_model_results = {}  # map original samples to scores and ranks from the original model
    _base_pt_model_results = {}   # map original samples to scores and ranks from the base post-trained model
    _base_cache_embeddings = {} # map embeddings of base node to its embedding 
    
    df = pd.DataFrame(columns=['to_explain', 'paths', 'length', 'AA', 'AB', 'BA', 'BB', 'CA', 'AC', 'CC', 'head', 'tail', 'path'])
    print_count = 0

    def __init__(self, paths: List[Path], sample_to_explain: Triple) -> None:
        print(f'init explanation: sample: {str(sample_to_explain)}, removing: {str([str(p) for p in paths])}')
        self.paths = paths
        self.sample_to_explain = sample_to_explain
        self.head = [path.head for path in paths]
        self.tail = [path.tail for path in paths]
        
        self.original_samples_to_remove = set()
        # 共同路径头/尾
        for p in paths:    # remove samples connected to head/tail
            self.original_samples_to_remove.add(p.head.forward().triple)
            self.original_samples_to_remove.add(p.tail.forward().triple)
        print('\tremoving samples:', self.original_samples_to_remove)
            
        self.kelpie_dataset = self._get_kelpie_dataset_for(entity_ids=[sample_to_explain.h, sample_to_explain.t])

    def _get_kelpie_dataset_for(self, entity_ids) -> KelpieDataset:
        """
        Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """
        name = strfy(entity_ids)
        if name not in self._kelpie_dataset_cache:

            kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_ids=entity_ids)
            self._kelpie_dataset_cache[name] = kelpie_dataset
            self._kelpie_dataset_cache.move_to_end(name)

            if len(self._kelpie_dataset_cache) > self._kelpie_dataset_cache_size:
                self._kelpie_dataset_cache.popitem(last=False)

        return self._kelpie_dataset_cache[name]
    
    def post_train(self,
                   kelpie_model_to_post_train: KelpieModel,
                   kelpie_train_samples: numpy.array):
        """

        :param kelpie_model_to_post_train: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_train_samples:
        :return:
        """
        # kelpie_model_class = self.model.kelpie_model_class()
        # kelpie_model = kelpie_model_class(model=self.model, dataset=kelpie_dataset)
        kelpie_model_to_post_train.to('cuda')

        optimizer = self.kelpie_optimizer_class(model=kelpie_model_to_post_train,
                                                hyperparameters=hyperparameters,
                                                verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]
        t = time.time()
        optimizer.train(train_samples=kelpie_train_samples)
        if self.print_count < 5:
            self.print_count += 1
            print(f'\t\t[post_train_time: {rd(time.time() - t)}]')
        return kelpie_model_to_post_train


    def calculate_score(self):
        start_time = time.time()

        AA = self.sample_to_explain.origin_score(retrain=True)
        self.model = self.sample_to_explain.model

        kelpie_model_class = self.model.kelpie_model_class()
        kelpie_model = kelpie_model_class(model=self.model, dataset=self.kelpie_dataset)

        BB_triple = Triple(self.kelpie_dataset.as_clone_sample(original_sample=self.sample_to_explain.triple))
        CC_triple = Triple(self.kelpie_dataset.as_kelpie_sample(original_sample=self.sample_to_explain.triple))
        self.kelpie_dataset.remove_training_samples(self.original_samples_to_remove)

        kelpie_model = self.post_train(kelpie_model_to_post_train=kelpie_model,
                        kelpie_train_samples=self.kelpie_dataset.kelpie_train_samples)  # type: KelpieModel
        # kelpie_model.summary('after post_train')

        BB = BB_triple.extract_detailed_performances(kelpie_model, 'BB')
        AB = BB_triple.replace_head(self.sample_to_explain).extract_detailed_performances(kelpie_model, 'AB')
        BA = BB_triple.replace_tail(self.sample_to_explain).extract_detailed_performances(kelpie_model, 'BA')
        CC = CC_triple.extract_detailed_performances(kelpie_model, 'CC')
        AC = CC_triple.replace_head(self.sample_to_explain).extract_detailed_performances(kelpie_model, 'AC')
        CA = CC_triple.replace_tail(self.sample_to_explain).extract_detailed_performances(kelpie_model, 'CA')

        self.head = Relevance(self.head, self.sample_to_explain, AA, BA, CA)
        self.tail = Relevance(self.tail, self.sample_to_explain, AA, AB, AC)
        self.path = Relevance(self.head + self.tail, self.sample_to_explain, AA, BB, CC)

        print('calculate time:', rd(time.time() - start_time))    
        kelpie_model.undo_last_training_samples_removal()
        self.df.loc[len(self.df)] = {
            'to_explain': self.sample_to_explain,
            'paths': [str(p) for p in self.paths],
            'length': len(self.paths),
            'AA': AA,
            'AB': AB,
            'BA': BA,
            'BB': BB,
            'CA': CA,
            'AC': AC,
            'CC': CC,
            'head': self.head.approx,
            'tail': self.tail.approx,
            'path': self.path.approx
        }
        self.df.to_csv(f'{args.output_folder}/explanation.csv', index=False)


    def has_negative(self):
        return self.head.approx < 0 or self.tail.approx < 0 or self.path.approx < 0

    @property
    def max_relevance(self):
        return max(self.head.approx, self.tail.approx, self.path.approx)
    
    @property
    def min_relevance(self):
        return min(self.head.approx, self.tail.approx, self.path.approx)
    
    @property
    def relevance(self):
        return self.path.approx
    
    def __str__(self):
        return f'''{self.fact_to_explain}: {[str(p) for p in self.paths]}
head: {str(self.head)}
tail: {str(self.tail)}
path: {str(self.path)}
        '''

class Relevance:
    get_truth = False
    df = pd.DataFrame(columns=['to_explain', 'triples', 'length', 'A', 'T', 'B', 'C', 'truth', 'approx']) 

    def __init__(self, triples: List[Triple], sample_to_explain: Triple, A, B, C) -> None:
        self.triples = triples
        self.sample_to_explain = sample_to_explain
        self.A = A
        self.B = B
        self.C = C

        self.df.loc[len(self.df)] = {
            'to_explain': str(sample_to_explain),
            'triples': [str(t) for t in triples],
            'length': len(triples),
            'A': A,
            'T': self.T if self.get_truth else None,
            'B': B,
            'C': C,
            'truth': self.truth if self.get_truth else None,
            'approx': self.approx}
        self.df.to_csv(f'{args.output_folder}/relevance.csv', index=False)

    @property
    def T(self):
        if '_T' in self.__dict__:
            return self._T
        self._T = self.sample_to_explain.origin_score(self.triples)
        return self._T
    
    @property
    def truth(self):
        return self.A - self.T
    
    @property
    def approx(self):
        return self.B - self.C

    def __str__(self):
        return f'{[str(t) for t in self.head]}: {self.approx}'
