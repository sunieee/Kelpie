import os
import argparse
import random
import time
import numpy
import torch

from dataset import Dataset
import yaml
import click
import pandas as pd
import numpy as np
import math
import logging
from datetime import datetime
from tqdm import tqdm
from dataset import Dataset
import json
from queue import PriorityQueue

from link_prediction.models.transe import TransE
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE, PostConvE
from link_prediction.models.gcn import CompGCN
from link_prediction.models.model import *
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER
from link_prediction.models.tucker import TuckER
from link_prediction.optimization.bce_optimizer import KelpieBCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import KelpieMultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import KelpiePairwiseRankingOptimizer
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional
from kelpie_dataset import KelpieDataset
import warnings

import scipy.stats as stats

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim")

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

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

config = get_yaml_data('config.yaml')

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

parser.add_argument('--relevance_method', type=str, default='kelpie', 
                    choices=['rank', 'score', 'kelpie', 'hybrid'], help="the method to compute relevance")

# parser.add_argument('--sort', dest='sort', default=False, action='store_true',
#                     help="whether sort the dataset")

args = parser.parse_args()
cfg = config[args.dataset][args.method]
coef = cfg['coef']
args.restrain_dic = config[args.dataset].get('tail_restrain', None)
# print(cfg)

rv_dic = {}
for rv_name in coef:
    rv_para = coef[rv_name]
    # create a t distribution with df = rv_para[0], loc = rv_para[1], scale = rv_para[2]
    rv_dic[rv_name] = stats.t(df=rv_para[0], loc=rv_para[1], scale=rv_para[2])
print('rv distribution dic:', rv_dic)
os.makedirs(f'{args.output_folder}/hyperpath', exist_ok=True)
os.makedirs(f'{args.output_folder}/head', exist_ok=True)
os.makedirs(f'{args.output_folder}/tail', exist_ok=True)

# deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

class CustomFormatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""
    def converter(self, timestamp):
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            t = dt.strftime(self.default_time_format)
            s = self.default_msec_format % (t, record.msecs)
        return s
    
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(f'{args.output_folder}/my_app.log')
file_handler.setFormatter(CustomFormatter('%(asctime)s:%(levelname)s:%(message)s', '%H:%M:%S'))
# StreamHandler for logging to stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(CustomFormatter('%(asctime)s:%(levelname)s:%(message)s', '%H:%M:%S'))   # .%f

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

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

if os.path.exists(args.model_path):
    ech(f'loading models from path: {args.model_path}')
    model.load_state_dict(torch.load(args.model_path))
else:
    ech(f'model does not exists! {args.model_path}')
    
torch.save(model.state_dict(), f'{args.output_folder}/params.pth')
args.state_dict = torch.load(f'{args.output_folder}/params.pth')

# MAX_POST_TRAIN_TIMES = 5
MAX_POST_TRAIN_TIMES = 3
MAX_TRAINING_THRESH = 300
MAX_COMBINATION_SIZE = 4
DEFAULT_XSI_THRESHOLD = 0.2
DEFAULT_VALID_THRESHOLD = 0.02
MAKE_COMBINATION = False
BASE_ADDITION_ON_PT = False    
# IMPORTANT: BASE_ADDITION_ON_PT can not be True, because, training base on pt will surely decrease the score  
CALCULATE_REAL_REL = False
MAX_EMBEDDING_DIFF_L2 = 0.1
NORMALIZE_DIFF = False
MAX_GROUP_CNT = 3


if isinstance(model, ComplEx):
    kelpie_optimizer_class = KelpieMultiClassNLLOptimizer
elif isinstance(model, ConvE):
    kelpie_optimizer_class = KelpieBCEOptimizer
elif isinstance(model, TransE):
    kelpie_optimizer_class = KelpiePairwiseRankingOptimizer
else:
    kelpie_optimizer_class = KelpieMultiClassNLLOptimizer
kelpie_model_class = model.kelpie_model_class()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def rd(x):
    return np.round(x, 4)

def mean(lis):
    return rd(np.mean(lis))

def tensor_head(t):
    return [rd(x) for x in t.view(-1)[:3].detach().cpu().numpy().tolist()]

def std(lis):
    return rd(np.std(lis))

def get_removel_relevance(rank_delta, score_delta):
    if args.relevance_method == 'kelpie':
        relevance = float(rank_delta + sigmoid(score_delta))
    elif args.relevance_method == 'rank':
        relevance = rank_delta
    elif args.relevance_method == 'score':
        relevance = score_delta
    elif args.relevance_method == 'hybrid':
        relevance = np.tanh(rank_delta) + np.tanh(score_delta)
    return rd(relevance)

def extract_performances(model: Model, sample: numpy.array):
    model.eval()
    head_id, relation_id, tail_id = sample

    # print('[extract]trainable_indices', model.trainable_indices)
    # check how the model performs on the sample to explain
    all_scores = model.all_scores(numpy.array([sample])).detach().cpu().numpy()[0]
    target_entity_score = all_scores[tail_id] # todo: this only works in "head" perspective
    filter_out = model.dataset.to_filter[(head_id, relation_id)] if (head_id, relation_id) in model.dataset.to_filter else []

    if model.is_minimizer():
        all_scores[filter_out] = 1e6
        # if the target score had been filtered out, put it back
        # (this may happen in necessary mode, where we may run this method on the actual test sample;
        all_scores[tail_id] = target_entity_score
        best_entity_score = numpy.min(all_scores)
        target_entity_rank = numpy.sum(all_scores <= target_entity_score)  # we use min policy here

    else:
        all_scores[filter_out] = -1e6
        # if the target score had been filtered out, put it back
        # (this may happen in necessary mode, where we may run this method on the actual test sample;
        all_scores[tail_id] = target_entity_score
        best_entity_score = numpy.max(all_scores)
        target_entity_rank = numpy.sum(all_scores >= target_entity_score)  # we use min policy here

    return rd(target_entity_score), rd(best_entity_score), target_entity_rank

def extract_performances_on_embeddings(trainable_entities, embedding: torch.Tensor, prediction: numpy.array, grad:bool=False):
    new_model = TargetModel(dataset=dataset, hyperparameters=hyperparameters)
    new_model.load_state_dict(state_dict=args.state_dict)
    new_model = new_model.to('cuda')
    new_model.start_post_train(trainable_indices=trainable_entities, init_tensor=embedding)
    if grad:
        new_model.eval()
        return new_model.calculate_grad(prediction)
    return extract_performances(new_model, prediction)


def extract_samples_with_entity(samples, entity_id):
    return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]


def extract_training_samples_length(trainable_entities) -> np.array:
    original_train_samples = []
    for entity in trainable_entities:
        original_train_samples.extend(dataset.entity_id_2_train_samples[entity])
    # stack a list of training samples, each of them is a tuple
    return len(original_train_samples)

def mean_of_tensor_list(tensor_list):
    return torch.mean(torch.stack(tensor_list), dim=0)

def unfold(lis):
    # check if is tensor, if so, tolist
    if isinstance(lis, torch.Tensor):
        lis = lis.tolist()
    if not hasattr(lis, '__iter__'):
        return rd(lis)
    if len(lis) == 1:
        return unfold(lis[0])
    return [rd(x) for x in lis]

def get_path_entities(prediction, path):
    last_entity_on_path = prediction[0]
    path_entities = [last_entity_on_path]
    for triple in path:
        target = triple[2] if triple[0] == last_entity_on_path else triple[0]
        path_entities.append(target)
        last_entity_on_path = target
    return path_entities

def update_df(df, dic, save_path):
    for key in set(dic.keys()) - set(df.columns):
        df[key] = None
    df.loc[len(df)] = dic
    df.to_csv(f'{args.output_folder}/{save_path}', index=False)


def overlapping_block_division(neighbors, m):
    neighbors = list(neighbors)
    n = len(neighbors)
    k = math.ceil(math.log(n, m))
    N = m ** k
    cnt = n // m
    print(f"n: {n}, m: {m}, k: {k}, N: {N}, cnt: {cnt}")

    group_id_to_elements = {}
    element_id_to_groups = defaultdict(list)

    # fill neighbors with -1 until it has N elements
    neighbors += [-1] * (N - n)
    # create a k-dim matrix with m elements in each dimension, and fill it with the elements in neighbors
    matrix = np.array(neighbors).reshape((m,) * k)

    for i in range(k):
        # get m slices from the i-th dimension and store them in a list(group), group_id = m * i + j
        for j in range(m):
            group = matrix.take(j, axis=i).flatten()
            group_id = m * i + j
            group_id_to_elements[group_id] = [element for element in group if element != -1]
            for element in group_id_to_elements[group_id]:
                element_id_to_groups[element].append(group_id)

    return group_id_to_elements, element_id_to_groups

identifier2explanations = defaultdict(list)
base_identifier2next_triples = defaultdict(set)
base_identifier2trainable_entities_embedding = {}

class Explanation:
    _original_model_results = {}  # map original samples to scores and ranks from the original model
    _base_pt_model_results = {}   # map original samples to scores and ranks from the base post-trained model
    _kelpie_init_tensor_cache = OrderedDict()

    """
        Given a "sample to explain" (that is, a sample that the model currently predicts as true,
        and that we want to be predicted as false);
        and given and a list of training samples containing the entity to convert;
        compute the relevance of the samples in removal, that is, an estimate of the effect they would have
        if removed (all together) from the perspective entity to worsen the prediction of the sample to convert.

        :param prediction: the sample that we would like the model to predict as "true",
                                    in the form of a tuple (head, relation, tail)
        :param samples_to_remove:   the list of samples containing the perspective entity
                                    that we want to analyze the effect of, if added to the perspective entity
    """
    df = pd.DataFrame()
    
    def _get_kelpie_init_tensor(self):
        embeddings = []
        for entity in self.trainable_entities:
            if entity not in self._kelpie_init_tensor_cache:
                kelpie_init_tensor_size = model.dimension if not isinstance(model, TuckER) else model.entity_dimension
                self._kelpie_init_tensor_cache[entity] = torch.rand(1, kelpie_init_tensor_size, device='cuda') - 0.5   # 
            embeddings.append(self._kelpie_init_tensor_cache[entity])
        return torch.cat(embeddings, dim=0)
    
    def _extract_training_samples(self) -> np.array:
        original_train_samples = []
        for entity in self.trainable_entities:
            original_train_samples.extend(dataset.entity_id_2_train_samples[entity])
        # stack a list of training samples, each of them is a tuple
        self.original_train_samples = np.stack(original_train_samples, axis=0)
        ids = []
        for sample in self.samples_to_remove:
            ids.append(self.original_train_samples.tolist().index(list(sample)))
        self.pt_train_samples = np.delete(self.original_train_samples, ids, axis=0)
        self.base_train_samples = self.original_train_samples

        # self.empty_samples = np.delete(self.original_train_samples, list(range(len(self.original_train_samples))), axis=0)
    
    def _extract_training_samples_sharing_others(self) -> np.array:
        """extract training samples from the dataset for the trainable entities

        Returns:
            np.array: _description_
        """
        path_samples = set()
        original_train_samples = set()
        for entity in self.trainable_entities:
            for sample in dataset.entity_id_2_train_samples[entity]:
                original_train_samples.add(sample)

                target = sample[2] if sample[0] == entity else sample[0]
                if target in self.trainable_entities:
                    path_samples.add(sample)

        # self.ids = []
        # original_train_samples_list = list(original_train_samples)
        # original_train_samples_list.sort()
        # for sample in self.samples_to_remove:
        #     self.ids.append(original_train_samples_list.index(sample))

        if self.base_identifier in base_identifier2next_triples:
            next_triples_cnt = len(base_identifier2next_triples[self.base_identifier])
            base_train_samples = path_samples | base_identifier2next_triples[self.base_identifier] | set(self.samples_to_remove)
        else:
            next_triples_cnt = np.inf
            base_train_samples = original_train_samples
        
        pt_train_samples = base_train_samples - set(self.samples_to_remove)
        share_train_samples = original_train_samples - base_train_samples

        logger.info(f'base_identifier: {self.base_identifier} path: {len(path_samples)} next: {next_triples_cnt} original: {len(original_train_samples)} base: {len(base_train_samples)} pt: {len(pt_train_samples)} share: {len(share_train_samples)}')

        # stack a list of training samples, each of them is a tuple
        self.original_train_samples = np.stack(list(original_train_samples), axis=0)
        self.base_train_samples = np.stack(list(base_train_samples), axis=0)
        self.addition_train_samples = np.stack(list(self.samples_to_remove), axis=0)
        if len(pt_train_samples) == 0:
            self.pt_train_samples = []
        else:
            self.pt_train_samples = np.stack(list(pt_train_samples), axis=0)

        if len(share_train_samples) == 0:
            self.sharing = False
            return

        if self.base_identifier not in base_identifier2trainable_entities_embedding:    
            logger.info(f'[post-training on share] {len(share_train_samples)}/{len(self.original_train_samples)} samples, {hyperparameters[RETRAIN_EPOCHS]} epoches')
            self.sharing = False
            results = []
            for _ in range(MAX_POST_TRAIN_TIMES):
                results.append(self.post_training_save(np.stack(list(share_train_samples), axis=0)))
            base_identifier2trainable_entities_embedding[self.base_identifier] = mean_of_tensor_list(results)
            self.sharing = True
            target_entity_score, best_entity_score, target_entity_rank = self.extract_performances_on_embeddings(base_identifier2trainable_entities_embedding[self.base_identifier])
            logger.info(f'[post-training on share] target_entity_score: {target_entity_score}, best_entity_score: {best_entity_score}, target_entity_rank: {target_entity_rank}')
            

        # self.original_train_samples = np.stack(list(original_train_samples), axis=0)
        # self.ids = []
        # for sample in self.samples_to_remove:
        #     self.ids.append(self.original_train_samples.tolist().index(list(sample)))
        # self.pt_training_samples = np.delete(self.original_train_samples, self.ids, axis=0)

        # self.empty_samples = np.delete(self.original_train_samples, list(range(len(self.original_train_samples))), axis=0)

    @staticmethod
    def build(prediction: Tuple[Any, Any, Any],
                 samples_to_remove: List[Tuple],
                 trainable_entities: List=None, sharing: bool=True):
        explanation = Explanation(prediction, samples_to_remove, trainable_entities, sharing)
        if explanation.identifier in identifier2explanations:
            return identifier2explanations[explanation.identifier]

        explanation.calculate_relevance()

        if len(explanation.samples_to_remove) <= MAX_COMBINATION_SIZE: 
            identifier2explanations[explanation.identifier] = explanation

        update_df(Explanation.df, explanation.ret, "output_details.csv")
        logger.info(f"Explanation created. {str(explanation.ret)}")
        return explanation

    def __init__(self, 
                 prediction: Tuple[Any, Any, Any],
                 samples_to_remove: List[Tuple],
                 trainable_entities: List=None, sharing: bool=True) -> None:
        logger.info("Create Explanation on sample: %s, removing: %s", prediction, samples_to_remove)
        # logger.info("Removing sample: %s", [dataset.sample_to_fact(x, True) for x in samples_to_remove])
        # for entity_id, samples in samples_to_remove.items():
        #     print("Entity:", dataset.entity_id_to_name(entity_id), "Samples:", [dataset.sample_to_fact(x, True) for x in samples])

        self.prediction = prediction
        self.samples_to_remove = samples_to_remove
        if trainable_entities is None:
            trainable_entities = [prediction[0]]

        self.head = prediction[0]
        # trainable_entities.sort()   # WE SHOULD NOT SORT HERE, BECAUSE THE ORDER OF TRAINABLE ENTITIES MATTERS
        self.trainable_entities = trainable_entities
        self.kelpie_init_tensor = self._get_kelpie_init_tensor()
        self.base_identifier = (tuple(self.prediction), tuple(self.trainable_entities))

        samples_to_remove.sort()
        self.remove_hash = sum([hash(x) for x in samples_to_remove])
        self.identifier = (tuple(self.prediction), tuple(self.trainable_entities), self.remove_hash)
        
        logger.info('extracting training samples...')
        self.sharing = sharing
        if sharing:
            self._extract_training_samples_sharing_others()
        else:
            self._extract_training_samples()
        # self.identifier = (tuple(self.prediction), tuple(self.trainable_entities), tuple(self.ids))
        self.paths = []    # only path (simple/compound) explanation has path
        

    def calculate_relevance(self):
        # if self.pt_training_samples.shape[0] > MAX_TRAINING_THRESH:
        #     print(f'cost of computing on {self.trainable_entities} is too large. Avoiding...')
        #     self.relevance = 0
        #     self.ret = {}
        #     return

        # print('[Explanation]origin score', self.original_results())

        # print('[Explanation]no post training:')
        # self.post_training_multiple(self.empty_samples)
        # create a numpy array of shape (0, 3) to avoid post training
        if BASE_ADDITION_ON_PT:
            pt_training = self.pt_training_addition
            base_training = self.base_training_addition
        else:
            pt_training = self.pt_training_multiple
            base_training = self.base_training_multiple

        pt_target_entity_score, \
        pt_best_entity_score, \
        pt_target_entity_rank, \
        pt_embeddings = pt_training()
        
        base_target_entity_score, \
        base_best_entity_score, \
        base_target_entity_rank, \
        base_embeddings = base_training()

        diff = pt_embeddings - base_embeddings

        if NORMALIZE_DIFF:
            delta_l2 = torch.norm(diff, p=2, dim=1)

            # print('delta embeddings', diff.shape)
            # print('delta_l2', delta_l2)

            # normalize the diff to have the same l2 norm as the original diff
            for i in range(len(delta_l2)):
                if delta_l2[i] > MAX_EMBEDDING_DIFF_L2:
                    diff[i] /= delta_l2[i] / MAX_EMBEDDING_DIFF_L2
            pt_embeddings = base_embeddings + diff
            pt_target_entity_score, \
            pt_best_entity_score, \
            pt_target_entity_rank = self.extract_performances_on_embeddings(pt_embeddings)
        # logger.info(f"Explanation created. Rank worsening: {rank_worsening}, score worsening: {score_worsening}")
        
        rank_worsening = pt_target_entity_rank - base_target_entity_rank
        score_worsening = base_target_entity_score - pt_target_entity_score
        if model.is_minimizer():
            score_worsening *= -1

        self.pt_embeddings = pt_embeddings
        self.base_score = base_target_entity_score
        self.pt_score = pt_target_entity_score

        # calculate the gradient using the average of the two embeddings
        self.grad = self.extract_performances_on_embeddings((pt_embeddings + base_embeddings) / 2, True)

        self.relevance = get_removel_relevance(rank_worsening, score_worsening)
        self.ret = {'prediction': dataset.sample_to_fact(self.prediction, True),
                'identifier': self.identifier,
                # 'samples_to_remove': self.samples_to_remove,
                'length': len(self.samples_to_remove),
                'base_score': base_target_entity_score,
                'base_best': base_best_entity_score,
                'base_rank': base_target_entity_rank,
                'pt_score': pt_target_entity_score,
                'pt_best': pt_best_entity_score,
                'pt_rank': pt_target_entity_rank,
                'rank_worsening': rank_worsening,
                'score_worsening': score_worsening,
                'relevance': self.relevance,
                'pt_delta_2': unfold(torch.norm(pt_embeddings - self.get_init_tensor(), p=2)),
                'base_delta_2': unfold(torch.norm(base_embeddings - self.get_init_tensor(), p=2)),
                'delta_1': unfold(torch.norm(diff, p=1, dim=1)),
                'delta_2': unfold(torch.norm(diff, p=2, dim=1)),
                'delta_inf': unfold(torch.norm(diff, p=float('inf'), dim=1)),
                **self.grad
        }


    def base_training_addition(self):
        lr = hyperparameters[LEARNING_RATE] / 10
        epoches = int(hyperparameters[RETRAIN_EPOCHS] / 10)
        logger.info(f'[base_training_addition] {len(self.addition_train_samples)}/{len(self.original_train_samples)} samples, {epoches} epoches, lr={lr}')
        results = []
        for i in range(MAX_POST_TRAIN_TIMES):
            results.append(self.post_training_save(self.addition_train_samples, init_tensor=self.pt_train_embeddings, lr=lr, epoches=epoches))
        target_entity_score, \
        best_entity_score, \
        target_entity_rank, \
        delta_embeddings = zip(*results)
        logger.info(f'train {i+1} times, score: {mean(target_entity_score)} ± {std(target_entity_score)}, best: {mean(best_entity_score)} ± {std(best_entity_score)}, rank: {mean(target_entity_rank)} ± {std(target_entity_rank)}')
        return mean(target_entity_score), mean(best_entity_score), mean(target_entity_rank), mean_of_tensor_list(delta_embeddings)
    
    def pt_training_addition(self):
        logger.info(f'[pt_training_addition] {len(self.pt_train_samples)}/{len(self.original_train_samples)} samples, {hyperparameters[RETRAIN_EPOCHS]} epoches')
        return self.post_training_save(self.pt_train_samples, save_pt_train=True)

    def base_training_multiple(self):
        if self.base_identifier in self._base_pt_model_results:
            return self._base_pt_model_results[self.base_identifier]

        logger.info(f'[base_training_multiple] {len(self.base_train_samples)}/{len(self.original_train_samples)} samples, {hyperparameters[RETRAIN_EPOCHS]} epoches')
        self._base_pt_model_results[self.base_identifier] = self.post_training_multiple(self.base_train_samples)
        return self._base_pt_model_results[self.base_identifier]

    def pt_training_multiple(self):
        logger.info(f'[pt_training_multiple] {len(self.pt_train_samples)}/{len(self.original_train_samples)} samples, {hyperparameters[RETRAIN_EPOCHS]} epoches')
        return self.post_training_multiple(self.pt_train_samples, early_stop=True)

    def is_valid(self):
        return self.relevance >= DEFAULT_VALID_THRESHOLD

    def post_training_multiple(self, training_samples, early_stop=False):
        results = []
        for i in range(MAX_POST_TRAIN_TIMES):
            results.append(self.post_training_save(training_samples))
            # if len(results) > 1 and early_stop:
            #     # if the CV of the score and rank is small enough, stop training
            #     target_entity_score, _, target_entity_rank, _ = zip(*results)
            #     if std(target_entity_score) / mean(target_entity_score) < 0.1 and std(target_entity_rank) / mean(target_entity_rank) < 0.1:
            #         break
            
        embeddings = mean_of_tensor_list(results)
        target_entity_score, \
        best_entity_score, \
        target_entity_rank = self.extract_performances_on_embeddings(embeddings)
        logger.info(f'train {i+1} times, score: {target_entity_score}, best: {best_entity_score}, rank: {target_entity_rank}')
        return mean(target_entity_score), mean(best_entity_score), mean(target_entity_rank), embeddings

    def original_results(self) :
        sample = self.prediction
        if not sample in self._original_model_results:
            target_entity_score, \
            best_entity_score, \
            target_entity_rank = extract_performances(model, sample)
            self._original_model_results[sample] = (target_entity_score, best_entity_score, target_entity_rank)
        return self._original_model_results[sample]

    
    def get_init_tensor(self):
        if self.sharing:
            init_tensor = base_identifier2trainable_entities_embedding[self.base_identifier]
            # IMPORTANT: initialize trainable_entities_embedding
        else:
            init_tensor = model.entity_embeddings[self.trainable_entities].clone().detach()
            init_tensor += self.kelpie_init_tensor
            # init_tensor = self.kelpie_init_tensor
        return init_tensor

    
    def post_training_save(self, post_train_samples: numpy.array=[], **para):
        new_model = TargetModel(dataset=dataset, hyperparameters=hyperparameters)
        new_model.load_state_dict(state_dict=args.state_dict)
        new_model = new_model.to('cuda')
        
        init_tensor = self.get_init_tensor()
        # if len(post_train_samples) == 0:
        #     return extract_performances(new_model, self.prediction)

        for param in new_model.parameters():
            if param.is_leaf:
                param.requires_grad = False
        
        # frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        # trainable_head_embedding = torch.nn.Parameter(self.kelpie_init_tensor, requires_grad=True)
        # frozen_entity_embeddings[self.head] = trainable_head_embedding
        # print(type(frozen_entity_embeddings))
        # model.entity_embeddings.requires_grad = True
        # model.frozen_indices = [i for i in range(model.entity_embeddings.shape[0]) if i != self.head]

        new_model.start_post_train(trainable_indices=self.trainable_entities, init_tensor=init_tensor)   

        # print('origin', extract_performances(new_model, self.prediction))
        # print('weight', tensor_head(new_model.convolutional_layer.weight))
        print('before embedding %s', tensor_head(new_model.trainable_entity_embeddings[0]))
        # print('other embedding', tensor_head(new_model.entity_embeddings[self.head+1]))

        if len(post_train_samples):
            # Now you can do your training. Only the entity_embeddings for self.head will get updated...
            optimizer = Optimizer(model=new_model, hyperparameters=hyperparameters, verbose=False)
            optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]
            if 'epoches' in para:
                optimizer.epochs = para['epoches']
            if 'lr' in para:
                optimizer.learning_rate = para['lr']
            
            # optimizer.learning_rate *= 10

            # print('original, post_train = ', len(original_train_samples), len(post_train_samples))
            optimizer.train(train_samples=post_train_samples, post_train=True)
        
        print('after embedding %s', tensor_head(new_model.trainable_entity_embeddings[0]))
        # print('weight', tensor_head(new_model.convolutional_layer.weight))
        # print('other embedding', tensor_head(new_model.entity_embeddings[self.head+1]))

        # ret = extract_performances(new_model, self.prediction)
        # print('pt', ret)

        # delta_embedding = new_model.trainable_entity_embeddings.clone().detach() - init_tensor
        # return *ret, delta_embedding,   # (#trainable_entities, #embedding_dim)
        return new_model.trainable_entity_embeddings.clone().detach()
    
    def extract_performances_on_embeddings(self, embedding: torch.Tensor, grad:bool = False):
        return extract_performances_on_embeddings(self.trainable_entities, embedding, self.prediction, grad)
    
    
from prefilters.prefilter import PreFilter
import threading
from multiprocessing.pool import ThreadPool as Pool
from config import MAX_PROCESSES
from dataset import MAX_PATH_LENGTH
class DividePrefilter(PreFilter):
    """The DividePrefilter divide all relevance into 2 groups and calculate relevance of both. If one group is valid, continue dividing.
    Ending criterion: for some layer, the total count of triples is no greater than M. 
    If the last layer has more than M valid triples, return the top M relevance triple.

    Args:
        PreFilter (_type_): _description_
    """
    df = pd.DataFrame()
    def __init__(self,
                 model: Model,
                 dataset: Dataset):
        """
        PostTrainingPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.max_path_length = 5
        self.threadLock = threading.Lock()
        self.counter = 0
        self.thread_pool = Pool(processes=MAX_PROCESSES)

    def top_promising_explanations(self,
                                  prediction:Tuple[Any, Any, Any],
                                  incomplete_path_entities: List=None,
                                  top_N=20,) -> list[Explanation]:
        """return top N k-hop simple explanations 
        k = len(incomplete_path_entities)

        Args:
            prediction (Tuple[Any, Any, Any]): _description_
            incomplete_path_entities (List, optional): entities on the incomplete path. Starts with head.
            top_N (int, optional): _description_. Defaults to 20.

        Returns:
            list[Explanation]: _description_
        """
        
        h, r, t = prediction
        if incomplete_path_entities is None:
            incomplete_path_entities = [h]
        # if extract_training_samples_length(incomplete_path_entities) > MAX_TRAINING_THRESH:
        #     # TODO: how to calculate the training set too large efficiently?
        #     logger.info(f'cost of computing on {incomplete_path_entities} is too large. Avoiding...')
        #     return []
        
        next_hop_triples = base_identifier2next_triples[(tuple(prediction), tuple(incomplete_path_entities))]
        explanations = []  
        # if len(incomplete_path_entities) == 1:
        #     # first add all 1-hop/2-hop triples. We do not need explanation to be valid here
        #     triples_hop3 = []
        #     for triple in next_hop_triples:
        #         target = triple[2] if triple[0] == h else triple[0]
        #         if target == t or target in dataset.entity_id_2_neighbors[t]:
        #             exp = Explanation.build(prediction, [triple], incomplete_path_entities)
        #             if exp.is_valid():
        #                 explanations.append(exp)
        #         else:
        #             triples_hop3.append(triple)
        #     next_hop_triples = triples_hop3
        #     logger.info(f'hop1-2 triples ({len(explanations)}) {[exp.relevance for exp in explanations]}')
        #     logger.info(f'hop3 triples ({len(next_hop_triples)})')

        # if the number of triples is less than top_k, no need to divide
        if len(next_hop_triples) <= top_N:
            logger.info(f'latent triples {len(next_hop_triples)} <= {top_N}. no need to divide')
            for triple in next_hop_triples:
                exp = Explanation.build(prediction, [triple], incomplete_path_entities)
                if exp.is_valid():
                    explanations.append(exp)
            
            explanations.sort(key=lambda x: x.relevance, reverse=True)
            return explanations[:top_N]

        # construct overall explanation
        overall_explanation = Explanation.build(prediction, list(next_hop_triples), incomplete_path_entities)
        if not overall_explanation.is_valid():
            logger.info('overall explanation is not valid. No valid explanations for current sample')
            return []
        
        # divide triples into groups of the same length randomly 
        logger.info('constructing groups...')  
        groups = [overall_explanation]
        while sum([len(group.samples_to_remove) for group in groups]) > top_N:
            new_groups = []
            for group in groups:
                if len(group.samples_to_remove) <= 1:
                    new_groups.append(group)
                else:
                    new_groups.extend(self.search_valid_groups(group))
            if len(new_groups) == len(groups):
                break
            groups = new_groups

        # combine all groups
        for group in groups:
            if len(group.samples_to_remove) == 1:
                explanations.append(group)
            else:
                for sample in group.samples_to_remove:
                    exp = Explanation.build(prediction, [sample], incomplete_path_entities)
                    if exp.is_valid():
                        explanations.append(exp)

        explanations.sort(key=lambda x: x.relevance, reverse=True)
        return explanations[:top_N]
        

    def search_valid_groups(self, exp: Explanation):
        # divide triples into 2 groups of the same length randomly
        triples = exp.samples_to_remove
        random.shuffle(triples)
        group1 = triples[:len(triples)//2]
        group2 = triples[len(triples)//2:]
        print('divided into 2 groups')
        print('group1', len(group1), group1)
        print('group2', len(group2), group2)

        group1_explanation = Explanation.build(exp.prediction, group1, exp.trainable_entities)
        group2_explanation = Explanation.build(exp.prediction, group2, exp.trainable_entities)

        valid_groups = []
        if group1_explanation.is_valid():
            valid_groups.append(group1_explanation)
        if group2_explanation.is_valid():
            valid_groups.append(group2_explanation)

        update_df(self.df, {
            'prediction': exp.prediction,
            'identifier': exp.identifier,
            'length': len(exp.samples_to_remove),
            'r1': group1_explanation.relevance,
            'r2': group2_explanation.relevance,
            'relevance': exp.relevance
        }, 'divide_prefilter.csv')

        return valid_groups
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (numpy.bool_,)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


class Path:
    paths = []
    rel_df = pd.DataFrame()

    def __init__(self, prediction, path) -> None:
        """Constructor for Path

        Args:
            prediction (_type_): _description_
            path (_type_): a list of triples connecting head and tail. The triples should be in the same order as the path
        """
        self.prediction = prediction
        self.path = path
        self.path_entities = self.get_path_entities()

        head, rel, tail = self.prediction
        all_triples = set()
        for entitiy in self.path_entities:
            all_triples.update(args.available_samples[entitiy])
        first_last_hop_triples = set()
        for entitiy in [head, tail]:
            first_last_hop_triples.update(args.available_samples[entitiy])

        base_identifier2next_triples[(tuple(self.prediction), tuple(self.path_entities))] = all_triples
        base_identifier2next_triples[(tuple(self.prediction), tuple([head]))] = args.available_samples[head]
        base_identifier2next_triples[(tuple(self.prediction), tuple([tail]))] = args.available_samples[tail]
        base_identifier2next_triples[(tuple(self.prediction), tuple([head, tail]))] = first_last_hop_triples

        self.build_explanations()
        self.explanations = [self.first_hop_exp, self.last_hop_exp, self.path_exp, self.real_exp]

        logger.info(f'path relevance: {[exp.relevance for exp in self.explanations if exp is not None]}')

        self.save_to_local()

    @property
    def triples(self):
        return self.path

    def save_to_local(self):
        self.paths.append(self)
        self.ret = {
            'prediction': self.prediction,
            'path': self.path,
            'head_rel': self.first_hop_exp.relevance,
            'tail_rel': self.last_hop_exp.relevance,
            'rel': self.path_exp.relevance,
            'real_rel': self.real_exp.relevance if CALCULATE_REAL_REL else None,
        }
        for p in [1, 2, float('inf')]:
            self.ret.update({
                f'delta_h_{p}': self.first_hop_exp.ret[f'delta_{p}'],
                f'delta_t_{p}': self.last_hop_exp.ret[f'delta_{p}'],
                f'delta_{p}': self.path_exp.ret[f'delta_{p}'],
                f'delta_real_{p}': self.real_exp.ret[f'delta_{p}'] if CALCULATE_REAL_REL else None,
                f'partial_{p}': self.path_exp.ret[f'partial_{p}'],
                f'partial_t_{p}': self.path_exp.ret[f'partial_t_{p}'],
                f'partial_h_{p}': self.path_exp.ret[f'partial_h_{p}']
            })
        update_df(self.rel_df, self.ret,'rel_df.csv')

        lis = [path.json() for path in self.paths]
        with open(f'{args.output_folder}/paths.json', 'w') as f:
            json.dump(lis, f, cls=NumpyEncoder, indent=4)

    def get_path_entities(self):
        return get_path_entities(self.prediction, self.path)

    def build_explanations(self):
        head, rel, tail = self.prediction
        self.first_hop_exp = Explanation.build(self.prediction, [self.path[0]], [head])
        self.last_hop_exp = Explanation.build(self.prediction, [self.path[-1]], [tail])
        self.path_exp = Explanation.build(self.prediction, [self.path[0], self.path[-1]], [head, tail])
        self.real_exp = Explanation.build(self.prediction, self.path, self.path_entities) if CALCULATE_REAL_REL else None

    def json(self):
        return {
            'prediction': self.prediction,
            'facts': [dataset.sample_to_fact(triple, True) for triple in self.triples],
            'path': self.path,
            'first_hop_exp': self.first_hop_exp.ret,
            'last_hop_exp': self.last_hop_exp.ret,
            'path_exp': self.path_exp.ret,
            'real_exp': self.real_exp.ret if CALCULATE_REAL_REL else None,
            'relevance': [exp.relevance for exp in self.explanations if exp is not None],
            'ret': self.ret
        }
        

class SuperPath(Path):
    def __init__(self, prediction, path) -> None:
        super().__init__(prediction, path)

    def get_path_entities(self):
        return self.path

    @property
    def triples(self):
        return self.all_samples

    def build_explanations(self):
        head, rel, tail = self.prediction
        first_hop = self.path[1]
        last_hop = self.path[-2]
        print('Constructing super path:  first_hop', first_hop, 'last_hop', last_hop, 'tail', tail)
        print('last_hop samples', len(args.available_samples[last_hop]), list(args.available_samples[last_hop])[:10])
        print('tail samples', len(args.available_samples[tail]), list(args.available_samples[tail])[:10])
        
        first_hop_samples = list(args.available_samples[head] & args.available_samples[first_hop])
        last_hop_samples = list(args.available_samples[tail] & args.available_samples[last_hop])

        for sample in last_hop_samples:
            print(sample, tail, last_hop)
            assert sample[0] in [tail, last_hop] and sample[2] in [tail, last_hop]

        self.all_samples = set()
        for i in range(len(self.path)-1):
            entity = self.path[i]
            for triple in args.available_samples[entity]:
                if self.path[i+1] in [triple[0], triple[2]]:
                    self.all_samples.add(triple)

        self.first_hop_exp = Explanation.build(self.prediction, first_hop_samples, [head])
        self.last_hop_exp = Explanation.build(self.prediction, last_hop_samples, [tail])
        self.path_exp = Explanation.build(self.prediction, first_hop_samples + last_hop_samples, [head, tail])
        self.real_exp = Explanation.build(self.prediction, list(self.all_samples), self.path) if CALCULATE_REAL_REL else None


class Xrule:
    def __init__(self,
                 model: Model,
                 dataset: Dataset) -> None:
        
        self.model = model
        self.dataset = dataset
        self.prefilter = DividePrefilter(model, dataset)

    def explain_necessary(self,
                          prediction: Tuple[Any, Any, Any]):
        """This method extracts necessary explanations for a specific sample

        Args:
            prediction (Tuple[Any, Any, Any]): _description_
        """
        h, r, t = prediction
        return self.get_k_hop_path(prediction)
    
    def get_k_hop_path(self, prediction, incomplete_path_entities=None, last_sample=[], last_exp=[]):
        """This method extracts necessary paths for a specific sample, k <= hops <= MAX_PATH_LENGTH
        k = len(incomplete_path_entities)
        len(incomplete_path_entities)=0 => one_hop_path
        len(incomplete_path_entities)=1 => two_hop_path
        len(incomplete_path_entities)=2 => three_hop_path

        Args:
            prediction (_type_): _description_
            incomplete_path_entities (list, optional): _description_. Defaults to [].
        """
        h, r, t = prediction
        if incomplete_path_entities is None:
            incomplete_path_entities = [h]
        k = len(incomplete_path_entities)
        prefix = '!'*k
        logger.info(f'{prefix} explanaing {k}-hop path on {prediction}: {self.dataset.sample_to_fact(prediction, True)}')
        logger.info(f'{prefix} incomplete_path_entities: {incomplete_path_entities}, last_sample: {last_sample}')
        
        remain_length = MAX_PATH_LENGTH - k + 1
        all_paths = self.dataset.find_all_path_within_k_hop(incomplete_path_entities[-1], t,
                                                            k=remain_length,
                                                            forbidden_entities=incomplete_path_entities)
        triples = set([tuple(path[0]) for path in all_paths])
        logger.info(f'latent triples({len(triples)}) in path({len(all_paths)}): {triples}')
        base_identifier2next_triples[(tuple(prediction), tuple(incomplete_path_entities))] = triples
        # logger.info(f'base_identifier2next_triples: {(tuple(prediction), tuple(incomplete_path_entities))} = {base_identifier2next_triples}')

        k_hop_explanations = self.prefilter.top_promising_explanations(prediction, incomplete_path_entities)

        all_paths = []
        for k_hop_exp in k_hop_explanations:
            assert len(k_hop_exp.samples_to_remove) == 1
            k_hop_sample = k_hop_exp.samples_to_remove[-1]
            k_hop_target = k_hop_sample[2] if k_hop_sample[0] == incomplete_path_entities[-1] else k_hop_sample[0]
            if k_hop_target == t:
                all_paths.append(Path(last_sample + [k_hop_sample], last_exp + [k_hop_exp]))
                continue

            all_paths.extend(self.get_k_hop_path(prediction, 
                                                 incomplete_path_entities + [k_hop_target], 
                                                 last_sample + [k_hop_sample], 
                                                 last_exp + [k_hop_exp]))
            
        if k == 1 and MAKE_COMBINATION:
            one_hop_combinations = Combination(k_hop_explanations)
            path_combinations = Combination([p.path_explanation for p in all_paths])
            os.makedirs(f'{args.output_folder}/prediction', exist_ok=True)
            json.dump({
                    'one_hop': one_hop_combinations.json(),
                    'path': path_combinations.json()
                }, open(f'{args.output_folder}/prediction/{prediction}.json', 'w'), indent=4, cls=NumpyEncoder)
            
        return all_paths
    

import itertools
class Combination:

    def __init__(self, explanations: List[Explanation], top_N=10) -> None:
        logger.info(f'building compound explanations on {len(explanations)} explanations')
        explanations.sort(key=lambda x: x.relevance, reverse=True)
        self.explanations = explanations
        self.window_size = 10
        self.top_N = top_N

        self.combination = self.build_combination()

    def json(self):
        lis = []
        for exp in self.combination:
            if len(exp.paths):
                lis.append({
                    **exp.ret,
                    'paths': [path.json() for path in exp.paths]
                })
        return lis


    def build_combination(self):
        """find top_N combination of explanations 

        Args:
            prediction (Tuple[Any, Any, Any]): _description_
        """
        all_explanations = []
        all_explanations.extend(self.explanations)

        if len(self.explanations) == 0:
            return all_explanations

        if self.explanations[0].relevance > DEFAULT_XSI_THRESHOLD:
            logger.info(f'Early terminate at length 1: {self.explanations[0].relevance} > {DEFAULT_XSI_THRESHOLD}')
            return self.explanations

        for cur_rule_length in range(2, min(len(self.explanations), MAX_COMBINATION_SIZE) + 1):
            compound_explanations = self.build_combination_with_length_k(cur_rule_length)

            if len(compound_explanations) == 0:
                continue

            compound_explanations.sort(key=lambda x: x.relevance, reverse=True)
            all_explanations.extend(compound_explanations)

            if compound_explanations[0].relevance > DEFAULT_XSI_THRESHOLD:
                logger.info(f'Early terminate at length {cur_rule_length}: {compound_explanations[0].relevance} > {DEFAULT_XSI_THRESHOLD}')
                break
        
        all_explanations.sort(key=lambda x: x.relevance, reverse=True)
        return all_explanations[:self.top_N]


    def build_combination_with_length_k(self, k):
        """find all combination of explanations 

        Args:
            prediction (Tuple[Any, Any, Any]): _description_
        """
        logger.info(f"{'*'*k} building compound explanations({k})")
        all_possible_combinations = list(itertools.combinations(self.explanations, k))
        all_possible_combinations.sort(key=lambda x: sum([exp.relevance for exp in x]), reverse=True)

        compound_explanations = []
        terminate = False
        best_relevance_so_far = -1e6  # initialize with an absurdly low value

        # initialize the relevance window with the proper size
        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        for combination in all_possible_combinations:
            sample_to_remove = []
            for exp in combination:
                for sample in exp.samples_to_remove:
                    if sample not in sample_to_remove:
                        sample_to_remove.append(sample)
            trainable_entities = []
            for exp in combination:
                for entity in exp.trainable_entities:
                    if entity not in trainable_entities:
                        trainable_entities.append(entity)
            paths = []
            for exp in combination:
                paths.extend(exp.paths)
            explanation = Explanation.build(combination[0].prediction, sample_to_remove, trainable_entities)
            explanation.paths = paths

            if not explanation.is_valid():
                continue
            compound_explanations.append(explanation)
            logger.info(f"compound explanation({k}) created: {explanation.relevance} {explanation.trainable_entities} {explanation.samples_to_remove}")
            sliding_window [i % self.window_size] = explanation.relevance

            if explanation.relevance > DEFAULT_XSI_THRESHOLD:
                logger.info(f'Early terminate at {i} explanations: {explanation.relevance} > {DEFAULT_XSI_THRESHOLD}')
                break

            if i < self.window_size or explanation.relevance >= best_relevance_so_far:
                i += 1
                continue
            
            continue_prob = np.mean(sliding_window) / best_relevance_so_far

            if random.random() > continue_prob:
                logger.info(f'terminate at {i} explanations, valid explanation: {len(compound_explanations)}, continue prob: {continue_prob}')
                break

        return compound_explanations
    

class Generator:
    def __init__(self, prediction) -> None:
        self.prediction = prediction
        self.windows = []
        self.queue = PriorityQueue()
        self.upperbound_dic = {}

    def finished(self):
        """Use a sliding window for all generators (window size=5). 
        Records all the relevance generated, 
        continue with the probability of mean of relevance in the recent window 
        dividing the largest relevance, else finish.

        Returns:
            _type_: _description_
        """
        if self.empty():
            return True
        if len(self.windows) < 5:
            return False
        prob = mean(self.windows[-5:]) / max(self.windows)
        # generate a random number between 0 and 1
        if random.random() > prob:
            return True
        return False
    
    def generate(self):
        pass

    def json(self):
        pass

    def empty(self):
        return self.queue.empty()



class OneHopGenerator(Generator):
    df = pd.DataFrame()
    def __init__(self, perspective, prediction, neighbors) -> None:
        """For given perspective, generate top one hop explanation 
        (1) k-dim cross partition
        (2) select top valid probability ph/pt, calculate and yield explanation

        Args:
            perspective (stinr): 'head' or 'tail'
            prediction (tuple): sample to explain
            neighbors (list[int]): neighbors of the perspective entity
        """
        super().__init__(prediction)

        self.perspective = perspective
        self.prediction = prediction
        self.neighbors = neighbors
        self.one_hop_explanations = {}
        self.entity = prediction[0] if perspective == 'head' else prediction[2]
        self.ele2exp = defaultdict(list)
        
        if len(neighbors) <= 20:
            for neighbor in neighbors:
                exp = self.calculate_group([neighbor])
                self.ele2exp[neighbor] = [exp]
                self.upperbound_dic[neighbor] = exp.relevance
                self.queue.put((-exp.relevance, neighbor))
            # self.neighbors.sort(key=lambda x: self.ele2exp[x][0].relevance, reverse=True)
            logger.info('==========no need to ODB==========')
            for neighbor in self.neighbors:
                logger.info(f'{neighbor}: {self.upperbound_dic[neighbor]}')

        else:
            m = max(math.ceil(len(neighbors) / 30), 3)
            group_id_to_elements, element_id_to_groups = overlapping_block_division(self.neighbors, m)

            print(self.neighbors)
            print(m)

            logger.info(f'==========need to ODB==========, n: {len(neighbors)}, m: {m}')

            for group_id, group in group_id_to_elements.items():
                if len(group) == 0:
                    continue
                exp = self.calculate_group(group)
                exp.group_id = group_id
                
                for element_id in group:
                    self.ele2exp[element_id].append(exp)
            
            for neighbor in self.neighbors:
                self.calculate_upperbound(neighbor)
            # self.neighbors.sort(key=lambda x: self.calculate_probability(x), reverse=True)
            logger.info('==========ODB completed, probability==========')
            for neighbor in self.neighbors:
                logger.info(f'{neighbor}: {self.upperbound_dic[neighbor]}')
    

    def calculate_upperbound(self, neighbor):
        """calculate valid probability of a list of explanations
        P(R_hi>x) = P(xi-d_x * eta_h > x) * P(yi-d_y * eta_h > x) * ...
                = P(eta_h < (xi-x)/d_x) * P(eta_h < (yi-x)/d_y) * ...
                = Fh( (xi-x)/d_x ) * Fh( (yi-x)/d_y ) * ...
        where Fh is the CDF of eta_h.

        Returns:
            float: upperbound
        """
        explanations = self.ele2exp[neighbor]
        lab = self.persective[0]
        ret = np.inf
        for exp in explanations:
            rel_group = exp.relevance
            delta_group = exp.ret['delta_inf'] * exp.ret[f'parital_{lab}_all_inf']
            # point = (rel_group - DEFAULT_VALID_THRESHOLD) / delta_group
            # ret *= rv_dic[f'eta_{lab}'].cdf(point)
            upperbound = rel_group - delta_group * coef[f'g_{lab}']

            
        
        self.upperbound_dic[neighbor] = ret
        self.queue.put((-ret, neighbor))
        return ret
    


    def generate(self):
        """return one top explanation
        """
        neg_prob, neighbor = self.queue.get()
        assert neighbor not in self.one_hop_explanations

        if len(self.ele2exp[neighbor]) != 1:
            exp = self.calculate_group([neighbor])
        else:
            exp = self.ele2exp[neighbor][0]
        self.one_hop_explanations[neighbor] = exp

        update_df(self.df, 
            {
                'neighbor': neighbor,
                'entity': self.entity,
                'perspective': self.perspective,
                'prediction': self.prediction,
                'relevance': exp.relevance,
                'probability': self.upperbound_dic[neighbor],
                'rank': self.neighbors.index(neighbor)
            }, f'{self.perspective}_explanations.csv')
        
        with open(f'{args.output_folder}/{self.perspective}/{self.prediction}.json', 'w') as f:
            json.dump(self.json(), f, indent=4)
        
        self.windows.append(exp.relevance)
        return neighbor, exp
            
    
    def json(self):
        ret = {}
        for neighbor, exp in self.one_hop_explanations.items():
            ret[neighbor] = {
                'neighbor': neighbor,
                'entity': self.entity,
                'relevance': exp.relevance,
                'probability': self.upperbound_dic[neighbor],
                'rank': self.neighbors.index(neighbor),
                'exp': exp.json()
            }
        ret = sorted(ret.items(), key=lambda x: x[1]['relevance'], reverse=True)
        return ret
    
    def calculate_group(self, group):
        all_samples = set()
        for neighbor in group:
            all_samples |= args.available_samples[neighbor] & args.available_samples[self.entity]
        logger.info(f'all samples: {len(all_samples)}, group: {len(group)}')
        return Explanation.build(self.prediction, list(all_samples), [self.entity])


class PathGenerator(Generator):
    df = pd.DataFrame()
    def __init__(self, prediction, hyperpaths) -> None:
        super().__init__(prediction)

        self.prediction = prediction
        self.hyperpaths = hyperpaths
        self.path_explanations = {}
        self.head_explanations = {}
        self.tail_explanations = {}
        self.head_hyperpaths = defaultdict(set)
        self.tail_hyperpaths = defaultdict(set)

        for hyperpath in hyperpaths:
            self.head_hyperpaths[hyperpath[1]].add(hyperpath)
            self.tail_hyperpaths[hyperpath[-2]].add(hyperpath)

        self.probability_dic = {}
        self.approx_rel_dic = {}

    def renew_head(self, head, explanation):
        if head in self.head_explanations:
            return
        self.head_explanations[head] = explanation
        for hyperpath in self.head_hyperpaths[head]:
            if hyperpath[-2] in self.tail_explanations and hyperpath not in self.path_explanations:
                self.add_to_queue(hyperpath)


    def renew_tail(self, tail, explanation):
        if tail in self.tail_explanations:
            return
        self.tail_explanations[tail] = explanation
        for hyperpath in self.tail_hyperpaths[tail]:
            if hyperpath[1] in self.head_explanations and hyperpath not in self.path_explanations:
                self.add_to_queue(hyperpath)


    def add_to_queue(self, hyperpath):
        self.path_explanations[hyperpath] = 1   # wait to be calculated
        prob = 1

        head_exp = self.head_explanations[hyperpath[1]]
        tail_exp = self.tail_explanations[hyperpath[-2]]
        Delta_h = head_exp.ret['partial_t_inf'] * tail_exp.ret['delta_2']
        Delta_t = tail_exp.ret['partial_h_inf'] * head_exp.ret['delta_2']
        Delta = head_exp.ret['partial_inf'] * head_exp.ret['delta_2'] * tail_exp.ret['delta_2']

        point = (DEFAULT_VALID_THRESHOLD - head_exp.relevance) / Delta_h
        prob *= 1 - rv_dic['xi_h'].cdf(point)

        point = (DEFAULT_VALID_THRESHOLD - tail_exp.relevance) / Delta_t
        prob *= 1 - rv_dic['xi_t'].cdf(point)

        point = (DEFAULT_VALID_THRESHOLD  - head_exp.relevance - tail_exp.relevance) / Delta
        prob *= 1 - rv_dic['xi'].cdf(point)
        
        # sort hyperpath by probability in a descending order
        self.queue.put((-prob, hyperpath))  
        self.probability_dic[hyperpath] = prob


    def generate(self):
        """
        return one top explanation
        You should examine whether the Generator is empty before calling this function
        """
        neg_prob, hyperpath = self.queue.get()
        head, relation, tail = self.prediction
        head_exp = self.head_explanations[hyperpath[1]]
        tail_exp = self.tail_explanations[hyperpath[-2]]
        # approx_exp = Explanation.build(self.prediction, head_exp.samples_to_remove + tail_exp.samples_to_remove, [head, tail])
        
        all_samples_to_remove = set()
        for i in range(len(hyperpath) - 1):
            a = hyperpath[i]
            b = hyperpath[i+1]
            all_samples_to_remove |= args.available_samples[a] & args.available_samples[b]
        real_exp = Explanation.build(self.prediction, list(all_samples_to_remove), list(hyperpath))

        assert self.path_explanations[hyperpath] == 1
        self.path_explanations[hyperpath] = real_exp
        self.window.append(real_exp.relevance)
        
        # concat the first of head_exp.pt_embedding and the last of tail_exp.pt_embedding (pt_embedding is a tensor)
        approx_embedding = torch.cat([head_exp.pt_embedding[0], tail_exp.pt_embedding[-1]], dim=0)
        approx_score = extract_performances_on_embeddings([head,tail], approx_embedding, self.prediction)
        logger.info(f'approx_score: {approx_score}, base_score: {head_exp.base_score}/{tail_exp.base_score}')
        self.approx_rel_dic[hyperpath] = (head_exp.base_score + tail_exp.base_score)/2 - approx_score

        update_df(self.df, {
            'prediction': self.prediction,
            'super_path': hyperpath,
            'relevance': real_exp.relevance,
            'probability': self.probability_dic[hyperpath],
            'head_rel': head_exp.relevance,
            'tail_rel': tail_exp.relevance,
            'triples': all_samples_to_remove,
            'approx_rel': self.approx_rel_dic[hyperpath],
        }, 'hyperpath.csv')
        
        with open(f'{args.output_folder}/hyperpath/{self.prediction}.json', 'w') as f:
            json.dump(self.json(), f, indent=4)

        return hyperpath, real_exp

    
    def json(self):
        ret = {}
        for hyperpath, exp in self.path_explanations.items():
            if exp == 1:    # wait to be calculated
                continue
            head_exp = self.head_explanations[hyperpath[1]]
            tail_exp = self.tail_explanations[hyperpath[-2]]
            # approx_exp = Explanation.build(self.prediction, head_exp.samples_to_remove + tail_exp.samples_to_remove, [head, tail])
            ret[hyperpath] = {
                'exp': exp.json(),
                'head_exp': head_exp.json(),
                'tail_exp': tail_exp.json(),
                'relevance': exp.relevance,
                'probability': self.probability_dic[hyperpath],
                'approx_rel': self.approx_rel_dic[hyperpath],
            }
        # sort hyperpath by rel in a descending order
        ret = sorted(ret.items(), key=lambda x: x[1]['relevance'], reverse=True)
        return ret