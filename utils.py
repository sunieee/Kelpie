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
args.restrain_dic = config[args.dataset].get('tail_restrain', None)


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
file_handler.setFormatter(CustomFormatter('%(asctime)s:%(levelname)s:%(message)s', '%Y-%m-%d %H:%M:%S.%f'))
# StreamHandler for logging to stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(CustomFormatter('%(asctime)s:%(levelname)s:%(message)s', '%Y-%m-%d %H:%M:%S.%f'))

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

post_train_times = 10

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
    return round(x, 4)

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
        relevance = np.tanh(rank_delta) * 10
    elif args.relevance_method == 'score':
        relevance = np.tanh(score_delta) * 10
    elif args.relevance_method == 'hybrid':
        relevance = np.tanh(rank_delta) * 5 + np.tanh(score_delta) * 5
    return rd(relevance)

def extract_detailed_performances(model: Model, sample: numpy.array):
    model.eval()
    head_id, relation_id, tail_id = sample

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

    return target_entity_score, best_entity_score, target_entity_rank

def extract_samples_with_entity(samples, entity_id):
    return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]


class KelpieExplanation:
    _original_model_results = {}  # map original samples to scores and ranks from the original model
    _base_pt_model_results = {}   # map original samples to scores and ranks from the base post-trained model
    _kelpie_init_tensor_cache = OrderedDict()

    """
        Given a "sample to explain" (that is, a sample that the model currently predicts as true,
        and that we want to be predicted as false);
        and given and a list of training samples containing the entity to convert;
        compute the relevance of the samples in removal, that is, an estimate of the effect they would have
        if removed (all together) from the perspective entity to worsen the prediction of the sample to convert.

        :param sample_to_explain: the sample that we would like the model to predict as "true",
                                    in the form of a tuple (head, relation, tail)
        :param samples_to_remove:   the list of samples containing the perspective entity
                                    that we want to analyze the effect of, if added to the perspective entity
    """
    df = pd.DataFrame(columns=['sample_to_explain', 'identifier', 'samples_to_remove', 'incomplete_path_entities', 'length', 'base_score', 'base_best', 'base_rank', 'pt_score', 'pt_best', 'pt_rank', 'rank_worsening', 'score_worsening', 'relevance'])
    
    def _get_kelpie_init_tensor(self):
        embeddings = []
        for entity in self.trainable_entities:
            if entity not in self._kelpie_init_tensor_cache:
                kelpie_init_tensor_size = model.dimension if not isinstance(model, TuckER) else model.entity_dimension
                self._kelpie_init_tensor_cache[entity] = torch.rand(1, kelpie_init_tensor_size, device='cuda') - 0.5
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
        self.training_samples = np.delete(self.original_train_samples, ids, axis=0)


    def __init__(self, 
                 sample_to_explain: Tuple[Any, Any, Any],
                 samples_to_remove: List[Tuple],
                 incomplete_path_entities: List=[]) -> None:
        logger.info("Create kelpie explanation on sample: %s", dataset.sample_to_fact(sample_to_explain, True))
        print("Removing sample:", [dataset.sample_to_fact(x, True) for x in samples_to_remove])
        # for entity_id, samples in samples_to_remove.items():
        #     print("Entity:", dataset.entity_id_to_name(entity_id), "Samples:", [dataset.sample_to_fact(x, True) for x in samples])

        self.sample_to_explain = sample_to_explain
        self.samples_to_remove = samples_to_remove
        self.incomplete_path_entities = incomplete_path_entities

        self.head = sample_to_explain[0]
        self.trainable_entities = [self.head] + self.incomplete_path_entities
        self.kelpie_init_tensor = self._get_kelpie_init_tensor()
        self.identifier = tuple(list(self.sample_to_explain) + self.incomplete_path_entities)
        self._extract_training_samples()
        
        base_pt_target_entity_score, \
        base_pt_best_entity_score, \
        base_pt_target_entity_rank = self.base_post_training_multiple()

        pt_target_entity_score, \
        pt_best_entity_score, \
        pt_target_entity_rank = self.post_training_multiple(self.training_samples)

        rank_worsening = (pt_target_entity_rank - base_pt_target_entity_rank) / base_pt_target_entity_rank
        score_worsening = base_pt_target_entity_score - pt_target_entity_score
        if model.is_minimizer():
            score_worsening *= -1

        # logger.info(f"Kelpie explanation created. Rank worsening: {rank_worsening}, score worsening: {score_worsening}")

        self.relevance = get_removel_relevance(rank_worsening, score_worsening)
        self.ret = {'sample_to_explain': dataset.sample_to_fact(sample_to_explain, True),
                'identifier': self.identifier,
                'samples_to_remove': [dataset.sample_to_fact(x, True) for x in samples_to_remove],
                'incomplete_path_entities': [dataset.entity_id_2_name[x] for x in incomplete_path_entities],
                'length': len(samples_to_remove),
                'base_score': base_pt_target_entity_score,
                'base_best': base_pt_best_entity_score,
                'base_rank': base_pt_target_entity_rank,
                'pt_score': pt_target_entity_score,
                'pt_best': pt_best_entity_score,
                'pt_rank': pt_target_entity_rank,
                'rank_worsening': rank_worsening,
                'score_worsening': score_worsening,
                'relevance': self.relevance}
        
        self.df.loc[len(self.df)] = self.ret
        self.df.to_csv(os.path.join(args.output_folder, f"output_details.csv"), index=False)
        logger.info(f"Kelpie explanation created. {str(self.ret)}")

    def base_post_training_multiple(self):
        if self.identifier in self._base_pt_model_results:
            return self._base_pt_model_results[self.identifier]

        self._base_pt_model_results[self.identifier] = self.post_training_multiple(self.original_train_samples)
        return self._base_pt_model_results[self.identifier]

    def is_valid(self):
        return self.relevance > 1

    def post_training_multiple(self, training_samples):
        results = []
        logger.info(f'[post_training_multiple] {len(training_samples)} samples, {post_train_times} times x {hyperparameters[RETRAIN_EPOCHS]} epoches')
        for _ in tqdm(range(post_train_times)):
            results.append(self.post_training_save(self.training_samples))
        target_entity_score, \
        best_entity_score, \
        target_entity_rank = zip(*results)
        logger.info(f'score: {mean(target_entity_score)} ± {std(target_entity_score)}, best: {mean(best_entity_score)} ± {std(best_entity_score)}, rank: {mean(target_entity_rank)} ± {std(target_entity_rank)}')
        return mean(target_entity_score), mean(best_entity_score), mean(target_entity_rank)

    def original_results(self) :
        sample = self.sample_to_explain
        if not sample in self._original_model_results:
            target_entity_score, \
            best_entity_score, \
            target_entity_rank = extract_detailed_performances(model, sample)
            self._original_model_results[sample] = (target_entity_score, best_entity_score, target_entity_rank)
        return self._original_model_results[sample]

    
    def post_training_save(self, post_train_samples: numpy.array=[]):
        model = TargetModel(dataset=dataset, hyperparameters=hyperparameters)
        model = model.to('cuda')
        model.load_state_dict(state_dict=args.state_dict)

        # print('origin', extract_detailed_performances(model, self.sample_to_explain))
        for param in model.parameters():
            if param.is_leaf:
                param.requires_grad = False
        
        # frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        # trainable_head_embedding = torch.nn.Parameter(self.kelpie_init_tensor, requires_grad=True)
        # frozen_entity_embeddings[self.head] = trainable_head_embedding
        # print(type(frozen_entity_embeddings))
        # model.entity_embeddings.requires_grad = True
        # model.frozen_indices = [i for i in range(model.entity_embeddings.shape[0]) if i != self.head]
        model.start_post_train(trainable_indices=self.trainable_entities, init_tensor=self.kelpie_init_tensor)

        # print('weight', tensor_head(model.convolutional_layer.weight))
        # print('embedding', tensor_head(model.entity_embeddings[self.head]))
        # print('other embedding', tensor_head(model.entity_embeddings[self.head+1]))

        # Now you can do your training. Only the entity_embeddings for self.head will get updated...
        optimizer = Optimizer(model=model, hyperparameters=hyperparameters, verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]
        # optimizer.learning_rate *= 10

        # print('original, post_train = ', len(original_train_samples), len(post_train_samples))
        optimizer.train(train_samples=post_train_samples, post_train=True)
        ret = extract_detailed_performances(model, self.sample_to_explain)
        # print('pt', ret)

        # print('weight', tensor_head(model.convolutional_layer.weight))
        # print('embedding', tensor_head(model.entity_embeddings[self.head]))
        # print('other embedding', tensor_head(model.entity_embeddings[self.head+1]))
        
        return ret

    

from prefilters.prefilter import PreFilter
import threading
from multiprocessing.pool import ThreadPool as Pool
from config import MAX_PROCESSES
class DividePrefilter(PreFilter):
    """The DividePrefilter divide all relevance into 2 groups and calculate relevance of both. If one group is valid, continue dividing.
    Ending criterion: for some layer, the total count of triples is no greater than M. 
    If the last layer has more than M valid triples, return the top M relevance triple.

    Args:
        PreFilter (_type_): _description_
    """
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
                                  sample_to_explain:Tuple[Any, Any, Any],
                                  incomplete_path_entities: List=[],
                                  top_k=20,) -> list[KelpieExplanation]:
        
        all_paths = self.dataset.find_all_path_within_k_hop(sample_to_explain[0], sample_to_explain[-1])
        all_paths_first_hop = [path[0] for path in all_paths]
        print('all paths first hop', set(all_paths_first_hop), len(all_paths))


        triples = self.dataset.find_all_neighbor_triples(sample_to_explain, incomplete_path_entities)
        print('latent triples', len(triples), triples)

        if len(triples) <= top_k:
            print(f'latent triples {len(triples)} <= {top_k}. no need to divide')
            explanations = []
            for triple in triples:
                explanations.append(KelpieExplanation(sample_to_explain, [triple], incomplete_path_entities))
            
            explanations.sort(key=lambda x: x.relevance, reverse=True)
            return explanations
        
        print('constructing groups...')
        # divide triples into 2 groups of the same length randomly
        groups = [KelpieExplanation(sample_to_explain, triples, incomplete_path_entities)]
        while sum([len(group) for group in groups]) > top_k:
            new_groups = []
            for group in groups:
                new_groups.extend(self.search_valid_groups(group, sample_to_explain, incomplete_path_entities))
            groups = new_groups

        # combine all groups
        explanations = []
        for group in groups:
            if len(group.samples_to_remove) == 1:
                explanations.append(group)
            else:
                for sample in group.samples_to_remove:
                    explanations.append(KelpieExplanation(sample_to_explain, [sample], incomplete_path_entities))

        explanations.sort(lambda x: x.relevance, reverse=True)
        return explanations
        

    def search_valid_groups(self, exp: KelpieExplanation):
        # divide triples into 2 groups of the same length randomly
        triples = exp.samples_to_remove
        random.shuffle(triples)
        group1 = triples[:len(triples)//2]
        group2 = triples[len(triples)//2:]
        print('divided into 2 groups')
        print('group1', len(group1), group1)
        print('group2', len(group2), group2)

        group1_explanation = KelpieExplanation(exp.sample_to_explain, group1, exp.incomplete_path_entities)
        group2_explanation = KelpieExplanation(exp.sample_to_explain, group2, exp.incomplete_path_entities)

        valid_groups = []
        if group1_explanation.is_valid():
            valid_groups.append(group1_explanation)
        if group2_explanation.is_valid():
            valid_groups.append(group2_explanation)

        return valid_groups
        

class Path:
    def __init__(self, triples, explanations) -> None:
        self.triples = triples
        self.explanations = explanations


class Xrule:
    DEFAULT_MAX_LENGTH = 4

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 max_explanation_length: int = DEFAULT_MAX_LENGTH) -> None:
        
        self.model = model
        self.dataset = dataset
        self.prefilter = DividePrefilter(model, dataset)

    def explain_necessary(self,
                          sample_to_explain: Tuple[Any, Any, Any]):
        """This method extracts necessary explanations for a specific sample

        Args:
            sample_to_explain (Tuple[Any, Any, Any]): _description_
        """
        h, r, t = sample_to_explain
        return self.get_k_hop_path(sample_to_explain)
    
    def get_k_hop_path(self, sample_to_explain, incomplete_path_entities=[], last_sample=[], last_exp=[]):
        """This method extracts necessary paths for a specific sample, k <= hops <= MAX_PATH_LENGTH
        k = len(incomplete_path_entities) + 1
        len(incomplete_path_entities)=0 => one_hop_path
        len(incomplete_path_entities)=1 => two_hop_path
        len(incomplete_path_entities)=2 => three_hop_path

        Args:
            sample_to_explain (_type_): _description_
            incomplete_path_entities (list, optional): _description_. Defaults to [].
        """
        k = len(incomplete_path_entities) + 1
        prefix = '!'*k
        logger.info(f'{prefix} explanaing {k}-hop path on {sample_to_explain}: {self.dataset.sample_to_fact(sample_to_explain, True)}')
        logger.info(f'{prefix} incomplete_path_entities: {incomplete_path_entities}, last_sample: {last_sample}')


        h, r, t = sample_to_explain
        trainable_entities = [h] + incomplete_path_entities
        all_paths = []
        k_hop_explanations = self.prefilter.top_promising_explanations(sample_to_explain, incomplete_path_entities)

        for k_hop_exp in k_hop_explanations:
            assert len(k_hop_exp.samples_to_remove) == 1
            k_hop_sample = k_hop_exp.samples_to_remove[-1]
            k_hop_target = k_hop_sample[2] if k_hop_sample[0] == trainable_entities[-1] else k_hop_sample[0]
            if k_hop_target == t:
                all_paths.append(Path(last_sample + [k_hop_sample], last_exp + [k_hop_exp]))
                continue

            all_paths.extend(self.get_k_hop_path(k_hop_sample, 
                                                 incomplete_path_entities + [k_hop_target], 
                                                 last_sample + [k_hop_sample], 
                                                 last_exp + [k_hop_exp]))
            
        return all_paths