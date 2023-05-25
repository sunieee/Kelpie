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
                    choices=['rank', 'score', 'kelpie'], help="the method to compute relevance")

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
state_dict = torch.load(f'{args.output_folder}/params.pth')

post_train_times = 5

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

def std(lis):
    return rd(np.std(lis))

def get_removel_relevance(rank_delta, score_delta):
    if args.relevance_method == 'kelpie':
        relevance = float(rank_delta + sigmoid(score_delta))
    elif args.relevance_method == 'rank':
        relevance = float(rank_delta)
    elif args.relevance_method == 'score':
        relevance = float(score_delta * 10)
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

    # The kelpie_cache is a simple LRU cache that allows reuse of KelpieDatasets and of base post-training results
    # without need to re-build them from scratch every time.
    _kelpie_dataset_cache_size = 20
    _kelpie_dataset_cache = OrderedDict()

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
    df = pd.DataFrame(columns=['sample_to_explain', 'samples_to_remove', 'length', 'base_score', 'base_best', 'base_rank', 'pt_score', 'pt_best', 'pt_rank', 'rank_worsening', 'score_worsening', 'relevance'])

    def __init__(self, 
                 sample_to_explain: Tuple[Any, Any, Any],
                 samples_to_remove: list) -> None:
        logger.info("Create kelpie explanation on sample: %s", dataset.sample_to_fact(sample_to_explain, True))
        print("Removing sample:", [dataset.sample_to_fact(x, True) for x in samples_to_remove])
        self.sample_to_explain = sample_to_explain
        self.samples_to_remove = samples_to_remove
        
        self.head = sample_to_explain[0]
        self.kelpie_dataset = self._get_kelpie_dataset_for(original_entity_id=self.head)
        self.kelpie_sample_to_predict = self.kelpie_dataset.as_kelpie_sample(original_sample=self.sample_to_explain)

        kelpie_init_tensor_size = model.dimension if not isinstance(model, TuckER) else model.entity_dimension
        self.kelpie_init_tensor = torch.rand(1, kelpie_init_tensor_size)

        base_pt_target_entity_score, \
        base_pt_best_entity_score, \
        base_pt_target_entity_rank = self.post_training_results_multiple()

        pt_target_entity_score, \
        pt_best_entity_score, \
        pt_target_entity_rank = self.post_training_results_multiple(self.samples_to_remove)

        rank_worsening = pt_target_entity_rank - base_pt_target_entity_rank
        score_worsening = base_pt_target_entity_score - pt_target_entity_score
        if model.is_minimizer():
            score_worsening *= -1

        # logger.info(f"Kelpie explanation created. Rank worsening: {rank_worsening}, score worsening: {score_worsening}")

        self.relevance = get_removel_relevance(rank_worsening, score_worsening)
        self.ret = {'sample_to_explain': dataset.sample_to_fact(sample_to_explain, True),
                'samples_to_remove': [dataset.sample_to_fact(x, True) for x in samples_to_remove],
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


    def post_training_results_multiple(self, samples_to_remove: list = []):
        if len(samples_to_remove) == 0 and self.sample_to_explain in self._base_pt_model_results:
            return self._base_pt_model_results[self.sample_to_explain]
        results = []
        logger.info(f'[post_training_results_multiple] {len(self.kelpie_dataset.kelpie_train_samples)} - {len(samples_to_remove)}, {post_train_times} times x {hyperparameters[RETRAIN_EPOCHS]} epoches')
        for _ in tqdm(range(post_train_times)):
            # results.append(self.post_training_results(samples_to_remove))
            results.append(self.post_training_save(samples_to_remove))
        target_entity_score, \
        best_entity_score, \
        target_entity_rank = zip(*results)
        if len(samples_to_remove) == 0:
            self._base_pt_model_results[self.sample_to_explain] = mean(target_entity_score), mean(best_entity_score), mean(target_entity_rank)
        logger.info(f'score: {mean(target_entity_score)} ± {std(target_entity_score)}, best: {mean(best_entity_score)} ± {std(best_entity_score)}, rank: {mean(target_entity_rank)} ± {std(target_entity_rank)}')
        return mean(target_entity_score), mean(best_entity_score), mean(target_entity_rank)

    def _get_kelpie_dataset_for(self, original_entity_id: int) -> KelpieDataset:
        """
        Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """

        if original_entity_id not in self._kelpie_dataset_cache:
            kelpie_dataset = KelpieDataset(dataset=dataset, entity_id=original_entity_id)
            self._kelpie_dataset_cache[original_entity_id] = kelpie_dataset
            self._kelpie_dataset_cache.move_to_end(original_entity_id)

            if len(self._kelpie_dataset_cache) > self._kelpie_dataset_cache_size:
                self._kelpie_dataset_cache.popitem(last=False)

        return self._kelpie_dataset_cache[original_entity_id]

    def original_results(self) :
        sample = self.sample_to_explain
        if not sample in self._original_model_results:
            target_entity_score, \
            best_entity_score, \
            target_entity_rank = extract_detailed_performances(model, sample)
            self._original_model_results[sample] = (target_entity_score, best_entity_score, target_entity_rank)
        return self._original_model_results[sample]
    
    def post_training_clone(self, samples_to_remove: numpy.array=[]):
        print('origin', extract_detailed_performances(model, self.sample_to_explain))
        
        post_model = PostConvE(model, self.head, self.kelpie_init_tensor)
        post_model = post_model.to('cuda')
        # Now you can do your training. Only the entity_embeddings for self.head will get updated...
        # optimizer = kelpie_optimizer_class(model=post_model,
        #                                     hyperparameters=hyperparameters,
        #                                     verbose=False)
        optimizer = Optimizer(model=post_model, hyperparameters=hyperparameters, verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]

        original_train_samples = extract_samples_with_entity(dataset.train_samples, self.head)
        ids = []
        for sample in samples_to_remove:
            ids.append(original_train_samples.tolist().index(list(sample)))
        post_train_samples = np.delete(original_train_samples, ids, axis=0)

        print('original, post_train = ', len(original_train_samples), len(post_train_samples))
        # post_train_samples = torch.tensor(post_train_samples).to('cuda')
        optimizer.train(train_samples=post_train_samples)
        ret = extract_detailed_performances(post_model, self.sample_to_explain)
        print('pt', ret)

        return ret
    
    def post_training_directly(self, samples_to_remove: numpy.array=[]):
        print('origin', extract_detailed_performances(model, self.sample_to_explain))

        # for param in model.parameters():
        #     param.requires_grad = False
        
        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        entity_embeddings = frozen_entity_embeddings.clone()
        trainable_head_embedding = torch.nn.Parameter(self.kelpie_init_tensor, requires_grad=True)
        entity_embeddings[self.head] = trainable_head_embedding
        model.entity_embeddings = torch.nn.Parameter(entity_embeddings)

        # Now you can do your training. Only the entity_embeddings for self.head will get updated...
        optimizer = Optimizer(model=model, hyperparameters=hyperparameters, verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]

        original_train_samples = extract_samples_with_entity(dataset.train_samples, self.head)
        ids = []
        for sample in samples_to_remove:
            ids.append(original_train_samples.tolist().index(list(sample)))
        post_train_samples = np.delete(original_train_samples, ids, axis=0)

        print('original, post_train = ', len(original_train_samples), len(post_train_samples))
        optimizer.train(train_samples=post_train_samples)
        ret = extract_detailed_performances(model, self.sample_to_explain)
        print('pt', ret)

        model.entity_embeddings = torch.nn.Parameter(frozen_entity_embeddings)

        return ret
    
    def post_training_save(self, samples_to_remove: numpy.array=[]):
        print('origin', extract_detailed_performances(model, self.sample_to_explain))
        for param in model.parameters():
            param.requires_grad = False

        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        trainable_head_embedding = torch.nn.Parameter(self.kelpie_init_tensor, requires_grad=True)
        frozen_entity_embeddings[self.head] = trainable_head_embedding
        model.entity_embeddings = torch.nn.Parameter(frozen_entity_embeddings)

        # Now you can do your training. Only the entity_embeddings for self.head will get updated...
        optimizer = Optimizer(model=model, hyperparameters=hyperparameters, verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]

        original_train_samples = extract_samples_with_entity(dataset.train_samples, self.head)
        ids = []
        for sample in samples_to_remove:
            ids.append(original_train_samples.tolist().index(list(sample)))
        post_train_samples = np.delete(original_train_samples, ids, axis=0)

        print('original, post_train = ', len(original_train_samples), len(post_train_samples))
        optimizer.train(train_samples=post_train_samples)
        ret = extract_detailed_performances(model, self.sample_to_explain)
        print('pt', ret)

        model.load_state_dict(state_dict)

        return ret


    def post_training_results(self, samples_to_remove: numpy.array=[]):
        print('origin', extract_detailed_performances(model,self.sample_to_explain))
        kelpie_model = kelpie_model_class(model=model,
                                        dataset=self.kelpie_dataset,
                                        init_tensor=self.kelpie_init_tensor)
        print('base origin', extract_detailed_performances(kelpie_model,self.sample_to_explain))
        print('base kelpie', extract_detailed_performances(kelpie_model,self.kelpie_sample_to_predict))
        if len(samples_to_remove):
            self.kelpie_dataset.remove_training_samples(samples_to_remove)
        base_pt_model = self.post_train(kelpie_model_to_post_train=kelpie_model,
                                        kelpie_train_samples=self.kelpie_dataset.kelpie_train_samples) # type: KelpieModel
        print('pt origin', extract_detailed_performances(kelpie_model,self.sample_to_explain))
        print('pt kelpie', extract_detailed_performances(kelpie_model,self.kelpie_sample_to_predict))
        if len(samples_to_remove):
            self.kelpie_dataset.undo_last_training_samples_removal()
        return extract_detailed_performances(base_pt_model, self.kelpie_sample_to_predict)

    def post_train(self,
                   kelpie_model_to_post_train: KelpieModel,
                   kelpie_train_samples: numpy.array):
        """
        :param kelpie_model_to_post_train: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_train_samples:
        :return:
        """
        kelpie_model_to_post_train.to('cuda')
        optimizer = kelpie_optimizer_class(model=kelpie_model_to_post_train,
                                            hyperparameters=hyperparameters,
                                            verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]
        # print(optimizer.epochs)
        t = time.time()
        optimizer.train(train_samples=kelpie_train_samples)
        # print(f'[post_train] kelpie_train_samples: {len(kelpie_train_samples)}, epoches: {optimizer.epochs}, time: {rd(time.time() - t)}')
        return kelpie_model_to_post_train