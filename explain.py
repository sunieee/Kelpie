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
import logging
from datetime import datetime

from link_prediction.models.transe import TransE
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.gcn import CompGCN
from link_prediction.models.model import *
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from link_prediction.evaluation.evaluation import Evaluator
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER

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
    for d in os.listdir(args.output_folder):
        if os.path.isdir(os.path.join(args.output_folder, d)) and d in ['haslinker', 'hasmetal', 'hassolvent']:
            df = pd.read_csv(os.path.join(args.output_folder, d, 'filtered_ranks.csv'), sep=';', header=None)
            df.columns = ['h', 'r', 't', 'hr', 'tr']
            size = len(dataset.rid2target[dataset.relation_name_2_id[d]])
            
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

ech(f"Reading facts to explain... from {args.explain_path}")
with open(args.explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

print("len(testing_facts):", len(testing_facts))

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


output_lines = []
def triple2str(triple):
    return '<' +','.join(triple) + '>'

def print_line(line):
    print(line)
    output_lines.append(line)


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

        if cur_relevance > 1:
            for i in range(10):
                retrain_without_samples(remove_triples=cur_rule_samples, sample_to_explain=sample_to_explain)

for i, fact in enumerate(testing_facts):
    head, relation, tail = fact
    print("Explaining fact " + str(i) + " on " + str(
        len(testing_facts)) + ": " + triple2str(fact))
    head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                    dataset.get_id_for_relation_name(relation), \
                                    dataset.get_id_for_entity_name(tail)
    sample_to_explain = (head_id, relation_id, tail_id)
    rule_samples_with_relevance = kelpie.explain_necessary(sample_to_explain=sample_to_explain,
                                                            perspective="head",
                                                            num_promising_samples=args.prefilter_threshold)
    print_line(f'output of fact {triple2str(fact)}')
    print_facts(rule_samples_with_relevance, sample_to_explain)

ech('explaination output:')
end_time = time.time()
print("Explain time: " + str(end_time - start_time) + " seconds")
with open(os.path.join(args.output_folder, f"{args.mode}.txt"), "w") as output:
    output.writelines(output_lines)

print('count_dic', count_dic)
print('count_dic_mean', {k: np.mean(v) for k, v in count_dic.items()})