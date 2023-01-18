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
    click.echo(click.style(s, fg))

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

parser.add_argument("--mode",
                    type=str,
                    default="sufficient",
                    choices=["sufficient", "necessary"],
                    help="The explanation mode")

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

# parser.add_argument('--sort', dest='sort', default=False, action='store_true',
#                     help="whether sort the dataset")

args = parser.parse_args()
cfg = config[args.dataset][args.method]
tail_restrain = config[args.dataset].get('tail_restrain', None)

# deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

# load the dataset and its training samples
ech(f"Loading dataset {args.dataset}...")
dataset = Dataset(name=args.dataset, separator="\t", load=True, tail_restrain=tail_restrain, args=args)
try:
    tail_restrain = dataset.tail_restrain
except:
    tail_restrain = None
args.tail_restrain = tail_restrain

ech("Initializing LP model...")
if args.method == "ConvE":
    hyperparameters = {DIMENSION: cfg['D'],
                    INPUT_DROPOUT: cfg['Drop']['in'],
                    FEATURE_MAP_DROPOUT: cfg['Drop']['feat'],
                    HIDDEN_DROPOUT: cfg['Drop']['h'],
                    HIDDEN_LAYER_SIZE: 9728,
                    BATCH_SIZE: cfg['B'],
                    LEARNING_RATE: cfg['LR'],
                    DECAY: cfg['Decay'],
                    LABEL_SMOOTHING: 0.1,
                    EPOCHS: cfg['Ep']}
    TargetModel = ConvE
    Optimizer = BCEOptimizer
elif args.method == "ComplEx":
    hyperparameters = {DIMENSION: cfg['D'],
                   INIT_SCALE: 1e-3,
                   LEARNING_RATE: cfg['LR'],
                   OPTIMIZER_NAME: 'Adagrad',  # 'Adagrad', 'Adam', 'SGD'
                   DECAY_1: 0.9,
                   DECAY_2: 0.999,
                   REGULARIZER_WEIGHT: cfg['Reg'],
                   EPOCHS: cfg['Ep'],
                   BATCH_SIZE: cfg['B'],
                   REGULARIZER_NAME: "N3"}
    TargetModel = ComplEx
    Optimizer = MultiClassNLLOptimizer
elif args.method == "TransE":
    hyperparameters = {DIMENSION: cfg['D'],
                   MARGIN: 5,
                   NEGATIVE_SAMPLES_RATIO: cfg['N'],
                   REGULARIZER_WEIGHT: cfg['Reg'],
                   BATCH_SIZE: cfg['B'],
                   LEARNING_RATE: cfg['LR'],
                   EPOCHS: cfg['Ep']}
    TargetModel = TransE
    Optimizer = PairwiseRankingOptimizer

if args.embedding_model and args.embedding_model != 'none':
    cf = config[args.dataset][args.embedding_model]
    print(cf)
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

model = TargetModel(dataset=dataset, hyperparameters=hyperparameters, \
                init_random=True, args=args)
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
    optimizer = Optimizer(model=model, hyperparameters=hyperparameters, args=args)
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
    for d in os.listdir(args.output_folder):
        if os.path.isdir(d):
            with open(os.path.join(args.output_folder, d, 'filtered_ranks.csv'), 'r') as f:
                pass


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

testing_fact_2_entities_to_convert = None
if args.mode == "sufficient" and args.entities_to_convert is not None:
    print("Reading entities to convert...")
    testing_fact_2_entities_to_convert = {}
    with open(args.entities_to_convert, "r") as entities_to_convert_file:
        entities_to_convert_lines = entities_to_convert_file.readlines()
        i = 0
        while i < len(entities_to_convert_lines):
            cur_head, cur_rel, cur_name = entities_to_convert_lines[i].strip().split(";")
            assert [cur_head, cur_rel, cur_name] in testing_facts
            cur_entities_to_convert = entities_to_convert_lines[i + 1].strip().split(",")
            testing_fact_2_entities_to_convert[(cur_head, cur_rel, cur_name)] = cur_entities_to_convert
            i += 3

output_lines = []
for i, fact in enumerate(testing_facts):
    head, relation, tail = fact
    print("Explaining fact " + str(i) + " on " + str(
        len(testing_facts)) + ": <" + head + "," + relation + "," + tail + ">")
    head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                    dataset.get_id_for_relation_name(relation), \
                                    dataset.get_id_for_entity_name(tail)
    sample_to_explain = (head_id, relation_id, tail_id)

    if args.mode == "sufficient":
        entities_to_convert_ids = None if testing_fact_2_entities_to_convert is None \
            else [dataset.entity_name_2_id[x] for x in testing_fact_2_entities_to_convert[(head, relation, tail)]]

        rule_samples_with_relevance, \
        entities_to_convert_ids = kelpie.explain_sufficient(sample_to_explain=sample_to_explain,
                                                            perspective="head",
                                                            num_promising_samples=args.prefilter_threshold,
                                                            num_entities_to_convert=args.coverage,
                                                            entities_to_convert=entities_to_convert_ids)

        if entities_to_convert_ids is None or len(entities_to_convert_ids) == 0:
            continue
        entities_to_convert = [dataset.entity_id_2_name[x] for x in entities_to_convert_ids]

        rule_facts_with_relevance = []
        for cur_rule_with_relevance in rule_samples_with_relevance:
            cur_rule_samples, cur_relevance = cur_rule_with_relevance

            cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
            cur_rule_facts = ";".join([";".join(x) for x in cur_rule_facts])
            rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))

        print(";".join(fact))
        print(", ".join(entities_to_convert))
        print(", ".join(rule_facts_with_relevance))
        print()
        output_lines.append(";".join(fact) + "\n")
        output_lines.append(",".join(entities_to_convert) + "\n")
        output_lines.append(",".join(rule_facts_with_relevance) + "\n")
        output_lines.append("\n")

    elif args.mode == "necessary":
        rule_samples_with_relevance = kelpie.explain_necessary(sample_to_explain=sample_to_explain,
                                                               perspective="head",
                                                               num_promising_samples=args.prefilter_threshold)
        rule_facts_with_relevance = []
        for cur_rule_with_relevance in rule_samples_with_relevance:
            cur_rule_samples, cur_relevance = cur_rule_with_relevance

            cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
            cur_rule_facts = ";".join([";".join(x) for x in cur_rule_facts])
            rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))
        print(";".join(fact))
        print(", ".join(rule_facts_with_relevance))
        print()
        output_lines.append(";".join(fact) + "\n")
        output_lines.append(",".join(rule_facts_with_relevance) + "\n")
        output_lines.append("\n")

end_time = time.time()
print("Explain time: " + str(end_time - start_time) + " seconds")
with open("output.txt", "w") as output:
    output.writelines(output_lines)
