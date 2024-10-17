import sys
import os
import argparse
import random
import time
import numpy
import torch

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from dataset import ALL_DATASET_NAMES, Dataset
from kelpie import Kelpie
from k1_abstract import K1_asbtract
from data_poisoning import DataPoisoning
from criage import Criage
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, \
    REGULARIZER_WEIGHT, EPOCHS, BATCH_SIZE, REGULARIZER_NAME, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, \
    HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY, MARGIN, NEGATIVE_SAMPLES_RATIO
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER

# Define model choices
MODEL_CHOICES = ['complex', 'conve', 'transe']

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")
parser.add_argument('--model', choices=MODEL_CHOICES, required=True, help="Model to use: complex, conve, transe")
parser.add_argument('--dataset', choices=ALL_DATASET_NAMES, required=True, help="Dataset to use")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model to explain the predictions of")
parser.add_argument('--facts_to_explain_path', type=str, required=True, help="Path of the file with the facts to explain")
parser.add_argument('--optimizer', choices=['Adagrad', 'Adam', 'SGD'], default='Adagrad', help="Optimizer to use")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--max_epochs', type=int, default=200, help="Number of epochs")
parser.add_argument('--dimension', type=int, default=200, help="Factorization rank or embedding dimension")
parser.add_argument('--learning_rate', type=float, default=1e-1, help="Learning rate")
parser.add_argument('--reg', type=float, default=0, help="Regularization weight")
parser.add_argument('--init', type=float, default=1e-3, help="Initial scale for complex")
parser.add_argument('--decay1', type=float, default=0.9, help="Decay rate for the first moment estimate in Adam")
parser.add_argument('--decay2', type=float, default=0.999, help="Decay rate for second moment estimate in Adam")
parser.add_argument('--margin', type=int, default=5, help="Margin for pairwise ranking loss (TransE)")
parser.add_argument('--negative_samples_ratio', type=int, default=3, help="Negative samples ratio (TransE)")
parser.add_argument('--input_dropout', type=float, default=0.3, help="Input layer dropout (ConvE)")
parser.add_argument('--hidden_dropout', type=float, default=0.4, help="Hidden layer dropout (ConvE)")
parser.add_argument('--feature_map_dropout', type=float, default=0.5, help="Feature map dropout (ConvE)")
parser.add_argument('--hidden_size', type=int, default=9728, help="Hidden layer size (ConvE)")
parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing (ConvE)")
parser.add_argument('--coverage', type=int, default=10, help="Number of random entities to extract and convert")
parser.add_argument('--baseline', type=str, default=None, help="Baseline engine to use")
parser.add_argument('--entities_to_convert', type=str, help="Path of the file with the entities to convert (baselines)")
parser.add_argument('--relevance_threshold', type=float, default=None, help="Relevance acceptance threshold")
parser.add_argument('--prefilter', choices=[TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER], default=NO_PREFILTER, help="Prefilter type")
parser.add_argument('--prefilter_threshold', type=int, default=20, help="Number of promising training facts to keep after prefiltering")
parser.add_argument('--relation', type=str, help="Relation to explain")

args = parser.parse_args()

# Set random seed for reproducibility
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

# Load dataset
print(f"Loading dataset {args.dataset}...")
dataset = Dataset(name=args.dataset, separator="\t", load=True)

# Read facts to explain
print("Reading facts to explain...")
with open(args.facts_to_explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

if args.relation is not None:
    print('total facts:', len(testing_facts))
    testing_facts = [fact for fact in testing_facts if fact[1] == args.relation]
    print('facts with relation:', len(testing_facts))

# Select and initialize the model
if args.model == 'complex':
    hyperparameters = {DIMENSION: args.dimension, INIT_SCALE: args.init, LEARNING_RATE: args.learning_rate,
                       OPTIMIZER_NAME: args.optimizer, DECAY_1: args.decay1, DECAY_2: args.decay2,
                       REGULARIZER_WEIGHT: args.reg, EPOCHS: args.max_epochs, BATCH_SIZE: args.batch_size,
                       REGULARIZER_NAME: "N3"}
    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)

elif args.model == 'conve':
    hyperparameters = {DIMENSION: args.dimension, INPUT_DROPOUT: args.input_dropout, FEATURE_MAP_DROPOUT: args.feature_map_dropout,
                       HIDDEN_DROPOUT: args.hidden_dropout, HIDDEN_LAYER_SIZE: args.hidden_size, BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate, DECAY: args.decay1, LABEL_SMOOTHING: args.label_smoothing, EPOCHS: args.max_epochs}
    model = ConvE(dataset=dataset, hyperparameters=hyperparameters, init_random=True)

elif args.model == 'transe':
    hyperparameters = {DIMENSION: args.dimension, MARGIN: args.margin, NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                       REGULARIZER_WEIGHT: args.reg, BATCH_SIZE: args.batch_size, LEARNING_RATE: args.learning_rate, EPOCHS: args.max_epochs}
    model = TransE(dataset=dataset, hyperparameters=hyperparameters, init_random=True)

model.to('cuda')
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Initialize Kelpie engine or other baseline
if args.baseline is None:
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=args.prefilter, relevance_threshold=args.relevance_threshold)
elif args.baseline == "data_poisoning":
    kelpie = DataPoisoning(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=args.prefilter)
elif args.baseline == "criage":
    kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters)
elif args.baseline == "k1":
    kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=args.prefilter,
                    relevance_threshold=args.relevance_threshold, max_explanation_length=1)
elif args.baseline == "k1_abstract":
    kelpie = K1_asbtract(model=model, dataset=dataset, hyperparameters=hyperparameters, relevance_threshold=args.relevance_threshold, max_explanation_length=1)

# Handle necessary explanations only
start_time = time.time()
output = open("output.txt", "w")

total_count = 10    # len(testing_facts)
for i, fact in enumerate(testing_facts[:total_count]):
    head, relation, tail = fact
    print(f"Explaining fact {i + 1}/{total_count}: <{head}, {relation}, {tail}>")
    head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), dataset.get_id_for_relation_name(relation), dataset.get_id_for_entity_name(tail)
    sample_to_explain = (head_id, relation_id, tail_id)

    # Necessary explanations
    rule_samples_with_relevance = kelpie.explain_necessary(sample_to_explain=sample_to_explain, perspective="head", num_promising_samples=args.prefilter_threshold)

    if args.baseline != "k1_abstract":
        # Collect and print the results
        rule_facts_with_relevance = []
        for cur_rule_with_relevance in rule_samples_with_relevance:
            cur_rule_samples, cur_relevance = cur_rule_with_relevance
            cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
            cur_rule_facts = ";".join([";".join(x) for x in cur_rule_facts])
            rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))

        output.write(f";{head};{relation};{tail}\n")
        output.write(",".join(rule_facts_with_relevance) + "\n\n")

end_time = time.time()
print(f"Required time: {end_time - start_time:.2f} seconds")
output.close()
