import sys
import os
import argparse
import copy
import random
import numpy
import torch

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from dataset import ALL_DATASET_NAMES, Dataset, MANY_TO_ONE, ONE_TO_ONE
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, \
    REGULARIZER_WEIGHT, EPOCHS, BATCH_SIZE, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, \
    HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY, MARGIN, NEGATIVE_SAMPLES_RATIO, REGULARIZER_NAME

# Define available models
MODEL_CHOICES = ['complex', 'conve', 'transe']

# Add optimizer choices
OPTIMIZER_CHOICES = ['Adagrad', 'Adam', 'SGD']

parser = argparse.ArgumentParser(description="Model-agnostic tool for verifying link predictions explanations")

parser.add_argument('--model', choices=MODEL_CHOICES, required=True, help="Model to use: complex, conve, transe")
parser.add_argument('--dataset', choices=ALL_DATASET_NAMES, required=True, help="Dataset to use")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model to explain the predictions of")
parser.add_argument('--optimizer', choices=OPTIMIZER_CHOICES, default='Adagrad', help="Optimizer to use")
parser.add_argument('--mode', type=str, default="necessary", choices=["sufficient", "necessary"], help="The explanation mode (default is necessary)")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--max_epochs', type=int, default=1000, help="Number of epochs")
parser.add_argument('--dimension', type=int, default=200, help="Factorization rank or embedding dimension")
parser.add_argument('--learning_rate', type=float, default=0.0005, help="Learning rate")
parser.add_argument('--reg', type=float, default=0, help="Regularization weight")
parser.add_argument('--init', type=float, default=1e-3, help="Initial scale for complex")
parser.add_argument('--decay1', type=float, default=0.9, help="Decay rate for the first moment estimate in Adam")
parser.add_argument('--decay2', type=float, default=0.999, help="Decay rate for second moment estimate in Adam")
parser.add_argument('--regularizer_name', type=str, default='N3', help="Regularizer name")
parser.add_argument('--margin', type=int, default=5, help="Margin for pairwise ranking loss (TransE)")
parser.add_argument('--negative_samples_ratio', type=int, default=3, help="Negative samples ratio (TransE)")
parser.add_argument('--input_dropout', type=float, default=0.3, help="Input layer dropout (ConvE)")
parser.add_argument('--hidden_dropout', type=float, default=0.4, help="Hidden layer dropout (ConvE)")
parser.add_argument('--feature_map_dropout', type=float, default=0.5, help="Feature map dropout (ConvE)")
parser.add_argument('--hidden_size', type=int, default=9728, help="Hidden layer size (ConvE)")
parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing (ConvE)")

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

# Read explanations from the output file
with open("output.txt", "r") as input_file:
    input_lines = input_file.readlines()

# Define model-specific parameters
if args.model == 'complex':
    hyperparameters = {DIMENSION: args.dimension, INIT_SCALE: args.init, LEARNING_RATE: args.learning_rate,
                       OPTIMIZER_NAME: args.optimizer, DECAY_1: args.decay1, DECAY_2: args.decay2, 
                       REGULARIZER_WEIGHT: args.reg, EPOCHS: args.max_epochs, BATCH_SIZE: args.batch_size,
                       REGULARIZER_NAME: args.regularizer_name}
    model_class = ComplEx
    optimizer_class = MultiClassNLLOptimizer

elif args.model == 'conve':
    hyperparameters = {DIMENSION: args.dimension, INPUT_DROPOUT: args.input_dropout, FEATURE_MAP_DROPOUT: args.feature_map_dropout,
                       HIDDEN_DROPOUT: args.hidden_dropout, HIDDEN_LAYER_SIZE: args.hidden_size, BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate, DECAY: args.decay1, LABEL_SMOOTHING: args.label_smoothing, 
                       EPOCHS: args.max_epochs, OPTIMIZER_NAME: args.optimizer, REGULARIZER_NAME: args.regularizer_name}
    model_class = ConvE
    optimizer_class = BCEOptimizer

elif args.model == 'transe':
    hyperparameters = {DIMENSION: args.dimension, MARGIN: args.margin, NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                       REGULARIZER_WEIGHT: args.reg, BATCH_SIZE: args.batch_size, LEARNING_RATE: args.learning_rate, 
                       EPOCHS: args.max_epochs, OPTIMIZER_NAME: args.optimizer, REGULARIZER_NAME: args.regularizer_name}
    model_class = TransE
    optimizer_class = PairwiseRankingOptimizer

# Initialize and load the model
model = model_class(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
model.to('cuda')
model.load_state_dict(torch.load(args.model_path))
model.eval()

facts_to_explain = []
samples_to_explain = []
perspective = "head"
sample_to_explain_2_best_rule = {}

# Explanation verification based on mode
if args.mode != "necessary":
    raise NotImplementedError("Only necessary mode is supported")
# Process necessary mode
i = 0
while i <= len(input_lines) - 3:
    fact_line = input_lines[i]
    rules_line = input_lines[i + 1]
    empty_line = input_lines[i + 2]

    fact = tuple(fact_line.strip().strip(';').split(";"))
    facts_to_explain.append(fact)
    sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
    samples_to_explain.append(sample)

    rules_with_relevance = []

    rule_relevance_inputs = rules_line.strip().split(",")
    best_rule, best_rule_relevance_str = rule_relevance_inputs[0].split(":")
    best_rule_bits = best_rule.split(";")

    best_rule_facts = []
    j = 0
    while j < len(best_rule_bits):
        cur_head_name = best_rule_bits[j]
        cur_rel_name = best_rule_bits[j + 1]
        cur_tail_name = best_rule_bits[j + 2]
        best_rule_facts.append((cur_head_name, cur_rel_name, cur_tail_name))
        j += 3

    best_rule_samples = [dataset.fact_to_sample(x) for x in best_rule_facts]
    relevance = float(best_rule_relevance_str)
    rules_with_relevance.append((best_rule_samples, relevance))

    sample_to_explain_2_best_rule[sample] = best_rule_samples
    i += 3

samples_to_remove = []

for sample_to_explain in samples_to_explain:
    best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]
    samples_to_remove += best_rule_samples

new_dataset = copy.deepcopy(dataset)

print("Removing samples: ")
for (head, relation, tail) in samples_to_remove:
    print("\t" + dataset.printable_sample((head, relation, tail)))

new_dataset.remove_training_samples(numpy.array(samples_to_remove))

original_scores, original_ranks, original_predictions = model.predict_samples(numpy.array(samples_to_explain))

new_model = model_class(dataset=new_dataset, hyperparameters=hyperparameters, init_random=True)
new_optimizer = optimizer_class(model=new_model, hyperparameters=hyperparameters)
new_optimizer.train(train_samples=new_dataset.train_samples)
new_model.eval()

new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_explain))

output_lines = []
for i in range(len(samples_to_explain)):
    cur_sample_to_explain = samples_to_explain[i]
    original_direct_score = original_scores[i][0]
    original_tail_rank = original_ranks[i][1]
    new_direct_score = new_scores[i][0]
    new_tail_rank = new_ranks[i][1]

    a = ";".join(dataset.sample_to_fact(cur_sample_to_explain))
    b = []
    samples_to_remove_from_this_entity = sample_to_explain_2_best_rule[cur_sample_to_explain]
    for x in range(4):
        if x < len(samples_to_remove_from_this_entity):
            b.append(";".join(dataset.sample_to_fact(samples_to_remove_from_this_entity[x])))
        else:
            b.append(";;")

    b = ";".join(b)
    c = str(original_direct_score) + ";" + str(new_direct_score)
    d = str(original_tail_rank) + ";" + str(new_tail_rank)
    output_lines.append(";".join([a, b, c, d]) + "\n")

with open("output_end_to_end.csv", "w") as outfile:
    outfile.writelines(output_lines)
