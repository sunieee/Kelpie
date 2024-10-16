import sys
import os
import argparse
import copy
import random
import numpy
import torch

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from dataset import ALL_DATASET_NAMES, Dataset
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, \
    REGULARIZER_WEIGHT, EPOCHS, BATCH_SIZE, REGULARIZER_NAME, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, \
    HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY, MARGIN, NEGATIVE_SAMPLES_RATIO

# Define available models
MODEL_CHOICES = ['complex', 'conve', 'transe']
OPTIMIZER_CHOICES = ['Adagrad', 'Adam', 'SGD']  # Added the optimizer choices

parser = argparse.ArgumentParser(description="Model-agnostic tool for verifying link predictions explanations")

parser.add_argument('--model', choices=MODEL_CHOICES, required=True, help="Model to use: complex, conve, transe")
parser.add_argument('--dataset', choices=ALL_DATASET_NAMES, required=True, help="Dataset to use")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model to explain the predictions of")
parser.add_argument('--mode', type=str, default="sufficient", choices=["sufficient", "necessary"], help="The explanation mode")
parser.add_argument('--optimizer', choices=OPTIMIZER_CHOICES, default='Adagrad', help="Optimizer to use")  # Added this argument
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--max_epochs', type=int, default=1000, help="Number of epochs")
parser.add_argument('--dimension', type=int, default=200, help="Factorization rank or embedding dimension")
parser.add_argument('--learning_rate', type=float, default=0.0005, help="Learning rate")
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
                       REGULARIZER_NAME: "N3"}
    model_class = ComplEx
    optimizer_class = MultiClassNLLOptimizer

elif args.model == 'conve':
    hyperparameters = {DIMENSION: args.dimension, INPUT_DROPOUT: args.input_dropout, FEATURE_MAP_DROPOUT: args.feature_map_dropout,
                       HIDDEN_DROPOUT: args.hidden_dropout, HIDDEN_LAYER_SIZE: args.hidden_size, BATCH_SIZE: args.batch_size,
                       LEARNING_RATE: args.learning_rate, DECAY: args.decay1, LABEL_SMOOTHING: args.label_smoothing, EPOCHS: args.max_epochs}
    model_class = ConvE
    optimizer_class = BCEOptimizer

elif args.model == 'transe':
    hyperparameters = {DIMENSION: args.dimension, MARGIN: args.margin, NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                       REGULARIZER_WEIGHT: args.reg, BATCH_SIZE: args.batch_size, LEARNING_RATE: args.learning_rate, EPOCHS: args.max_epochs}
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
if args.mode == "sufficient":
    sample_to_explain_2_entities_to_convert = {}

    i = 0
    while i <= len(input_lines) - 4:
        fact_line = input_lines[i]
        similar_entities_line = input_lines[i + 1]
        rules_line = input_lines[i + 2]
        empty_line = input_lines[i + 3]

        fact = tuple(fact_line.strip().split(";"))
        facts_to_explain.append(fact)
        sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
        samples_to_explain.append(sample)

        similar_entities_names = similar_entities_line.strip().split(",")
        similar_entities = [dataset.entity_name_2_id[x] for x in similar_entities_names]
        sample_to_explain_2_entities_to_convert[sample] = similar_entities

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
        i += 4

    # Process training and prediction
    samples_to_add = []
    samples_to_convert = []
    sample_to_convert_2_original_sample_to_explain = {}
    samples_to_convert_2_added_samples = {}

    for sample_to_explain in samples_to_explain:
        entity_to_explain = sample_to_explain[0] if perspective == "head" else sample_to_explain[2]
        cur_entities_to_convert = sample_to_explain_2_entities_to_convert[sample_to_explain]
        cur_best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]

        for cur_entity_to_convert in cur_entities_to_convert:
            cur_sample_to_convert = Dataset.replace_entity_in_sample(sample=sample_to_explain, old_entity=entity_to_explain, new_entity=cur_entity_to_convert, as_numpy=False)
            cur_samples_to_add = Dataset.replace_entity_in_samples(samples=cur_best_rule_samples, old_entity=entity_to_explain, new_entity=cur_entity_to_convert, as_numpy=False)

            samples_to_convert.append(cur_sample_to_convert)
            samples_to_convert_2_added_samples[cur_sample_to_convert] = cur_samples_to_add

            for cur_sample_to_add in cur_samples_to_add:
                samples_to_add.append(cur_sample_to_add)

            sample_to_convert_2_original_sample_to_explain[tuple(cur_sample_to_convert)] = sample_to_explain

    new_dataset = copy.deepcopy(dataset)

    # Handle conflicting samples and add new samples to dataset
    print("Adding samples: ")
    for (head, relation, tail) in samples_to_add:
        print("\t" + dataset.printable_sample((head, relation, tail)))
        if new_dataset.relation_2_type[relation] in [MANY_TO_ONE, ONE_TO_ONE]:
            for pre_existing_tail in new_dataset.to_filter[(head, relation)]:
                new_dataset.remove_training_sample(numpy.array((head, relation, pre_existing_tail)))

    new_dataset.add_training_samples(numpy.array(samples_to_add))

    # Obtain original predictions
    original_scores, original_ranks, original_predictions = model.predict_samples(numpy.array(samples_to_convert))

    # Train new model and obtain new predictions
    new_model = model_class(dataset=new_dataset, hyperparameters=hyperparameters, init_random=True)
    new_optimizer = optimizer_class(model=new_model, hyperparameters=hyperparameters)
    new_optimizer.train(train_samples=new_dataset.train_samples)
    new_model.eval()

    new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_convert))

    # Print comparison results
    for i in range(len(samples_to_convert)):
        cur_sample = samples_to_convert[i]
        original_direct_score = original_scores[i][0]
        original_tail_rank = original_ranks[i][1]

        new_direct_score = new_scores[i][0]
        new_tail_rank = new_ranks[i][1]

        print("<" + ", ".join([dataset.entity_id_2_name[cur_sample[0]], dataset.relation_id_2_name[cur_sample[1]], dataset.entity_id_2_name[cur_sample[2]]]) + ">")
        print("\tDirect score: from " + str(original_direct_score) + " to " + str(new_direct_score))
        print("\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank))
        print()

    # Write results to CSV
    output_lines = []
    for i in range(len(samples_to_convert)):
        cur_sample_to_convert = samples_to_convert[i]
        cur_added_samples = samples_to_add[i]
        original_sample_to_explain = sample_to_convert_2_original_sample_to_explain[tuple(cur_sample_to_convert)]

        original_direct_score = original_scores[i][0]
        original_tail_rank = original_ranks[i][1]

        new_direct_score = new_scores[i][0]
        new_tail_rank = new_ranks[i][1]

        a = ";".join(dataset.sample_to_fact(original_sample_to_explain))
        b = ";".join(dataset.sample_to_fact(cur_sample_to_convert))

        c = []
        samples_to_add_to_this_entity = samples_to_convert_2_added_samples[cur_sample_to_convert]
        for x in range(4):
            if x < len(samples_to_add_to_this_entity):
                c.append(";".join(dataset.sample_to_fact(samples_to_add_to_this_entity[x])))
            else:
                c.append(";;")

        c = ";".join(c)
        d = str(original_direct_score) + ";" + str(new_direct_score)
        e = str(original_tail_rank) + ";" + str(new_tail_rank)
        output_lines.append(";".join([a, b, c, d, e]) + "\n")

    with open("output_end_to_end.csv", "w") as outfile:
        outfile.writelines(output_lines)
