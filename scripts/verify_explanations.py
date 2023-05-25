import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import argparse
import copy
import numpy
import torch

from dataset import ALL_DATASET_NAMES, Dataset, MANY_TO_ONE, ONE_TO_ONE
from link_prediction.models.conve import ConvE
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.models.model import DIMENSION, LEARNING_RATE, EPOCHS, \
    BATCH_SIZE, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, HIDDEN_LAYER_SIZE, \
    LABEL_SMOOTHING, DECAY

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--dataset",
                    type=str,
                    choices=ALL_DATASET_NAMES,
                    help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")

parser.add_argument("--max_epochs",
                    type=int, default=1000,
                    help="Number of epochs.")

parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Batch size.")

parser.add_argument("--learning_rate",
                    type=float,
                    default=0.0005,
                    help="Learning rate.")

parser.add_argument("--decay_rate",
                    type=float,
                    default=1.0,
                    help="Decay rate.")

parser.add_argument("--dimension",
                    type=int,
                    default=200,
                    help="Embedding dimensionality.")

parser.add_argument("--input_dropout",
                    type=float,
                    default=0.3,
                    nargs="?",
                    help="Input layer dropout.")

parser.add_argument("--hidden_dropout",
                    type=float,
                    default=0.4,
                    help="Dropout after the hidden layer.")

parser.add_argument("--feature_map_dropout",
                    type=float,
                    default=0.5,
                    help="Dropout after the convolutional layer.")

parser.add_argument("--label_smoothing",
                    type=float,
                    default=0.1,
                    help="Amount of label smoothing.")

parser.add_argument('--hidden_size',
                    type=int,
                    default=9728,
                    help='The side of the hidden layer. '
                         'The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')

parser.add_argument("--model_path",
                    type=str,
                    required=True,
                    help="path of the model to explain the predictions of.")

args = parser.parse_args()

torch.backends.cudnn.deterministic = True
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed_all(seed)

########## LOAD DATASET


# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

# get the ids of the elements of the fact to explain and the perspective entity

hyperparameters = {DIMENSION: args.dimension,
                   INPUT_DROPOUT: args.input_dropout,
                   FEATURE_MAP_DROPOUT: args.feature_map_dropout,
                   HIDDEN_DROPOUT: args.hidden_dropout,
                   HIDDEN_LAYER_SIZE: args.hidden_size,
                   BATCH_SIZE: args.batch_size,
                   LEARNING_RATE: args.learning_rate,
                   DECAY: args.decay_rate,
                   LABEL_SMOOTHING: args.label_smoothing,
                   EPOCHS: args.max_epochs}

with open("output.txt", "r") as input_file:
    input_lines = input_file.readlines()


original_model = ConvE(dataset=dataset,
                hyperparameters=hyperparameters,
                init_random=True) # type: ConvE
original_model.to('cuda')
original_model.load_state_dict(torch.load(args.model_path))
original_model.eval()


facts_to_explain = []
samples_to_explain = []
perspective = "head"    # for all samples the perspective was head for simplicity
sample_to_explain_2_best_rule = {}

i = 0
while i <= len(input_lines) -3:
    fact_line = input_lines[i]
    rules_line = input_lines[i + 1]
    empty_line = input_lines[i + 2]

    # sample to explain
    fact = tuple(fact_line.strip().split(";"))
    facts_to_explain.append(fact)
    sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
    samples_to_explain.append(sample)

    # rules
    rules_with_relevance = []

    if rules_line.strip() != "":
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
    else:
        sample_to_explain_2_best_rule[sample] = []

    i += 3

samples_to_remove = []  # the samples to remove from the training set before retraining

for sample_to_explain in samples_to_explain:
    best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]
    samples_to_remove += best_rule_samples

new_dataset = copy.deepcopy(dataset)

print("Removing samples: ")
for (head, relation, tail) in samples_to_remove:
    print("\t" + dataset.printable_sample((head, relation, tail)))

# remove the samples_to_remove from training samples of new_dataset (and update new_dataset.to_filter accordingly)
new_dataset.remove_training_samples(numpy.array(samples_to_remove))

# obtain tail ranks and scores of the original model for all samples_to_explain
original_scores, original_ranks, original_predictions = original_model.predict_samples(numpy.array(samples_to_explain))

######

new_model = ConvE(dataset=new_dataset,
                    hyperparameters=hyperparameters,
                    init_random=True)  # type: ConvE
new_optimizer = BCEOptimizer(model=new_model, hyperparameters=hyperparameters)
new_optimizer.train(train_samples=new_dataset.train_samples)
new_model.eval()

new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_explain))

for i in range(len(samples_to_explain)):
    cur_sample = samples_to_explain[i]
    original_direct_score = original_scores[i][0]
    original_tail_rank = original_ranks[i][1]

    new_direct_score = new_scores[i][0]
    new_tail_rank = new_ranks[i][1]

    print("<" + ", ".join([dataset.entity_id_2_name[cur_sample[0]],
                            dataset.relation_id_2_name[cur_sample[1]],
                            dataset.entity_id_2_name[cur_sample[2]]]) + ">")
    print("\tDirect score: from " + str(original_direct_score) + " to " + str(new_direct_score))
    print("\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank))
    print()

output_lines = []
for i in range(len(samples_to_explain)):
    cur_sample_to_explain = samples_to_explain[i]

    original_direct_score = original_scores[i][0]
    original_tail_rank = original_ranks[i][1]

    new_direct_score = new_scores[i][0]
    new_tail_rank = new_ranks[i][1]

    # original_head, original_relation, original_tail,
    # fact_to_convert_head, fact_to_convert_rel, fact_to_convert_tail,

    # original_direct_score, new_direct_score,
    #  original_tail_rank, new_tail_rank

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