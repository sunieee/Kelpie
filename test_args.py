import yaml
import argparse

def read_yaml(file_path):
    """
    Read a YAML file and parse it into a dictionary.
    
    :param file_path: str, the path to the YAML file
    :return: dict, the parsed YAML content as a dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10 = "YAGO3-10"

ALL_DATASET_NAMES = [FB15K, FB15K_237, WN18, WN18RR, YAGO3_10]
MODEL_CHOICES = ['complex', 'conve', 'transe']

parser = argparse.ArgumentParser(description="Model-agnostic tool for verifying link predictions explanations")

parser.add_argument('--model', choices=MODEL_CHOICES, required=True, help="Model to use: complex, conve, transe")
parser.add_argument('--dataset', choices=ALL_DATASET_NAMES, required=True, help="Dataset to use")
parser.add_argument('--reg', type=float, default=0, help="Regularization weight")

args = parser.parse_args()
model = args.model
dataset = args.dataset
config = read_yaml("config.yaml")
print(config[f'{model}_{dataset}'])

for item in config[f'{model}_{dataset}']:
    for k, v in item.items():
        setattr(args, k, v)  # Use setattr to add/modify attributes in args

print(args)
