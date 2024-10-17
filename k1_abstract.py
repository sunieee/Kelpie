from typing import Tuple, Any
from dataset import Dataset
from prefilters.no_prefilter import NoPreFilter
from prefilters.prefilter import TYPE_PREFILTER, TOPOLOGY_PREFILTER, NO_PREFILTER
from prefilters.type_based_prefilter import TypeBasedPreFilter
from prefilters.topology_prefilter import TopologyPreFilter
from relevance_engines.post_training_engine import PostTrainingEngine
from link_prediction.models.model import Model
from explanation_builders.stochastic_necessary_builder import StochasticNecessaryExplanationBuilder
from explanation_builders.stochastic_sufficient_builder import StochasticSufficientExplanationBuilder
import json
import os
from collections import deque, defaultdict
import re
# from flask_cors import CORS
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})

max_length = 3
dataset2triples = {}
explanations = {}

def read_train_triples(dataset):
    if dataset in dataset2triples:
        return dataset2triples[dataset]

    train_file = f"data/{dataset}/train.txt"
    head_to_triples = defaultdict(set)
    tail_to_triples = defaultdict(set)
    
    with open(train_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triple = ','.join(parts)
                head_to_triples[parts[0]].add(triple)
                tail_to_triples[parts[2]].add(triple)
    
    print(f"Loaded {len(head_to_triples)} head entities and {len(tail_to_triples)} tail entities for {dataset}")
    dataset2triples[dataset] = (head_to_triples, tail_to_triples)
    return head_to_triples, tail_to_triples


def find_paths_bfs(head_to_triples, tail_to_triples, head_id, tail_id):
    # 使用BFS进行路径搜索
    paths = []
    queue = deque([(head_id, [])])  # 队列中存储当前节点和路径
    visited = set()  # 记录已访问节点
    
    while queue:
        current_id, current_path = queue.popleft()
        
        if current_id == tail_id and len(current_path) > 0:
            paths.append(current_path[:])  # 记录从head到tail的路径
            continue

        if len(current_path) >= max_length:
            continue
        
        visited.add(current_id)
        
        # 从head映射中找到所有以current_id为head的三元组
        for triple in head_to_triples.get(current_id, set()):
            t = triple.split(',')[2]
            if t not in visited:  # triple[2] 是 tail
                queue.append((t, current_path + [triple]))
        
        # 从tail映射中找到所有以current_id为tail的三元组（反向边）
        for triple in tail_to_triples.get(current_id, set()):
            t = triple.split(',')[0]
            if t not in visited:  # triple[0] 是 head
                queue.append((t, current_path + [triple]))

    return paths


def search_subgraph(prediction, dataset):
    head_to_triples, tail_to_triples = read_train_triples(dataset)
    head_id, _, tail_id = prediction.split(',')
    
    # 使用BFS查找路径
    paths = find_paths_bfs(head_to_triples, tail_to_triples, head_id, tail_id)
    
    # Collect unique triples used in paths
    triples_map = {}
    paths_with_triples = []
    
    for path in paths:
        path_indices = []
        for triple in path:
            # 用一个元组表示triple，避免使用不可哈希的字典
            if triple not in triples_map:
                l = len(triples_map)
                triples_map[triple] = l
            path_indices.append(triples_map[triple])
        paths_with_triples.append(path_indices)

    return {
        "prediction": prediction,
        "triples": [t[0] for t in sorted(triples_map.items(), key=lambda x: x[1])],
        "paths": paths_with_triples
    }


def search_ht_relation(prediction, dataset):
    ret = search_subgraph(prediction, dataset)
    head, _, tail = prediction.split(',')
    head_relation2triples = defaultdict(list)    
    tail_relation2triples = defaultdict(list)
    for path in ret['paths']:
        path_vector = [head]
        current_id = head
        for ix in path:
            head_id, relation_id, tail_id = ret['triples'][ix].split(',')
            if head_id == current_id:
                current_id = tail_id
            else:
                current_id = head_id
                relation_id = f"{relation_id}'"
            path_vector.append(relation_id)
            path_vector.append(current_id)
        assert current_id == tail
        head_relation2triples[path_vector[1]].append(ret['triples'][path[0]])
        tail_relation2triples[path_vector[-2]].append(ret['triples'][path[-1]])

    return {
        **ret,
        'head_relation2triples': head_relation2triples,
        'tail_relation2triples': tail_relation2triples
    }




class K1_asbtract:
    """
    The Kelpie object is the overall manager of the explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations
    to the Pre-Filter, Explanation Builder and Relevance Engine modules.
    """

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 relevance_threshold: float = None,
                 max_explanation_length: int = 1):
        """
        Kelpie object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param prefilter_type: the type of prefilter to employ
        :param relevance_threshold: the threshold of relevance that, if exceeded, leads to explanation acceptance
        :param max_explanation_length: the maximum number of facts that the explanations to extract can contain
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.relevance_threshold = relevance_threshold
        self.max_explanation_length = max_explanation_length
        self.engine = PostTrainingEngine(model=model,
                                         dataset=dataset,
                                         hyperparameters=hyperparameters)


    def explain_necessary(self,
                          sample_to_explain: Tuple[Any, Any, Any],
                          perspective: str,
                          num_promising_samples: int = 50):
        """
        This method extracts necessary explanations for a specific sample,
        from the perspective of either its head or its tail.

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the sample head and relation, why is the sample tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the sample relation and tail, why is the sample head predicted as head?"
        :param num_promising_samples: the number of samples relevant to the sample to explain
                                     that must be identified and removed from the entity under analysis
                                     to verify whether they worsen the target prediction or not

        :return: a list containing for each relevant n-ple extracted, a couple containing
                                - that relevant n-ple
                                - its value of relevance

        """
        # sample 是序号三元组
        # fact 是id三元组
        
        relation_map = {}
        print('sample_to_explain', sample_to_explain)
        self.perspective = perspective
        
        data = search_ht_relation(','.join(self.dataset.sample_to_fact(sample_to_explain)), self.dataset.name)
        if perspective == 'head':
            relation2triples = data['head_relation2triples']

            # this is an exception: all samples (= rules with length 1) are tested
            i = 0
            for relation, triples in relation2triples.items():
                i += 1
                print("\n\tComputing relevance for sample " + str(i) + " on " + str(
                    len(triples)) + "(head relation): " + relation)
                relevance = self._compute_relevance_for_rule(sample_to_explain, [self.dataset.fact_to_sample(t.split(',')) for t in triples])
                relation_map[relation] = {
                    'perspective': 'head',
                    'relation': relation,
                    'triples': triples,
                    'relevance': relevance
                }
                print("\tObtained relevance: " + str(relevance))
        
        # sort the relation_map by relevance
        relation_list = relation_map.values()
        relation_list.sort(key=lambda x: x['relevance'], reverse=True)

        # save the relation_list to a json file
        with open('output.json', 'w') as f:
            json.dump(relation_list, f)

        return relation_list
    

    def _compute_relevance_for_rule(self, sample_to_explain, nple_to_remove: list):
        rule_length = len(nple_to_remove)

        # convert the nple to remove into a list
        assert (len(nple_to_remove[0]) == 3)

        relevance, \
        original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
        base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank, \
        pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, execution_time = \
            self.engine.removal_relevance(sample_to_explain=sample_to_explain,
                                          perspective=self.perspective,
                                          samples_to_remove=nple_to_remove)

        cur_line = ";".join(self.dataset.sample_to_fact(sample_to_explain)) + ";" + \
                   ";".join([";".join(self.dataset.sample_to_fact(x)) for x in nple_to_remove]) + ";" + \
                   str(original_target_entity_score) + ";" + \
                   str(original_target_entity_rank) + ";" + \
                   str(base_pt_target_entity_score) + ";" + \
                   str(base_pt_target_entity_rank) + ";" + \
                   str(pt_target_entity_score) + ";" + \
                   str(pt_target_entity_rank) + ";" + \
                   str(relevance) + ";" + \
                   str(execution_time)

        with open("output_details_" + str(rule_length) + ".csv", "a") as output_file:
            output_file.writelines([cur_line + "\n"])

        return relevance