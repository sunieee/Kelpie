import json
import os
from collections import deque, defaultdict
from tqdm import tqdm
from threading import Thread
import re
import numpy as np
import numpy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import threading
from dataset import ALL_DATASET_NAMES
# from flask_cors import CORS
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})

CALCULATE_METRIC = True
MODEL_CHOICES = ['complex', 'conve', 'transe']

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")
parser.add_argument('--dataset', choices=ALL_DATASET_NAMES, required=True, help="Dataset to use")
parser.add_argument('--path', required=True, help="Input path to use")
parser.add_argument('--max_hop', default=3, help="The length of max hop of paths")

args = parser.parse_args()
max_length = args.max_hop

dataset2triples = {}
dataset_map = {}
explanations = {}

def read_dateset(dataset):
    if dataset in dataset_map:
        return dataset_map[dataset]

    print(f'[read_dateset] loading {dataset}')
    train_file = f"data/{dataset}/train.txt"
    entity_file = f"data/{dataset}/entities.dict"
    head_to_triples = defaultdict(set)
    tail_to_triples = defaultdict(set)

    entity_to_index = {}
    with open(entity_file, 'r') as file:
        for line in file:
            ix, entity = line.split('\t')
            entity_to_index[entity.strip()] = int(ix)

    num_entities = len(entity_to_index)
    relation_to_matrix = defaultdict(lambda: np.zeros((num_entities, num_entities), dtype=int))
    relation_to_triples = defaultdict(set)
    # triples = set()
    
    with open(train_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            triple = ','.join(parts)
            # triples.add(triple)
            
            # Update mappings
            head_to_triples[parts[0]].add(triple)
            tail_to_triples[parts[2]].add(triple)
            relation_to_triples[parts[1]].add(triple)
            
            # Update relation-specific adjacency matrix
            h_idx = entity_to_index[parts[0]]
            t_idx = entity_to_index[parts[2]]
            relation_to_matrix[parts[1]][h_idx, t_idx] = 1

    # return triples, relation_to_matrix, head_to_triples, relation_to_triples
    ret = {
        # 'triples': triples,
        'head_to_triples': head_to_triples,
        'tail_to_triples': tail_to_triples,
        'relation_to_matrix': relation_to_matrix,
        'relation_to_triples': relation_to_triples,
        'entity_to_index': entity_to_index,
    }
    dataset_map[dataset] = ret
    print(f'[read_dateset] {dataset} loaded')
    return ret


def read_triples(dataset):
    if dataset in dataset2triples:
        return dataset2triples[dataset]
    
    relation_to_triples = defaultdict(set)
    head_to_triples = defaultdict(set)
    tail_to_triples = defaultdict(set)
    
    print(f'[read_dateset] loading {dataset}')
    train_file = f"data/{dataset}/train.txt"
    with open(train_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            triple = ','.join(parts)
            
            head_to_triples[parts[0]].add(triple)
            tail_to_triples[parts[2]].add(triple)
            relation_to_triples[parts[1]].add(triple)

    ret = {
        # 'triples': triples,
        'head_to_triples': head_to_triples,
        'tail_to_triples': tail_to_triples,
        'relation_to_triples': relation_to_triples,
    }
    dataset2triples[dataset] = ret
    return ret


def calculate_rule_metrics(head_rel, body_relations, dataset):
    filename = body_relations.replace('/', '_')
    if len(filename) > 200:
        filename = filename[::2]
    file_path = f"json/{dataset}/{head_rel.replace('/', '_')}/{filename}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    t = time.time()

    data = read_triples(dataset)
    train_rel_to_triples = data['relation_to_triples']
    train_head_to_triples = data['head_to_triples']
    head2entity_map = {}

    # 解析 body 关系
    body_rels = body_relations.split(',')
    for i in range(1, len(body_rels) + 1):
        head2entity_map[i] = defaultdict(set) # hop -> entity_map (head -> entity set)

    # 计算 body 匹配数和 support
    rel = body_rels[0]
    reverse = False
    if rel.endswith("'"):
        rel = rel[:-1]
        reverse = True
    for triple in train_rel_to_triples[rel]:
        if reverse:
            tail, _, head = triple.split(',')
        else:
            head, _, tail = triple.split(',')
        head2entity_map[1][head].add(tail)

    ix = 1
    for rel in body_rels[1:]:
        ix += 1
        reverse = False
        if rel.endswith("'"):
            rel = rel[:-1]
            reverse = True
        for triple in train_rel_to_triples[rel]:
            if reverse:
                tail, _, head = triple.split(',')
            else:
                head, _, tail = triple.split(',')
            for k, v in head2entity_map[ix - 1].items():
                if head in v:
                    head2entity_map[ix][k].add(tail)

    # 计算 head count
    head_count = len(train_rel_to_triples[head_rel])
    body_count = 0
    supp = 0

    for k, v in head2entity_map[len(body_rels)].items():
        body_count += len(v)
        tails = [t.split(',')[2] for t in train_rel_to_triples[head_rel] if t.split(',')[0] == k]
        supp += len(v.intersection(tails))

    # 计算 SC 和 HC
    SC = supp / body_count if body_count > 0 else 0
    HC = supp / head_count if head_count > 0 else 0

    ret = {
        'supp': supp,
        'body': body_count,
        'head': head_count,
        'HC': HC,
        'SC': SC
    }

    print(f"Time: {time.time() - t}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ret, f)

    return ret


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


def search_subgraph(prediction, dataset, condense=False):
    file_name = prediction.replace('/', '+')
    file_path = f"json/{dataset}/{file_name}.json"

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
        
    data = read_triples(dataset)
    head_to_triples = data['head_to_triples']
    tail_to_triples = data['tail_to_triples']
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

    ret = {
        "prediction": prediction,
        "triples": [t[0] for t in sorted(triples_map.items(), key=lambda x: x[1])],
        "paths": paths_with_triples
    }
    if condense:
        ret.update(condense_graph(ret))

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ret, f)
    return ret


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
        'head_relation2triples': head_relation2triples,
        'tail_relation2triples': tail_relation2triples
    }


def condense_graph(data, max_hop=3):
    relation_entity_map = {}  # 用于合并关系路径并生成 abstract node
    abstract_nodes_map = {}  # 用于存储抽象节点
    abstract_edges_map = {}  # 用于存储抽象边
    statistics = []  # 用于存储统计信息
    relation_path_map = defaultdict(list)  # relation_path2paths

    # 首先过滤掉超过 max_hop 的 paths
    paths = [path for path in data['paths'] if len(path) <= max_hop]
    head, relation, tail = data['prediction'].split(',')
    abstract_node_id_ix = 0

    def merge_into(old_abstract_node, abstract_node):
        if old_abstract_node == abstract_node:
            return
        print(f'merging {old_abstract_node} into {abstract_node}')
        
        # 合并relation_entities并删除old_abstract_node
        abstract_nodes_map[abstract_node]['relation_entities'].update(
            abstract_nodes_map.pop(old_abstract_node, {}).get('relation_entities', {})
        )

        # 更新abstract_edges_map中的边信息
        removed_keys = set()
        for k, v in abstract_edges_map.items():
            h, r, t = k.split('-')
            if h == old_abstract_node:
                abstract_edges_map['-'.join([old_abstract_node, r, t])]['pairs'].update(v['pairs'])
                removed_keys.add(k)
            elif t == old_abstract_node:
                abstract_edges_map['-'.join([h, r, old_abstract_node])]['pairs'].update(v['pairs'])
                removed_keys.add(k)
        for k in removed_keys:
            abstract_edges_map[k] = {}
            del abstract_edges_map[k]

        # 更新relation_entity_map中的abstract_node_id
        for v in relation_entity_map.values():
            if v['abstract_node_id'] == old_abstract_node:
                v['abstract_node_id'] = abstract_node

    def addEdge(name, entity_pair):
        h, r, t = name.split('-')
        if name in abstract_edges_map:
            abstract_edges_map[name]['pairs'].add(entity_pair)
        else:
            abstract_edges_map[name] = {
                'source': h,
                'target': t,
                'relation': r,
                'pairs': set([entity_pair])
            }

    # 统计所需的各种集合
    hop_triple_ix_set = set()
    head_relation_set = set()
    tail_relation_set = set()
    head_triple_ix_set = set()
    tail_triple_ix_set = set()
    entity_path_set = set()
    path_count = 0
    for current_hop in range(1, max_hop + 1):
        current_paths = [p for p in paths if len(p) == current_hop]
        for path in tqdm(current_paths):
            path_count += 1
            path_vector = [head]
            current_id = head
            for ix in path:
                head_id, relation_id, tail_id = data['triples'][ix].split(',')
                if head_id == current_id:
                    current_id = tail_id
                else:
                    current_id = head_id
                    relation_id = f"{relation_id}'" # 用'代表反向关系
                path_vector.append(relation_id)
                path_vector.append(current_id)
            assert current_id == tail

            entity_path = path_vector[::2]
            relation_path = path_vector[1::2]
            entity_path_set.add(','.join(entity_path))
            # 记录path所在的index
            relation_path_map[','.join(relation_path)].append(path_count - 1)

            hop_triple_ix_set.update(path)
            head_relation_set.add(relation_path[0])
            tail_relation_set.add(relation_path[-1])
            head_triple_ix_set.add(path[0])
            tail_triple_ix_set.add(path[-1])

            source = head
            for k in range(1, current_hop):
                relation_entity_head = ','.join(relation_path[:k]) + ','
                relation_entity_tail = ',' + ','.join(relation_path[k:])
                
                if relation_entity_head in relation_entity_map and relation_entity_tail in relation_entity_map:
                    abstract_node_id = relation_entity_map[relation_entity_head]['abstract_node_id']
                    replace_abstract_node_id = relation_entity_map[relation_entity_tail]['abstract_node_id']
                    merge_into(replace_abstract_node_id, abstract_node_id)
                    abstract_nodes_map[abstract_node_id]['relation_entities'].add(relation_entity_tail)

                elif relation_entity_head in relation_entity_map:
                    abstract_node_id = relation_entity_map[relation_entity_head]['abstract_node_id']
                    abstract_nodes_map[abstract_node_id]['relation_entities'].add(relation_entity_tail)
                elif relation_entity_tail in relation_entity_map:
                    abstract_node_id = relation_entity_map[relation_entity_tail]['abstract_node_id']
                    abstract_nodes_map[abstract_node_id]['relation_entities'].add(relation_entity_head)
                else:
                    abstract_node_id = str(abstract_node_id_ix)
                    abstract_node_id_ix += 1
                    abstract_nodes_map[abstract_node_id] = {
                        'id': abstract_node_id,
                        'type': 'abstract',
                        'relation_entities': set([relation_entity_head, relation_entity_tail])
                    }

                if relation_entity_head in relation_entity_map:
                    relation_entity_map[relation_entity_head]['entities'].add(entity_path[k])
                else:
                    relation_entity_map[relation_entity_head] = {
                        'id': relation_entity_head,
                        'abstract_node_id': abstract_node_id,
                        'entities': set([entity_path[k]])
                    }
                    
                if relation_entity_tail in relation_entity_map:
                    relation_entity_map[relation_entity_tail]['entities'].add(entity_path[k])
                else:
                    relation_entity_map[relation_entity_tail] = {
                        'id': relation_entity_tail,
                        'abstract_node_id': abstract_node_id,
                        'entities': set([entity_path[k]])
                    }

                addEdge(f'{source}-{relation_path[k-1]}-{abstract_node_id}', f'{entity_path[k-1]}-{entity_path[k]}')
                source = abstract_node_id
            
            addEdge(f'{source}-{relation_path[-1]}-{tail}', f'{entity_path[-2]}-{entity_path[-1]}')

        # 更新统计信息
        statistics.append({
            'current_hop': current_hop,
            'triple_count': len(hop_triple_ix_set),
            'head_relation_count': len(head_relation_set),
            'tail_relation_count': len(tail_relation_set),
            'head_triple_count': len(head_triple_ix_set),
            'tail_triple_count': len(tail_triple_ix_set),
            'path_count': len(current_paths),
            'path_count': len(current_paths),
            'relation_path_count': len(relation_path_map),
            'entity_path_count': len(entity_path_set),
        })

    return set_to_list({
        'abstract_nodes': list(abstract_nodes_map.values()),
        'abstract_edges': list(abstract_edges_map.values()),
        'relation_entities_map': relation_entity_map,
        'statistics': statistics,
        'relation_path_map': relation_path_map
    })


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (numpy.bool_,)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

def set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: set_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [set_to_list(i) for i in obj]
    else:
        return obj
    
# model = 'ComplEx'
# dataset = 'FB15k-237'
# setting = 'V3'
# filename = f"json/{model}/{dataset}/{setting}.json"
dataset = args.dataset
input_path = os.path.join(args.path, 'output.json')
output_path = os.path.join(args.path, 'process')
with open(input_path, 'r') as file:
    predictions = json.load(file)

all_keys = [','.join(t['prediciton']) for t in predictions]

def calculateConfidence(ls):
    unconfidence = 1
    for l in ls:
        unconfidence *= 1 - l
    return 1 - unconfidence


def calculatePrediction(prediction):
    print('# calculating prediction: ', prediction['prediction'])
    head = prediction['prediction'][0]
    relation = prediction['prediction'][1]
    tail = prediction['prediction'][2]
    pred = ','.join(prediction['prediction'])
    explanations = prediction['explanation']
    
    while True:
        predictionData = search_subgraph(pred, dataset, True)
        if 'abstract_edges' in predictionData:
            break
        print('retrying prediction: ', pred)
        file_name = pred.replace('/', '+')
        file_path = f"json/{dataset}/{file_name}.json"
        os.remove(file_path)

    abstract_edges = predictionData['abstract_edges']

    headEdges = []
    tailEdges = []
    headEdgeAll = None
    tailEdgeAll = None
    for d in explanations:
        for e in abstract_edges:
            if d['relation'] == e['relation'] and (e['source'] == head and d['perspective'] == 'head' or e['target'] == tail and d['perspective'] == 'tail'):
                d.update(e)
        if d['perspective'] == 'head':
            if d['relation'] == 'all':
                headEdgeAll = d
            else:
                headEdges.append(d)
        elif d['perspective'] == 'tail':
            if d['relation'] == 'all':
                tailEdgeAll = d
            else:
                tailEdges.append(d)

    facts_map = {}
    for k, v in predictionData['relation_path_map'].items():
        headR = k.split(',')[0]
        tailR = k.split(',')[-1]
        Rh = 0
        Rt = 0
        for e in headEdges:
            if e['relation'] == headR and 'score_deduction' in e:
                Rh = e['score_deduction']
        for e in tailEdges:
            if e['relation'] == tailR and 'score_deduction' in e:
                Rt = e['score_deduction']

        ret = {
            'id': k,
            'length': len(v),
            'paths': v,
            'Rh': Rh,
            'Rt': Rt,
            'R': Rh * Rt
        }
        if headEdgeAll['score_deduction'] < 0:
            ret['R'] = Rt
        if tailEdgeAll['score_deduction'] < 0:
            ret['R'] = Rh

        if CALCULATE_METRIC:
            if ret['R'] > 0:
                print('calculating rule metrics for ', k)
                ret.update(calculate_rule_metrics(relation, k, dataset))

                # if ret['SC'] < 0.1 or ret['HC'] < 0.01:
                #     print('low quality rule, skipping: ', k)
                #     continue

                ret['GA'] = geometricAverage([ret['SC'], ret['HC'], ret['R'] / 10000])
                ret['RHC'] = geometricAverage([ret['HC'], ret['R'] / 10000])
                ret['RSC'] = geometricAverage([ret['SC'], ret['R'] / 10000])
                ret['HCSC'] = geometricAverage([ret['SC'], ret['HC']])
                for p in ret['paths']:
                    for t in predictionData['paths'][p]:
                        if t in facts_map:
                            facts_map[t]['SC'].append(ret['SC'])
                            facts_map[t]['HC'].append(ret['HC'])
                            facts_map[t]['R'].append(ret['R'] / 10000)
                            facts_map[t]['GA'].append(ret['GA'])
                            facts_map[t]['RHC'].append(ret['RHC'])
                            facts_map[t]['RSC'].append(ret['RSC'])
                            facts_map[t]['HCSC'].append(ret['HCSC'])
                            facts_map[t]['#rule'] += 1
                        else:
                            facts_map[t] = {
                                'ix': t,
                                'triple': predictionData['triples'][t],
                                'SC': [ret['SC']],
                                'HC': [ret['HC']],
                                'R': [ret['R'] / 10000],
                                'GA': [ret['GA']],
                                'RHC': [ret['RHC']],
                                'RSC': [ret['RSC']],
                                'HCSC': [ret['HCSC']],
                                '#rule': 1
                            }
        else:
            if ret['R'] > 0:
                print('calculating rule metrics for ', k)
                # if ret['SC'] < 0.1 or ret['HC'] < 0.01:
                #     print('low quality rule, skipping: ', k)
                #     continue

                for p in ret['paths']:
                    for t in predictionData['paths'][p]:
                        if t in facts_map:
                            facts_map[t]['R'].append(ret['R'] / 10000)
                            facts_map[t]['#rule'] += 1
                        else:
                            facts_map[t] = {
                                'ix': t,
                                'triple': predictionData['triples'][t],
                                'R': [ret['R'] / 10000],
                                '#rule': 1
                            }

    for v in facts_map.values():
        v['R'] = calculateConfidence(v['R'])
        if CALCULATE_METRIC:
            v['SC'] = calculateConfidence(v['SC'])
            v['HC'] = calculateConfidence(v['HC'])
            v['GA'] = calculateConfidence(v['GA'])
            v['RHC'] = calculateConfidence(v['RHC'])
            v['RSC'] = calculateConfidence(v['RSC'])
            v['HCSC'] = calculateConfidence(v['HCSC'])
    extractedFacts = list(facts_map.values())
    # sort by onfidence_SC, confidence_HC, rules.length in order
    if CALCULATE_METRIC:
        extractedFacts.sort(key=lambda x: (x['SC'], x['HC'], x['#rule']), reverse=True)

    return extractedFacts
    
def geometricAverage(ls):
    return np.prod(ls) ** (1 / len(ls))

# extractedFactsMap = {}
# for prediction in tqdm(predictions):
#     pred = ','.join(prediction['prediction'])
#     extractedFactsMap[pred] = calculatePrediction(prediction)

#     filepath = f'json/{model}/{dataset}/{setting}-extractedFactsMap.json'
#     # os.makedirs(os.path.dirname(filepath), exist_ok=True)
#     with open(filepath, 'w') as file:
#         json.dump(extractedFactsMap, file, cls=NumpyEncoder)


def process_prediction(prediction):
    pred = ','.join(prediction['prediction'])
    result = calculatePrediction(prediction)
    
    # 构建文件保存路径，并替换非法字符
    pred_sanitized = pred.replace('/', '+')
    filepath = f'{output_path}/{pred_sanitized}.json'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存结果到单独的文件
    with open(filepath, 'w') as file:
        json.dump(result, file, cls=NumpyEncoder)

# 创建多进程池，多线程真的不行
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_prediction, prediction) for prediction in predictions]
    for future in tqdm(as_completed(futures), total=len(predictions)):
        future.result()  # 等待每个进程完成


ret = {}
for filename in os.listdir(output_path):
    if filename.endswith('.json') and len(filename.split(',')) == 3:
        with open(os.path.join(output_path, filename)) as f:
            key = filename[:-5].replace('+', '/')
            # 需要过滤掉不在output.json中的key
            if key in all_keys:
                ret[key] = json.load(f)

with open(f'{args.path}/extractedFactsMap.json', 'w') as f:
    json.dump(ret, f)