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
import html
from dataset import ALL_DATASET_NAMES
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
# from flask_cors import CORS
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}})

CALCULATE_METRIC = True
MODEL_CHOICES = ['complex', 'conve', 'transe']

minSC = 0.01
minHC = 0.01
minSupp = 10

dataset2triples = {}
dataset_map = {}
explanations = {}

# 注意一定要使用与kelpie相同的预处理方式
def read_txt(triples_path, separator="\t"):
    with open(triples_path, 'r') as file:
        lines = file.readlines()
    
    textual_triples = []
    for line in lines:
        line = html.unescape(line).lower()   # this is required for some YAGO3-10 lines
        head_name, relation_name, tail_name = line.strip().split(separator)

        # remove unwanted characters
        head_name = head_name.replace(",", "").replace(":", "").replace(";", "")
        relation_name = relation_name.replace(",", "").replace(":", "").replace(";", "")
        tail_name = tail_name.replace(",", "").replace(":", "").replace(";", "")

        textual_triples.append((head_name, relation_name, tail_name))
    return textual_triples


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
    
    # 使用LIL格式初始化关系矩阵
    relation_to_matrix = defaultdict(lambda: lil_matrix((num_entities, num_entities), dtype=int))
    relation_to_triples = defaultdict(set)

    textual_triples = read_txt(train_file)
    for head_name, relation_name, tail_name in textual_triples:
        triple = f"{head_name},{relation_name},{tail_name}"            
        # 更新映射
        head_to_triples[head_name].add(triple)
        tail_to_triples[tail_name].add(triple)
        relation_to_triples[relation_name].add(triple)
        
        # 更新关系特定的邻接矩阵
        h_idx = entity_to_index[head_name]
        t_idx = entity_to_index[tail_name]
        
        # 使用LIL格式进行更新
        relation_to_matrix[relation_name][h_idx, t_idx] = 1

    # 转换为CSR格式以提高后续的性能
    for rel in relation_to_matrix:
        relation_to_matrix[rel] = relation_to_matrix[rel].tocsr()

    # 返回字典
    ret = {
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
    textual_triples = read_txt(f"data/{dataset}/train.txt")
    
    print(f'[read_dateset] loading {dataset}')
    for head_name, relation_name, tail_name in textual_triples:
        triple = f"{head_name},{relation_name},{tail_name}"
        reverse_triple = f"{tail_name},{relation_name}',{head_name}"
        head_to_triples[head_name].add(triple)
        tail_to_triples[tail_name].add(triple)
        relation_to_triples[relation_name].add(triple)
        relation_to_triples[relation_name + "'"].add(reverse_triple)

    ret = {
        # 'triples': triples,
        'head_to_triples': head_to_triples,
        'tail_to_triples': tail_to_triples,
        'relation_to_triples': relation_to_triples,
    }
    dataset2triples[dataset] = ret
    return ret


def read_triples_bidirection(dataset):
    if dataset in dataset2triples:
        return dataset2triples[dataset]
    
    relation_to_triples = defaultdict(set)
    head_to_triples = defaultdict(set)
    tail_to_triples = defaultdict(set)
    textual_triples = read_txt(f"data/{dataset}/train.txt")
    
    print(f'[read_dateset] loading {dataset}')
    for head_name, relation_name, tail_name in textual_triples:
        triple = f"{head_name},{relation_name},{tail_name}"
        reverse_triple = f"{tail_name},{relation_name}',{head_name}"
        head_to_triples[head_name].add(triple)
        head_to_triples[tail_name].add(reverse_triple)
        tail_to_triples[tail_name].add(triple)
        tail_to_triples[head_name].add(reverse_triple)
        relation_to_triples[relation_name].add(triple)
        relation_to_triples[relation_name + "'"].add(reverse_triple)

    ret = {
        # 'triples': triples,
        'head_to_triples': head_to_triples,
        'tail_to_triples': tail_to_triples,
        'relation_to_triples': relation_to_triples,
    }
    dataset2triples[dataset] = ret
    return ret


def calculate_rule_metrics_with_matrix(head_rel, body_relations, dataset):
    filename = body_relations.replace('/', '_')
    if len(filename) > 200:
        filename = filename[::2]
    file_path = f"json/{dataset}/{head_rel.replace('/', '_')}/{filename}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    st = time.time()
    data = read_dateset(dataset)
    # Read triples and build matrices for the training set
    train_rel_to_matrix = data['relation_to_matrix']
    train_rel_to_triples = data['relation_to_triples']
    entity_to_index = data['entity_to_index']
    body_rels = body_relations.split(',')

    # Multiply matrices for each subsequent relation in the body
    first_rel = body_rels[0]
    print(f'[calculate_rule_metrics_with_matrix] relation0: {first_rel}')
    base_matrix = train_rel_to_matrix[first_rel[:-1]].T if first_rel.endswith("'") else train_rel_to_matrix[first_rel]

    # Multiply matrices for each subsequent relation in the body
    body_matrix = base_matrix
    for ix, body_rel in enumerate(body_rels[1:]):
        print(f'[calculate_rule_metrics_with_matrix] relation{ix}: {body_rel}')
        next_matrix = train_rel_to_matrix[body_rel[:-1]].T if body_rel.endswith("'") else train_rel_to_matrix[body_rel]
        # Multiply and binarize the result
        t = time.time()
        body_matrix = body_matrix.dot(next_matrix)  # 直接点乘
        body_matrix.data = np.ones_like(body_matrix.data)  # 二值化
        body_matrix.setdiag(0)  # 去除对角线上的1
        # body_matrix = (body_matrix > 0).astype(int)
        print(f'time: {time.time() - t}')

    # Calculate body count: number of non-zero entries in the matrix, not np.sum, because elements are not binary
    body_count = body_matrix.nnz

    # Calculate support (supp) for the rule in the training set using 'relation_to_triples'
    supp = 0
    for triple in train_rel_to_triples[head_rel]:
        head, rel, tail = triple.split(',')
        h_idx = entity_to_index.get(head, -1)
        t_idx = entity_to_index.get(tail, -1)
        if h_idx != -1 and t_idx != -1 and body_matrix[h_idx, t_idx] > 0:
            supp += 1

    # Calculate head count in the training set
    head_count = len(train_rel_to_triples[head_rel])

    # Calculate SC and HC
    SC = supp / body_count if body_count > 0 else 0
    HC = supp / head_count if head_count > 0 else 0

    ret = {
        'supp': supp,
        '#body': body_count,
        '#head': head_count,
        'HC': HC,
        'SC': SC
    }

    print(f"Total Time: {time.time() - st}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ret, f)

    return ret

def calculate_Ud_metrics(Ud_head, Ud_body, dataset):
    data = read_triples_bidirection(dataset)
    supp = 0
    head_constant = Ud_head.split('(')[1][:-1]
    head_constant = head_constant.replace('X,', '').replace(',Y', '')
    head_rel = Ud_head.split('(')[0]
    body_rel = Ud_body.split('(')[0]
    head_count = 0

    if '(X,' in Ud_body:
        # head U_d
        for f in data['relation_to_triples'][head_rel]:
            h, _, t = f.split(',')
            if t == head_constant:
                head_count += 1
                if len(data['head_to_triples'][h] & data['relation_to_triples'][body_rel]):
                    supp += 1
        # body_count = len(set([t.split(',')[0] + ',' + t.split(',')[1]  for t in data['relation_to_triples'][body_rel]]))
    elif ',Y)' in Ud_body:
        # tail U_d
        for f in data['relation_to_triples'][head_rel]:
            h, _, t = f.split(',')
            if h == head_constant:
                head_count += 1
                if len(data['tail_to_triples'][t] & data['relation_to_triples'][body_rel]):
                    supp += 1
        # body_count = len(set([t.split(',')[1] + ',' + t.split(',')[2]  for t in data['relation_to_triples'][body_rel]]))

    # body_count = len(set([t.split(',')[0] + ',' + t.split(',')[1]  for t in data['relation_to_triples'][body_rel]]))
    body_count = len(data['relation_to_triples'][body_rel])
    return {
        'supp': supp,
        '#body': body_count,
        '#head': head_count,
        'HC': supp / head_count if head_count > 0 else 0,
        'SC': supp / body_count if body_count > 0 else 0
    }


def calculate_Uc_metrics(Uc_head, Uc_body, dataset):
    data = read_triples_bidirection(dataset)
    supp = 0
    head_constant = Uc_head.split('(')[1][:-1]
    head_constant = head_constant.replace('X,', '').replace(',Y', '')
    head_rel = Uc_head.split('(')[0]
    body_rel = Uc_body.split('(')[0]
    body_constant = Uc_body.split('(')[1][:-1]
    body_constant = body_constant.replace('X,', '').replace(',Y', '')
    head_count = 0

    if '(X,' in Uc_body:
        # head U_c
        for f in data['relation_to_triples'][head_rel]:
            h, _, t = f.split(',')
            if t == head_constant:
                head_count += 1
                valid = False
                for ff in data['head_to_triples'][h] & data['tail_to_triples'][body_constant]:
                    if ff.split(',')[1] == body_rel:
                        valid = True
                        break
                supp += valid
        body_count = len(data['relation_to_triples'][body_rel] & data['tail_to_triples'][body_constant])
    elif ',Y)' in Uc_body:
        # tail U_c
        for f in data['relation_to_triples'][head_rel]:
            h, _, t = f.split(',')
            if h == head_constant:
                head_count += 1
                valid = False
                for ff in data['tail_to_triples'][t] & data['head_to_triples'][body_constant]:
                    if ff.split(',')[1] == body_rel:
                        valid = True
                        break
                supp += valid
        body_count = len(data['relation_to_triples'][body_rel] & data['head_to_triples'][body_constant])
    
    return {
        'supp': supp,
        '#body': body_count,
        '#head': head_count,
        'HC': supp / head_count if head_count > 0 else 0,
        'SC': supp / body_count if body_count > 0 else 0
    }
    


def calculate_rule_metrics(head_rel, body, dataset):
    filename = body.replace('/', '_')
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
    body_rels = body.split(',')
    for i in range(1, len(body_rels) + 1):
        head2entity_map[i] = defaultdict(set) # hop -> entity_map (head -> entity set)

    # 计算 body 匹配数和 support
    rel = body_rels[0]
    for triple in train_rel_to_triples[rel]:
        h, _, t = triple.split(',')
        head2entity_map[1][h].add(t)

    ix = 1
    for rel in body_rels[1:]:
        print(f"[calculate_rule_metrics] {head_rel} <= {body_rels}: Processing {rel}")
        ix += 1
        endpoint_entity_set = set()
        for k, v in head2entity_map[ix - 1].items():
            endpoint_entity_set.update(v)
        print('endpoint_entity_set', len(endpoint_entity_set))
        
        for triple in train_rel_to_triples[rel]:
            h, _, t = triple.split(',')
            if h in endpoint_entity_set:
                for k, v in head2entity_map[ix - 1].items():
                    if h in v:
                        head2entity_map[ix][k].add(t)

    # 计算 head count
    head_count = len(train_rel_to_triples[head_rel])
    body_count = 0
    supp = 0

    for k, v in head2entity_map[len(body_rels)].items():
        body_count += len(v)
        tails = [t.split(',')[2] for t in train_rel_to_triples[head_rel] if t.split(',')[0] == k]
        supp += len(v.intersection(tails))

    # 计算 SC 和 HC
    ret = {
        'supp': supp,
        '#body': body_count,
        '#head': head_count,
        'HC': supp / head_count if head_count > 0 else 0,
        'SC': supp / body_count if body_count > 0 else 0
    }

    print(f"Time: {time.time() - t}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ret, f)

    return ret


def find_paths_bfs(head_to_triples, tail_to_triples, head_id, tail_id, max_length=3):
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
            ret = json.load(f)
            if len(ret['paths']) >= 20:
                return ret

        
    data = read_triples(dataset)
    head_to_triples = data['head_to_triples']
    tail_to_triples = data['tail_to_triples']
    head, _, tail = prediction.split(',')
    
    # 使用BFS查找路径
    max_length = 3
    while True:
        paths = find_paths_bfs(head_to_triples, tail_to_triples, head, tail, max_length=max_length)
        print(f'[{prediction}]', 'max_length: ', max_length, 'path count: ', len(paths))
        max_length += 1 
        # 注意这里修改了最低限制，避免搜索长度太短！ WN18需要重新跑
        if len(paths) >= 20 or max_length > 5:
            break

    # Collect unique triples used in paths
    triples_map = {}
    paths_with_triples = []
    relation_path_map = defaultdict(list)
    
    for path in paths:
        path_indices = []
        path_vector = [head]
        current_id = head

        for triple in path:
            # 用一个元组表示triple，避免使用不可哈希的字典
            if triple not in triples_map:
                l = len(triples_map)
                triples_map[triple] = l
            path_indices.append(triples_map[triple])

            head_id, relation_id, tail_id = triple.split(',')
            if head_id == current_id:
                current_id = tail_id
            else:
                current_id = head_id
                relation_id = f"{relation_id}'" # 用'代表反向关系
            path_vector.append(relation_id)
            path_vector.append(current_id)

        relation_path = path_vector[1::2]
        relation_path_map[','.join(relation_path)].append(len(paths_with_triples))
        paths_with_triples.append(path_indices)


    ret = {
        "prediction": prediction,
        "triples": [t[0] for t in sorted(triples_map.items(), key=lambda x: x[1])],
        "paths": paths_with_triples,
        "relation_path_map": relation_path_map
    }
    if condense:
        ret.update(condense_graph(ret))

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ret, f)
    return ret


# def search_ht_relation(prediction, dataset):
#     ret = search_subgraph(prediction, dataset)
#     head, _, tail = prediction.split(',')
#     head_relation2triples = defaultdict(list)    
#     tail_relation2triples = defaultdict(list)
#     for path in ret['paths']:
#         path_vector = [head]
#         current_id = head
#         for ix in path:
#             head_id, relation_id, tail_id = ret['triples'][ix].split(',')
#             if head_id == current_id:
#                 current_id = tail_id
#             else:
#                 current_id = head_id
#                 relation_id = f"{relation_id}'"
#             path_vector.append(relation_id)
#             path_vector.append(current_id)
#         assert current_id == tail
#         head_relation2triples[path_vector[1]].append(ret['triples'][path[0]])
#         tail_relation2triples[path_vector[-2]].append(ret['triples'][path[-1]])
#     return {
#         'head_relation2triples': head_relation2triples,
#         'tail_relation2triples': tail_relation2triples
#     }



def search_ht_relation(prediction, dataset):
    print('search_ht_relation', prediction, dataset)
    ret = read_triples(dataset)
    head_to_triples, tail_to_triples = ret['head_to_triples'], ret['tail_to_triples']
    print('head_to_triples', len(head_to_triples), 'tail_to_triples', len(tail_to_triples))
    head, _, tail = prediction.split(',')
    head_relation2triples = defaultdict(list)    
    tail_relation2triples = defaultdict(list)

    # 不一定是head_to_triple，这样只找到以head开头的，但是以head结尾的也需要！
    for triple in head_to_triples.get(head, set()):
        head_relation2triples[triple.split(',')[1]].append(triple)
    for triple in tail_to_triples.get(head, set()):
        head_relation2triples[triple.split(',')[1] + "'"].append(triple)
    for triple in tail_to_triples.get(tail, set()):
        tail_relation2triples[triple.split(',')[1]].append(triple)
    for triple in head_to_triples.get(tail, set()):
        tail_relation2triples[triple.split(',')[1] + "'"].append(triple)

    print('head_triples: forward/backward', len(head_to_triples.get(head, set())), len(tail_to_triples.get(head, set())))
    print('tail_triples: forward/backward', len(tail_to_triples.get(tail, set())), len(head_to_triples.get(tail, set())))
    print('head_relations', len(head_relation2triples), 'tail_relations', len(tail_relation2triples))
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

def calculateConfidence(ls):
    unconfidence = 1
    for l in ls:
        unconfidence *= 1 - l
    return 1 - unconfidence

if __name__ == '__main__':
    ret = search_subgraph('13112664,_hyponym,12374418',
                    'WN18')
    print(ret)