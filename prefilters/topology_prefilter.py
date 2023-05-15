import copy
from multiprocessing.pool import ThreadPool as Pool
from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import *
from prefilters.prefilter import PreFilter
from utils import *

from collections import defaultdict
from config import MAX_PROCESSES
import threading

class TopologyPreFilter(PreFilter):
    """
    The TopologyPreFilter object is a PreFilter that relies on the graph topology
    to extract the most promising samples for an explanation.
    """
    def __init__(self,
                 model: Model,
                 dataset: Dataset):
        """
        PostTrainingPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.max_path_length = 5
        self.entity_id_2_train_samples = defaultdict(list)
        self.threadLock = threading.Lock()
        self.counter = 0
        self.thread_pool = Pool(processes=MAX_PROCESSES)

        for (h, r, t) in dataset.train_samples:
            self.entity_id_2_train_samples[h].append((h, r, t))
            self.entity_id_2_train_samples[t].append((h, r, t))

    def get_paths(self, entity, length, fact_to_explain: tuple) -> Dict[Any, List[Path]]:
        '''
        获取 entity 长度为 length 的路径列表
        '''
        if hasattr(entity, '__iter__') and type(entity) != str:
            raise Exception('list of entities not allowed!')
        
        lis = []
        for s in self.entity_id_2_train_samples[entity]:
            sample = Triple(s) if s[0] == entity else Triple(s).reverse()
            lis.append(Path([sample], fact_to_explain))
            
        for i in range(1, length):
            new_lis = []
            for p in lis:
                for s in self.entity_id_2_train_samples[p.tail.t]:
                    sample = Triple(s) if s[0] == p.tail.t else Triple(s).reverse()
                    if not p.has_entity(sample.t):
                        new_lis.append(p.extend(sample))
            lis = new_lis

        # print('lis', [str(p) for p in lis])
        dic = defaultdict(list)
        for p in lis:
            dic[p.tail.t].append(p)
        return dic

    def top_promising_samples_for(self,
                                  sample_to_explain:Triple,
                                  perspective:str,
                                  top_k: int,
                                  verbose=True) -> List[Path]:

        """
        This method extracts the top k promising samples for interpreting the sample to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
                                - if "head", find the most promising samples featuring the head of the sample to explain
                                - if "tail", find the most promising samples featuring the tail of the sample to explain
        :param top_k: the number of top promising samples to extract.
        :return: the sorted list of the k most promising samples.
        """
        self.counter = 0

        if verbose:
            print("Extracting promising facts for" + str(sample_to_explain))

        head, tail = sample_to_explain.h, sample_to_explain.t
        start_entity, end_entity = (head, tail) if perspective == "head" else (tail, head)

        if args.relation_path:
            rel_count = self.dataset.num_direct_relations
            print('\thead:', head, '; tail:', tail, '; rel count:', rel_count)
            # relation_path = list(self.dataset.all_simple_paths(head, tail))
            relation_path = []
            for length in range(2, 4):
                cnt = len(relation_path)
                half = length // 2
                head_map = self.get_paths(head, half, sample_to_explain)
                tail_map = self.get_paths(tail, length - half, sample_to_explain)
                keys = head_map.keys() & tail_map.keys()

                for key in keys:
                    # print(key, head_map[key], tail_map[key])
                    for p1 in head_map[key]:
                        for p2 in tail_map[key]:
                            relation_path.append(p1 + p2.reverse())
                print(f'\tpath of length {length}: {len(relation_path) - cnt}')

            # l1 = self.get_paths(head, 1, sample_to_explain)[tail]
            # l2 = self.get_paths(head, 2, sample_to_explain)[tail]
            # l3 = self.get_paths(head, 3, sample_to_explain)[tail]

            # print('length 1:', len(l1))
            # print('length 2:', len(l2))
            # print('length 3:', len(l3))

            node_count = defaultdict(int)
            rel_path_count = defaultdict(int)
            inverse_count = defaultdict(int)
            
            def sort_dic(x):
                return dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
            
            for path in relation_path:
                node_count[path.head.t] += 1
                node_count[path.tail.h] += 1
                rel_path_count[path.rel_path] += 1
                inverse_count[path.inverse_rel] += 1
            
            rel_path_count = sort_dic(rel_path_count)
            node_count = sort_dic(node_count)
            inverse_count = sort_dic(inverse_count)
            plot_dics({
                'rel_path_count': rel_path_count,
                'node_count': node_count,
                'inverse_count': inverse_count
            }, 'results/plot')
            # print('rel path', rel_path_count)
            # print('node', node_count)
            print('inverse', inverse_count)

            # pre-filter top relevance 
            # relation_path.sort(key=lambda path: has_inverse_path(path), reverse=True)
            relation_path.sort(key=lambda path: rel_path_count[path.rel_path] * (node_count[path.head.t] + node_count[path.tail.h]))
            relation_path.sort(key=lambda path: len(path))           
            
            # print('relation path:', relation_path)
            if top_k == 0:  # donot filter
                return relation_path
            return relation_path[:top_k]

        samples_featuring_start_entity = self.entity_id_2_train_samples[start_entity]

        sample_to_analyze_2_min_path_length = {}
        sample_to_analyze_2_min_path = {}

        worker_processes_inputs = [(len(samples_featuring_start_entity),
                                   start_entity, end_entity, samples_featuring_start_entity[i], verbose)
                                   for i in range(len(samples_featuring_start_entity))]

        results = self.thread_pool.map(self.analyze_sample, worker_processes_inputs)

        for i in range(len(samples_featuring_start_entity)):
            _, _, _, sample_to_analyze, _ = worker_processes_inputs[i]
            shortest_path_lengh, shortest_path = results[i]

            sample_to_analyze_2_min_path_length[sample_to_analyze] = shortest_path_lengh
            sample_to_analyze_2_min_path[sample_to_analyze] = shortest_path

        results = sorted(sample_to_analyze_2_min_path_length.items(), key=lambda x:x[1])
        results = [x[0] for x in results]

        return results[:top_k]

    def analyze_sample(self, input_data):
        all_samples_number, start_entity, end_entity, sample_to_analyze, verbose = input_data

        with self.threadLock:
            self.counter+=1
            i = self.counter

        if verbose:
            print("\tAnalyzing sample " + str(i) + " on " + str(all_samples_number) + ": " + self.dataset.printable_sample(sample_to_analyze))

        sample_to_analyze_head, sample_to_analyze_relation, sample_to_analyze_tail = sample_to_analyze

        cur_path_length = 1
        next_step_incomplete_paths = []   # each incomplete path is a couple (list of triples in this path, accretion entity)

        # if the sample to analyze is already a path from the start entity to the end entity,
        # then the shortest path length is 1 and you can move directly to the next sample to analyze
        if (sample_to_analyze_head == start_entity and sample_to_analyze_tail == end_entity) or \
                (sample_to_analyze_tail == start_entity and sample_to_analyze_head == end_entity):
            return cur_path_length, \
                   [(sample_to_analyze_head, sample_to_analyze_relation, sample_to_analyze_tail)]


        initial_accretion_entity = sample_to_analyze_tail if sample_to_analyze_head == start_entity else sample_to_analyze_head
        next_step_incomplete_paths.append(([sample_to_analyze], initial_accretion_entity))

        # this set contains the entities seen so far in the search.
        # we want to avoid any loops / unnecessary searches, so it is not allowed for a path
        # to visit an entity that has already been featured by another path
        # (that is, another path that has either same or smaller size!)
        entities_seen_so_far = {start_entity, initial_accretion_entity}

        terminate = False
        while not terminate:
            cur_path_length += 1

            cur_step_incomplete_paths = next_step_incomplete_paths
            next_step_incomplete_paths = []

            #print("\tIncomplete paths of length " + str(cur_path_length - 1) + " to analyze: " + str(len(cur_step_incomplete_paths)))
            #print("\tExpanding them to length: " + str(cur_path_length))
            for (incomplete_path, accretion_entity) in cur_step_incomplete_paths:
                samples_featuring_accretion_entity = self.entity_id_2_train_samples[accretion_entity]

                # print("Current path: " + str(incomplete_path))

                for (cur_head, cur_rel, cur_tail) in samples_featuring_accretion_entity:

                    cur_incomplete_path = copy.deepcopy(incomplete_path)

                    # print("\tCurrent accretion path: " + self.dataset.printable_sample((cur_h, cur_r, cur_t)))
                    if (cur_head == accretion_entity and cur_tail == end_entity) or (cur_tail == accretion_entity and cur_head == end_entity):
                        cur_incomplete_path.append((cur_head, cur_rel, cur_tail))
                        return cur_path_length, cur_incomplete_path

                    # ignore self-loops
                    if cur_head == cur_tail:
                        # print("\t\tMeh, it was just a self-loop!")
                        continue

                    # ignore facts that would re-connect to an entity that is already in this path
                    next_step_accretion_entity = cur_tail if cur_head == accretion_entity else cur_head
                    if next_step_accretion_entity in entities_seen_so_far:
                        # print("\t\tMeh, it led to a loop in this path!")
                        continue

                    cur_incomplete_path.append((cur_head, cur_rel, cur_tail))
                    next_step_incomplete_paths.append((cur_incomplete_path, next_step_accretion_entity))
                    entities_seen_so_far.add(next_step_accretion_entity)
                    # print("\t\tThe search continues")

            if terminate is not True:
                if cur_path_length == self.max_path_length or len(next_step_incomplete_paths) == 0:
                    return 1e6, ["None"]
