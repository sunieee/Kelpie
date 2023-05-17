import itertools
import random
from typing import Tuple, Any

from dataset import Dataset
from link_prediction.models.model import *
from utils import *
from explanation_builders.explanation_builder import NecessaryExplanationBuilder
import numpy
import os
from collections import defaultdict

DEAFAULT_XSI_THRESHOLD = 5


class StochasticNecessaryExplanationBuilder(NecessaryExplanationBuilder):
    """
    The StochasticNecessaryExplanationBuilder object guides the search for necessary rules with a probabilistic policy
    """

    def __init__(self, model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Triple,
                 perspective: str,
                 relevance_threshold: float = None,
                 max_explanation_length: int = -1):
        """
        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param sample_to_explain: the predicted sample to explain
        :param perspective: the explanation perspective, either "head" or "tail"
        :param max_explanation_length: the maximum number of facts to include in the explanation to extract
        """

        super().__init__(model=model, dataset=dataset,
                         sample_to_explain=sample_to_explain, perspective=perspective,
                         max_explanation_length=max_explanation_length)

        self.args = dataset.args
        self.xsi = relevance_threshold if relevance_threshold is not None else DEAFAULT_XSI_THRESHOLD
        self.window_size = 10

    def build_explanations(self,
                           samples_to_remove: List[Path],
                           l_max: int = 4) -> List[Explanation]:


        print(f'all possible rules with length 1:', len(samples_to_remove))
        # get relevance for rules with length 1 (that is, samples)
        sample_2_relevance = self.extract_rules_with_length_1(samples_to_remove=samples_to_remove)
        
        # samples_with_relevance = sorted(sample_2_relevance.items(), key=lambda x: x[1], reverse=True)
        # samples_with_relevance = prefilter_negative(sample_2_relevance)
        samples_number = len(sample_2_relevance)
        print('\tvalid rules with length 1: ', samples_number)
        
        # Dict[str, Scores]
        explanations = []
        explanations.extend(sample_2_relevance)

        if len(explanations) == 0:
            print('\tNo valid rules with length 1')
            return []
        
        best_rule_relevance = max([score.max_relevance for score in sample_2_relevance.values()])
        # print(all_rules_with_relevance)
        # print(best_rule_relevance)

        if best_rule_relevance > self.xsi:
            print('\tEarly termination after length 1')
            return explanations

        cur_rule_length = 2

        # stop if you have too few samples (e.g. if you have only 2 samples, you can not extract rules of length 3)
        # or if you get to the length cap
        while cur_rule_length <= samples_number and cur_rule_length <= self.length_cap and cur_rule_length < l_max:
            rule_2_relevance = self.extract_rules_with_length(samples_to_remove=samples_to_remove,
                                                              length=cur_rule_length,
                                                              sample_2_relevance=sample_2_relevance)
            # current_rules_with_relevance = prefilter_negative(rule_2_relevance)
            explanations.extend(rule_2_relevance)

            current_best_rule_relevance = max([score.max_relevance for score in rule_2_relevance.values()])
            if current_best_rule_relevance > best_rule_relevance:
                best_rule_relevance = current_best_rule_relevance
            # else:
            #   break       if searching for additional rules does not seem promising, you should exit now

            if best_rule_relevance > self.xsi:
                break

            cur_rule_length += 1

        # 只去 > 0 的前 k 个
        # all_rules = sorted(all_rules_with_relevance, key=lambda x: x[1], reverse=True)
        # return prefilter_negative(explanations, top_k)
        # for rule, score in score_dic.items():
        #     relevance_df.loc[len(relevance_df)] = {
        #         'triple': self.sample_to_explain,
        #         'explanation': rule,
        #         'head_relevance': score.head_exp.relevance,
        #         'tail_relevance': score.tail_exp.relevance,
        #         'relevance': score.path_exp.relevance,
        #     }

        # explanations = []
        # for score in score_dic.values():
        #     explanations.append(score.head_exp)
        #     explanations.append(score.tail_exp)
        #     explanations.append(score.path_exp)

        explanations.sort(key=lambda x: x.relevance, reverse=True)

        return explanations



    def extract_rules_with_length_1(self, samples_to_remove: List[Path]) -> List[Explanation]:
        explanations = []

        # this is an exception: all samples (= rules with length 1) are tested
        for i, sample_to_remove in enumerate(samples_to_remove):
            # scores = self._compute_relevance_for_rule(([sample_to_remove]))
            exp = Explanation([sample_to_remove], self.sample_to_explain)
            exp.calculate_score()
            if exp.relevance > 0:
                explanations.append(exp)
                print(f"\t{i+1}/{len(samples_to_remove)}: [{sample_to_remove}] {exp.relevance}")
        return explanations

    def extract_rules_with_length(self,
                                  length: int,
                                  sample_2_relevance: List[Explanation]) -> List[Explanation]:

        all_possible_rules = itertools.combinations(sample_2_relevance, length)
        # all_possible_rules_with_preliminary_scores = [(x, 
        #                                             self._preliminary_rule_score(x, sample_2_relevance)) for x in
        #                                             all_possible_rules]
        all_possible_rules_with_preliminary_scores = []
        for rule in all_possible_rules:
            paths = []
            for exp in rule:
                paths.extend(exp.paths)
            all_possible_rules_with_preliminary_scores.append((paths, sum([exp.relevance for exp in rule])))

        explanations = []

        terminate = False
        best_relevance_so_far = -1e6  # initialize with an absurdly low value

        # initialize the relevance window with the proper size
        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        print(f'all possible rules with length {length}:', len(all_possible_rules_with_preliminary_scores))
        while i < len(all_possible_rules_with_preliminary_scores) and not terminate:

            current_rule, current_preliminary_score = all_possible_rules_with_preliminary_scores[i]
            exp = Explanation(current_rule, self.sample_to_explain)
            exp.calculate_score()
            
            if exp.relevance <= 0:
                continue
            explanations.append(exp)

            print(f"\trule: [{current_rule}] {exp.relevance} ({exp.min_relevance}~{exp.max_relevance}))")

            first_relevance = exp.relevance
            # put the obtained relevance in the window
            sliding_window[i % self.window_size] = first_relevance

            # for t in range(3):
            #     prelimentary_df.loc[len(prelimentary_df)] = {
            #         'explanation': current_rule,
            #         'prelimentary': current_preliminary_score[t],
            #         'true': current_rule_relevance[t],
            #         'type_ix': t
            #     }

            # early termination
            if first_relevance > self.xsi:
                i += 1
                print(f"\tEarly Terminate: {first_relevance} > {self.xsi}")
                terminate_at(length, i)
                return explanations

            # else, if the current relevance value is an improvement over the best relevance value seen so far, continue
            elif first_relevance >= best_relevance_so_far:
                best_relevance_so_far = first_relevance
                i += 1
                continue

            # else, if the window has not been filled yet, continue
            elif i < self.window_size:
                i += 1
                continue

            # else, use the average of the relevances in the window to assess the termination condition
            else:
                cur_avg_window_relevance = numpy.mean(sliding_window)
                terminate = random.random() > cur_avg_window_relevance / best_relevance_so_far  
                # termination condition
                i += 1

                # rewrite the upper code without iteration
                print(f"\tCurrent: {first_relevance}, Current window: {cur_avg_window_relevance}, Max: {best_relevance_so_far}, Terminate threshhold: {cur_avg_window_relevance / best_relevance_so_far}")

                if terminate:
                    print("Terminate!")
                    terminate_at(length, i)
                else:
                    print()

        return explanations

    # def _compute_relevance_for_rule(self, nple_to_remove: List[Path]):
    #     rule_length = len(nple_to_remove)

    #     # convert the nple to remove into a list
    #     # assert (len(nple_to_remove[0]) == 3)

    #     dic = self.engine.removal_relevance(sample_to_explain=self.sample_to_explain,
    #                                       perspective=self.perspective,
    #                                       samples_to_remove=nple_to_remove)

    #     print('-----------------------------------------', nple_to_remove)
    #     cur_line = ",".join(self.triple_to_explain) + ";" + \
    #                paths2str(self.dataset, nple_to_remove) + ";" \
    #                 + ';'.join([str(x) for x in dic.values()])
        
    #     print(dic.keys())
    #     print(cur_line)

    #     with open(os.path.join(args.output_folder, "output_details_" + str(rule_length) + ".csv"), "a") as output_file:
    #         output_file.writelines([cur_line + "\n"])

    #     return [dic['relevance'], dic['head_relevance'], dic['tail_relevance']]

    # def _preliminary_rule_score(self, rule, sample_2_relevance):
    #     return numpy.sum([get_first(sample_2_relevance[path2str(self.dataset, x)]) for x in rule])
