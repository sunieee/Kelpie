import math
import time
from typing import Tuple, Any
import numpy
import torch

from dataset import Dataset
from kelpie_dataset import KelpieDataset
from relevance_engines.engine import ExplanationEngine
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE
from link_prediction.models.transe import TransE
from link_prediction.models.tucker import TuckER
from link_prediction.optimization.bce_optimizer import KelpieBCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import KelpieMultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import KelpiePairwiseRankingOptimizer
from link_prediction.models.model import *
from utils import *
from collections import OrderedDict
import numpy as np

class PostTrainingEngine(ExplanationEngine):

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        """
            PostTrainingEngine constructor.

            :param model: the trained Model to explain the behaviour of. This can NOT be a KelpieModel.
            :param dataset: the Dataset used to train the model
            :param hyperparameters: dict containing all the hyperparameters necessary for running the post-training
                                    (for both the model and the optimizer)
        """

        ExplanationEngine.__init__(self, model=model, dataset=dataset, hyperparameters=hyperparameters)

        if isinstance(self.model, ComplEx):
            self.kelpie_optimizer_class = KelpieMultiClassNLLOptimizer
        elif isinstance(self.model, ConvE):
            self.kelpie_optimizer_class = KelpieBCEOptimizer
        elif isinstance(self.model, TransE):
            self.kelpie_optimizer_class = KelpiePairwiseRankingOptimizer
        else:
            self.kelpie_optimizer_class = KelpieMultiClassNLLOptimizer

        if isinstance(model, KelpieModel):
            raise Exception("The model passed to the PostTrainingEngine is already a post-trainable KelpieModel.")

        # these data structures are used store permanently, for any fact:
        #   - the score
        #   - the score obtained by the best scoring tail (in "head" perspective) or head (in "tail" perspective)
        #   - the rank obtained by the target tail (in "head" perspective) or head (in "tail" perspective) score)
        self._original_model_results = {}  # map original samples to scores and ranks from the original model
        self._base_pt_model_results = {}   # map original samples to scores and ranks from the base post-trained model
        self._base_cache_embeddings = {} # map embeddings of base node to its embedding 

        # The kelpie_cache is a simple LRU cache that allows reuse of KelpieDatasets and of base post-training results
        # without need to re-build them from scratch every time.
        self._kelpie_dataset_cache_size = kelpie_dataset_cache_size
        self._kelpie_dataset_cache = OrderedDict()
        self.print_count = 0
        self.origin = None
        self.kelpie_dataset = None
    
    def get_kelpie_dataset(self, sample_to_explain: Triple, perspective: str):
        # get from the cache a Kelpie Dataset focused on the original id of the entity to explain,
        # (or create it from scratch if it is not in cache)
        if args.relation_path:
            return self._get_kelpie_dataset_for(entity_ids=[sample_to_explain.h, sample_to_explain.t])
        
        original_entity_to_convert = sample_to_explain.h if perspective == "head" else sample_to_explain.t
        return self._get_kelpie_dataset_for(entity_ids=[original_entity_to_convert])


    def removal_relevance(self,
                           sample_to_explain: Triple,
                           perspective: str,
                           samples_to_remove: List[Path]) -> Score:
        """
            Given a "sample to explain" (that is, a sample that the model currently predicts as true,
            and that we want to be predicted as false);
            given the perspective from which to analyze it;
            and given and a list of training samples containing the entity to convert;
            compute the relevance of the samples in removal, that is, an estimate of the effect they would have
            if removed (all together) from the perspective entity to worsen the prediction of the sample to convert.

            :param sample_to_explain: the sample that we would like the model to predict as "true",
                                      in the form of a tuple (head, relation, tail)
            :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
            :param samples_to_remove:   the list of samples containing the perspective entity
                                        that we want to analyze the effect of, if added to the perspective entity
        """
        print(f'removal relevance of sample: {str(sample_to_explain)}, removing: {str([str(p) for p in samples_to_remove])}')
        start_time = time.time()

        self.origin = sample_to_explain
        self.kelpie_dataset = self.get_kelpie_dataset(sample_to_explain, perspective)

        metrics = {}
        # check how the original model performs on the original sample to convert (no need)
        self._original_model_results[str(self.origin)] = self.origin.origin_score()

        metrics.update(self._original_model_results[str(self.origin)])
        # get from the cache a Kelpie Dataset focused on the original id of the entity to explain
        metrics.update(self.base_post_training_results())
        # run actual post-training by adding the passed samples to the perspective entity and see how it performs in the sample to convert
        metrics.update(self.removal_post_training_results(original_samples_to_remove=samples_to_remove))
        metrics['time'] = rd(time.time() - start_time)

        return Score(samples_to_remove, sample_to_explain, metrics)

    # private methods that know how to access cache structures

    def _get_kelpie_dataset_for(self, entity_ids) -> KelpieDataset:
        """
        Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """

        name = strfy(entity_ids)
        if name not in self._kelpie_dataset_cache:

            kelpie_dataset = KelpieDataset(dataset=self.dataset, entity_ids=entity_ids)
            self._kelpie_dataset_cache[name] = kelpie_dataset
            self._kelpie_dataset_cache.move_to_end(name)

            if len(self._kelpie_dataset_cache) > self._kelpie_dataset_cache_size:
                self._kelpie_dataset_cache.popitem(last=False)

        return self._kelpie_dataset_cache[name]


    def base_post_training_results(self):

        """
        :param kelpie_dataset:
        :param original_sample_to_predict:
        :return:
        """
        if str(self.origin) in self._base_pt_model_results:   # cache
            return self._base_pt_model_results[str(self.origin)]
        
        # kelpie_model: an UNTRAINED kelpie model that has just been initialized
        kelpie_model_class = self.model.kelpie_model_class()
        kelpie_model = kelpie_model_class(model=self.model, dataset=self.kelpie_dataset)

        kelpie_sample = Triple(self.kelpie_dataset.as_kelpie_sample(original_sample=self.origin.triple))

        # kelpie_model.summary('before post_train')
        # base_pt_model = kelpie_model
        base_pt_model = self.post_train(kelpie_model_to_post_train=kelpie_model,
                                        kelpie_train_samples=self.kelpie_dataset.kelpie_train_samples) # type: KelpieModel
        # kelpie_model.summary('after post_train')

        # then check how the base post-trained model performs on the kelpie sample to explain.
        # This means checking how the "clone entity" (with no additional samples) performs
        self._base_pt_model_results[str(self.origin)] = {
            **kelpie_sample.extract_detailed_performances(base_pt_model, 'BB'),
            **kelpie_sample.replace_head(self.origin).extract_detailed_performances(base_pt_model, 'AB'),
            **kelpie_sample.replace_tail(self.origin).extract_detailed_performances(base_pt_model, 'BA'),
        }
        
        return self._base_pt_model_results[str(self.origin)]


    def removal_post_training_results(self, original_samples_to_remove: List[Path]):
        """
        :param kelpie_dataset:
        :param original_sample_to_predict:
        :param original_samples_to_remove:
        :return:
        """
        if args.relation_path:
            print('\tpaths:', original_samples_to_remove)
            tmp = set()
            # 共同路径头/尾
            for p in original_samples_to_remove:    # remove samples connected to head/tail
                tmp.add(p.head.forward().triple)
                tmp.add(p.tail.forward().triple)
            original_samples_to_remove = tmp
        print('\tremoving samples:', original_samples_to_remove)

        # kelpie_model: an UNTRAINED kelpie model that has just been initialized
        kelpie_model_class = self.model.kelpie_model_class()
        kelpie_model = kelpie_model_class(model=self.model, dataset=self.kelpie_dataset)

        kelpie_sample = Triple(self.kelpie_dataset.as_kelpie_sample(original_sample=self.origin.triple))

        # these are original samples, and not "kelpie" samples.
        # the "remove_training_samples" method replaces the original entity with the kelpie entity by itself
        self.kelpie_dataset.remove_training_samples(original_samples_to_remove)

        # post-train a kelpie model on the dataset that has undergone the removal
        # kelpie_model.summary('before post_train')
        # cur_kelpie_model = kelpie_model
        cur_kelpie_model = self.post_train(kelpie_model_to_post_train=kelpie_model,
                                           kelpie_train_samples=self.kelpie_dataset.kelpie_train_samples)  # type: KelpieModel
        # kelpie_model.summary('after post_train')

        # undo the removal, to allow the following iterations of this loop
        self.kelpie_dataset.undo_last_training_samples_removal()

        # checking how the "kelpie entity" (without the removed samples) performs, rather than the original entity
        return {
            **kelpie_sample.extract_detailed_performances(cur_kelpie_model, 'CC'),
            **kelpie_sample.replace_head(self.origin).extract_detailed_performances(cur_kelpie_model, 'AC'),
            **kelpie_sample.replace_tail(self.origin).extract_detailed_performances(cur_kelpie_model, 'CA'),
        }


    # private methods to do stuff
    def post_train(self,
                   kelpie_model_to_post_train: KelpieModel,
                   kelpie_train_samples: numpy.array):
        """

        :param kelpie_model_to_post_train: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_train_samples:
        :return:
        """
        # kelpie_model_class = self.model.kelpie_model_class()
        # kelpie_model = kelpie_model_class(model=self.model, dataset=kelpie_dataset)
        kelpie_model_to_post_train.to('cuda')

        optimizer = self.kelpie_optimizer_class(model=kelpie_model_to_post_train,
                                                hyperparameters=self.hyperparameters,
                                                verbose=False)
        optimizer.epochs = self.hyperparameters[RETRAIN_EPOCHS]
        t = time.time()
        optimizer.train(train_samples=kelpie_train_samples)
        if self.print_count < 5:
            self.print_count += 1
            print(f'\t\t[post_train_time: {rd(time.time() - t)}]')
        return kelpie_model_to_post_train
