from typing import Tuple, Any
from dataset import Dataset
from link_prediction.models.model import Model
from utils import Triple

class NecessaryExplanationBuilder:
    """
    The NecessaryExplanationBuilder object guides the search for necessary explanations.
    """

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 sample_to_explain: Triple,
                 perspective: str,
                 max_explanation_length: int):
        """
        NecessaryExplanationBuilder object constructor.
        """
        self.model = model
        self.dataset = dataset
        self.sample_to_explain = sample_to_explain

        self.perspective = perspective
        self.perspective_entity = sample_to_explain.h if perspective == "head" else sample_to_explain.t

        self.length_cap = max_explanation_length

    def build_explanations(self,
                           samples_to_add: list,
                           top_k: int = 10):
        pass

    def _average(self, l: list):
        result = 0.0
        for item in l:
            result += float(item)
        return result / float(len(l))
