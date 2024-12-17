from typing import Literal, List, Any
import numpy as np

Vec = List
Val = Any

class KnowledgeBase:
    def __init__(self, dim: int):
        """
        Initialize a knowledge base with a given dimensionality.
        :param dim: the dimensionality of the vectors to be stored
        """
        self.dim = dim
        self.store = []

    def add_item(self, key: Vec, val: Val):
        """
        Store the key-value pair in the knowledge base.
        :param key: key
        :param val: value
        """
        if len(key) != self.dim:
            raise ValueError(f"len of keys must be {self.dim}, was given {len(key)}")
        self.store.append((key, val))

    def retrieve(
        self, key: Vec, metric: Literal['l2', 'cos', 'ip'], k: int = 1
    ) -> List[Val]:
        """
        Retrieve the top k values from the knowledge base given a key and similarity metric.
        :param key: key
        :param metric: Similarity metric to use.
        :param k: Top k similar items to retrieve.
        :return: List of top k similar values.
        """
        # sort the whole persistent structure and return the top k
        similarities = []
        for item_key, val in self.store:
          if metric == 'l2':
              similarities.append((self._sim_euclidean(key, item_key), val))
          elif metric == 'cos':
              similarities.append((self._sim_cosine(key, item_key), val))
          elif metric == 'ip':
              similarities.append((self._sim_inner_product(key, item_key), val))
          else:
              raise ValueError(f"unknown metric {metric}")

        # asc for similarity, desc for distance
        if metric in ['cos', 'ip']:
            similarities.sort(reverse=True, key=lambda x: x[0])
        else:
            similarities.sort(reverse=False, key=lambda x: x[0])

        # return the top k values
        return [val for _, val in similarities[:k]]

    @staticmethod
    def _sim_euclidean(a: Vec, b: Vec) -> float:
        """
        Compute Euclidean (L2) distance between two vectors.
        :param a: Vector a
        :param b: Vector b
        :return: Similarity score
        """
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def _sim_cosine(a: Vec, b: Vec) -> float:
        """
        Compute the cosine similarity between two vectors.
        :param a: Vector a
        :param b: Vector b
        :return: Similarity score
        """
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def _sim_inner_product(a: Vec, b: Vec) -> float:
        """
        Compute the inner product between two vectors.
        :param a: Vector a
        :param b: Vector b
        :return: Similarity score
        """
        return np.dot(np.array(a), np.array(b))
