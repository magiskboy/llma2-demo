import numpy as np
from scipy.spatial import distance
from ._base import DistanceCalculator

class CosineDistance(DistanceCalculator):
    def compute(self, u: list[float], v: list[float]) -> float:
        u = np.array(u).reshape(-1)
        v = np.array(v).reshape(-1)

        try:
            dist = distance.cosine(u, v)
            return dist
        except ValueError:
            return 0