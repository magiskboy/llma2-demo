from typing import Protocol

class DistanceCalculator(Protocol):
    def compute(self, u: list[float], v: list[float]) -> float:
        raise NotImplementedError()
