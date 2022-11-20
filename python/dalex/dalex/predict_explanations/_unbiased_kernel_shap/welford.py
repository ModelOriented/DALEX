from dataclasses import astuple, dataclass
from typing import Tuple

import numpy as np


@dataclass
class WelfordState:
    """From Welford's algorithm"""

    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def __iter__(self):
        return iter(astuple(self))

    def update(self, observations: np.ndarray) -> None:
        self.count += observations.size
        delta = observations - self.mean
        self.mean += np.sum(delta, axis=0) / self.count
        delta2 = observations - self.mean
        self.M2 += np.sum(delta * delta2)

    @property
    def stats(self) -> Tuple[float, float]:
        """returns mean and std"""
        return self.mean, self.M2 / (self.count - 1)
