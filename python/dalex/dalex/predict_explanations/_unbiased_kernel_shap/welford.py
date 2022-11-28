from dataclasses import astuple, dataclass
from typing import Tuple, Union

import numpy as np


@dataclass
class WelfordState:
    """From Welford's algorithm"""

    count: int = 0
    mean: Union[float, np.ndarray] = 0.0
    M2: Union[float, np.ndarray] = 0.0

    def __iter__(self):
        return iter(astuple(self))

    def update(self, observations: np.ndarray) -> None:
        self.count += observations.shape[0]
        delta = observations - self.mean # n x D
        self.mean += np.sum(delta, axis=0) / self.count # D
        delta2 = observations - self.mean # n x D
        self.M2 += np.sum(delta[:, :, None] * delta2[:, None, :], axis=0) # D x D

    @property
    def stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """returns mean and sample covariance matrix"""
        return self.mean, self.M2 / (self.count - 1)
