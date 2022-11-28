from typing_extensions import Protocol

import numpy as np
import pandas as pd


class Explainer(Protocol):
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass

    def residual(self, data: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @property
    def model(self):
        pass

    @property
    def data(self) -> pd.DataFrame:
        pass

    @property
    def label(self) -> str:
        pass
