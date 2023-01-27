import pandas as pd
from typing_extensions import Protocol

from evaluation.config import MODELS


class Model(Protocol):
    def fit(X: pd.DataFrame, y: pd.Series) -> "Model":
        ...

    def predict(X: pd.DataFrame) -> pd.Series:
        ...


def create_model(model: str) -> Model:
    if model not in MODELS:
        raise ValueError
    # TODO
