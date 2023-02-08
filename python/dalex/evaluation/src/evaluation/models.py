import pandas as pd
from evaluation.config import RANDOM_STATE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from typing_extensions import Protocol


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...


def create_model(model: str) -> Model:
    if model == "svm":
        return SVR(kernel="linear")
    elif model == "xgboost":
        return GradientBoostingRegressor(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"model {model} not supported")
