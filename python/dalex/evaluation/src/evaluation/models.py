import pandas as pd
from typing_extensions import Protocol
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from config import MODELS, RANDOM_STATE


class Model(Protocol):
    def fit(X: pd.DataFrame, y: pd.Series) -> "Model":
        self.fit(X, y)

    def predict(X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.predict(X))


def create_model(model: str) -> Model:
    if model not in MODELS:
        raise ValueError
    elif model == "svm":
        return SVR(kernel='linear')
    elif model == "xgboost":
        return GradientBoostingRegressor(random_state=RANDOM_STATE)
