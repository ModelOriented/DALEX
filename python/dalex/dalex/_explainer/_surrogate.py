"""Optional sklearn surrogates with backend-neutral performance tables."""
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from .._explanation import ResultTables, CompatibilityTable
from .._dataframe import to_numpy
from narwhals.dependencies import is_pandas_dataframe


def _model_input(data):
    return data if is_pandas_dataframe(data) else to_numpy(data)


class SurrogateTables(ResultTables):
    performance = CompatibilityTable()

    def result_as(self, backend='native', *, table='performance', partition=None):
        return super().result_as(backend, table=table, partition=partition)

    def predict(self, X, *args, **kwargs):
        return super().predict(_model_input(X), *args, **kwargs)


class NativeDecisionTreeRegressor(SurrogateTables, DecisionTreeRegressor):
    _legacy_class_name = 'DecisionTreeRegressor'


class NativeDecisionTreeClassifier(SurrogateTables, DecisionTreeClassifier):
    _legacy_class_name = 'DecisionTreeClassifier'

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(_model_input(X), *args, **kwargs)


class NativeLinearRegression(SurrogateTables, LinearRegression):
    _legacy_class_name = 'LinearRegression'


class NativeLogisticRegression(SurrogateTables, LogisticRegression):
    _legacy_class_name = 'LogisticRegression'

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(_model_input(X), *args, **kwargs)


def predict_surrogate_model(model, data):
    values = to_numpy(data)
    if hasattr(model, 'predict_proba'):
        import numpy as np
        positive = np.flatnonzero(model.classes_ == 1)
        return model.predict_proba(values)[:, positive[0]] if len(positive) else np.zeros(len(values))
    return model.predict(values)
