"""Decimal counterfactuals retain decimal values instead of floating coercion."""
from decimal import Decimal
import numpy as np
import polars as pl
import narwhals.stable.v2 as nw
import pytest
import dalex as dx
from dalex._dataframe import column_values
from test_native_profile_workflows import native_computation_only


class DecimalModel:
    def predict(self, data):
        values = column_values(data, 'd', preserve_nulls=True)
        assert all(isinstance(value, Decimal) for value in values)
        return np.asarray([float(value % Decimal(97)) for value in values])


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_decimal_cp_interventions_preserve_exact_values(backend, monkeypatch):
    splits = [Decimal('9007199254.740993'), Decimal('9007199254.740997')]
    original = pl.DataFrame({'d': pl.Series(splits, dtype=pl.Decimal(38, 6))})
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    model = DecimalModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    observation = data.head(1) if backend != 'arrow' else data.slice(0, 1)
    with native_computation_only(monkeypatch, backend):
        cp = exp.predict_profile(observation, variable_splits={'d': splits}, verbose=False)
        result = nw.from_native(cp.result_as(partition='d'))
        assert result['d'].to_list() == splits
        np.testing.assert_array_equal(result['_yhat_'].to_numpy(), [float(value % Decimal(97)) for value in splits])
