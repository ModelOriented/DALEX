"""Automatic grids include observation cells without replacing null by NaN."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx
from dalex._dataframe import take_rows


class MissingStateModel:
    def predict(self, frame):
        values = frame['x'].to_list() if isinstance(frame, pl.DataFrame) else frame['x'].to_pylist()
        return np.array([7. if value is None else 19. if np.isnan(value) else value for value in values]) + 2 * frame['z'].to_numpy()


@pytest.mark.parametrize('backend', ['polars', 'arrow'])
@pytest.mark.parametrize('case', ['null_observation', 'nan_observation', 'all_null_new_value'])
def test_automatic_numeric_grid_includes_actual_native_observation(backend, case):
    original = pl.DataFrame({'x': pl.Series([None] * 4 if case == 'all_null_new_value' else [1., None, np.nan, 3.],
                                           dtype=pl.Float64), 'z': [2., 5., 1., 4.]})
    obs = original.slice(1, 1) if case != 'nan_observation' else original.slice(2, 1)
    if case == 'all_null_new_value':
        obs = obs.with_columns(pl.lit(2.5).alias('x'))
    data = original if backend == 'polars' else original.to_arrow()
    observation = obs if backend == 'polars' else obs.to_arrow()
    model = MissingStateModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    result = exp.predict_profile(observation, variables=['x'], grid_points=3, variable_splits_with_obs=True, verbose=False)
    expected = {'null_observation': [11., 12., 13., 17.], 'nan_observation': [3., 4., 5., 21.],
                'all_null_new_value': [12.5, 17.]}[case]
    np.testing.assert_allclose(result.result['_yhat_'], expected)
    assert model.predict(observation)[0] in result.result['_yhat_'].to_list()


@pytest.mark.parametrize('dtype', ['Int64', 'int64[pyarrow]'])
def test_pandas_nullable_observation_grid_accepts_missing_integer(dtype):
    class NullableModel:
        def predict(self, frame):
            return frame['x'].fillna(7).astype(float).to_numpy() + frame['z'].to_numpy()
    data = pd.DataFrame({'x': pd.Series([1, None, 3], dtype=dtype), 'z': [2., 5., 1.]})
    model = NullableModel()
    exp = dx.Explainer(model, data, model.predict(data), verbose=False)
    result = exp.predict_profile(take_rows(data, [1]), variables=['x'], grid_points=3, verbose=False).result
    np.testing.assert_allclose(result['_yhat_'], [6., 7., 8., 12.])
