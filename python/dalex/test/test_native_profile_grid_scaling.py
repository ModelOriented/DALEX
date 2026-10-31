"""Automatic reference grids avoid exporting/scanning every Python scalar."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from dalex.predict_explanations._ceteris_paribus import utils


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('strategy', ['uniform', 'quantiles'])
def test_reference_grid_filters_missing_values_without_python_cell_export(backend, strategy, monkeypatch):
    original = pl.DataFrame({'x': [1., None, np.nan, 2., 5.]})
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    def forbidden(*args, **kwargs):
        raise AssertionError('Grid construction exported all reference cells to Python')
    monkeypatch.setattr(utils, 'column_values', forbidden)
    result = utils.calculate_variable_split(data, ['x'], 5, strategy)
    np.testing.assert_array_equal(result['x'], [1., 2., 3., 4., 5.] if strategy == 'uniform'
                                  else [1., 1.5, 2., 3.5, 5.])


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('nullable', [False, True])
def test_automatic_integer_grids_retain_legacy_interpolation(backend, nullable):
    original = pl.DataFrame({'x': [1, None if nullable else 3, 5]})
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    result = utils.calculate_variable_split(data, ['x'], 4)
    np.testing.assert_array_equal(result['x'], np.linspace(1, 5, 4))
