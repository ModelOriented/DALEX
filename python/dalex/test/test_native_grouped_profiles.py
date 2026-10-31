"""Observed profile groups include missing memberships, not unused categories."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx
from dalex._dataframe import column_values


class GroupedModel:
    def predict(self, data):
        return column_values(data, 'x') + column_values(data, 'z')


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('categorical', [False, True])
@pytest.mark.parametrize('family', ['partial', 'conditional', 'accumulated'])
def test_grouped_profiles_retain_missing_groups_without_unused_levels(backend, categorical, family):
    original = pl.DataFrame({'x': [0., 1., 2., 3.], 'z': [10., 20., 30., 40.],
                             'group': ['a', None, 'a', None]})
    if categorical:
        original = original.with_columns(pl.col('group').cast(pl.Enum(['a', 'unused'])))
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    model = GroupedModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', label='grouped', verbose=False)
    result = exp.model_profile(type=family, N=None, variables=['x'], groups=['group'],
                               variable_splits={'x': [0., 1., 2.]}, center=False, span=.25, verbose=False).result
    assert len(result) == 6
    assert len(result['_groups_'].unique()) == 2
    assert 'unused' not in result['_groups_'].unique()
    for missing, positions in ((False, [0, 2]), (True, [1, 3])):
        group = result.loc[(result['_groups_'] != 'a') if missing else (result['_groups_'] == 'a')].sort_values('_x_')
        values = []
        for split in [0., 1., 2.]:
            if family == 'accumulated':
                expected = split
            elif family == 'partial':
                expected = split + np.mean(np.array([10., 20., 30., 40.])[positions])
            else:
                weights = np.exp(-.5 * ((np.array([0., 1., 2., 3.])[positions] - split) / .5)**2)
                expected = split + np.average(np.array([10., 20., 30., 40.])[positions], weights=weights)
            values.append(expected)
        np.testing.assert_allclose(group['_yhat_'], values, atol=1e-12, rtol=1e-12)
    if backend == 'pandas':
        pd.testing.assert_frame_equal(data, original.to_pandas())
    else:
        assert data.schema == (original.schema if backend == 'polars' else original.to_arrow().schema)
