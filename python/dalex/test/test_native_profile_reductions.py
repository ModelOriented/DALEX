"""Analytical expectations for native profile statistics, without pandas kernels."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import narwhals.stable.v2 as nw
import pytest
from dalex.model_explanations._aggregated_profiles._native import reduce_profiles


def native(values, backend):
    return pd.DataFrame(values) if backend == 'pandas' else pl.DataFrame(values) if backend == 'polars' else pa.table(values)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('kind', ['partial', 'conditional', 'accumulated'])
@pytest.mark.parametrize('grouped', [False, True])
def test_numeric_reductions_match_direct_analytical_values(backend, kind, grouped, monkeypatch):
    grid = np.tile([0., 1., 2.], 3)
    originals = np.repeat([0., 1., 3.], 3)
    offsets = np.repeat([10., 20., 40.], 3)
    frame = native({'_x_': grid, '_original_': originals, '_ids_': np.repeat([0, 1, 2], 3),
                    '_yhat_': grid ** 2 + offsets, 'group': ['a'] * 3 + [None] * 3 + ['a'] * 3}, backend)
    def forbidden(*args, **kwargs):
        raise AssertionError('Profile reduction called pandas conversion')
    with monkeypatch.context() as patch:
        patch.setattr(nw.DataFrame, 'to_pandas', forbidden)
        result = nw.from_native(reduce_profiles(frame, kind, ['group'] if grouped else [], span=.25))
    columns = result.to_dict(as_series=False)
    memberships = [('a', [0, 2]), (None, [1])] if grouped else [(None, [0, 1, 2])]
    for membership, positions in memberships:
        mask = np.array([pd.isna(value) if membership is None else value == membership
                         for value in columns['group']]) if grouped else np.ones(len(result), dtype=bool)
        actual = np.asarray(columns['_yhat_'])[mask]
        expected = []
        for split in [0., 1., 2.]:
            if kind == 'accumulated':
                value = split ** 2
            elif kind == 'partial':
                value = split ** 2 + np.mean(np.array([10., 20., 40.])[positions])
            else:
                weights = np.exp(-.5 * ((np.array([0., 1., 3.])[positions] - split) / .5) ** 2)
                value = split ** 2 + np.average(np.array([10., 20., 40.])[positions], weights=weights)
            expected.append(value)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('kind', ['partial', 'conditional', 'accumulated'])
def test_categorical_weights_and_input_order(backend, kind):
    # The first observed grid order is deliberately not alphabetical.
    frame = native({'_x_': ['b', 'a', 'b', 'a'], '_original_': ['a', 'a', 'b', 'b'],
                    '_ids_': [0, 0, 1, 1], '_yhat_': [12., 10., 22., 20.]}, backend)
    result = nw.from_native(reduce_profiles(frame, kind)).to_dict(as_series=False)
    assert result['_x_'] == ['b', 'a']
    expected = {'partial': [17., 15.], 'conditional': [22., 10.], 'accumulated': [0., -2.]}[kind]
    np.testing.assert_allclose(result['_yhat_'], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_conditional_missing_bin_and_zero_weight_groups(backend):
    frame = native({'_x_': [1., float('nan'), 1., float('nan')],
                    '_original_': [0., 0., float('nan'), float('nan')],
                    '_ids_': [0, 0, 1, 1], '_yhat_': [11., 12., 21., 22.]}, backend)
    result = nw.from_native(reduce_profiles(frame, 'conditional')).to_dict(as_series=False)
    np.testing.assert_allclose(result['_yhat_'], [11., 22.])
    with pytest.raises(ValueError, match='missing value'):
        reduce_profiles(frame, 'accumulated')


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_constant_numeric_grid_has_finite_conditional_weights(backend):
    frame = native({'_x_': [1., 1.], '_original_': [1., 1.], '_ids_': [0, 1], '_yhat_': [10., 20.]}, backend)
    result = nw.from_native(reduce_profiles(frame, 'conditional'))
    np.testing.assert_allclose(result['_yhat_'].to_numpy(), [15.])
