"""Exact native grid identities and integer distances above floating precision."""
import numpy as np
import polars as pl
import narwhals.stable.v2 as nw
import pytest
import dalex as dx
from test_native_missing_attributions import ExactIntegerModel
from test_native_missing_attributions import NullNanModel
from test_native_profile_workflows import native_computation_only


@pytest.mark.parametrize('backend', ['polars', 'arrow'])
@pytest.mark.parametrize('kind', ['partial', 'conditional', 'accumulated'])
@pytest.mark.parametrize('grouped', [False, True])
def test_exact_large_unsigned_numeric_profile_keys_and_distances(backend, kind, grouped, monkeypatch):
    base = 2**63
    original = pl.DataFrame({'x': pl.Series([base+1, None, base+3, base+5], dtype=pl.UInt64),
                             'z': [2., 5., 1., 4.], 'group': ['a', 'a', 'b', 'b']})
    data = original if backend == 'polars' else original.to_arrow()
    model = ExactIntegerModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    grid = [base+1, base+3]
    with native_computation_only(monkeypatch, backend):
        result = exp.model_profile(type=kind, variables=['x'], variable_splits={'x': grid}, N=None,
                                    groups=['group'] if grouped else None, center=False, verbose=False)
        native = nw.from_native(result.result_as())
        assert native['_x_'].dtype == nw.UInt64
        assert native['_x_'].unique().sort().to_list() == grid
    values = original['x'].to_list()
    z = original['z'].to_numpy()
    columns = native.to_dict(as_series=False)
    groups = {'a': [0, 1], 'b': [2, 3]} if grouped else {None: [0, 1, 2, 3]}
    for group, positions in groups.items():
        actual = np.asarray(columns['_yhat_'])[[member == group for member in columns['group']]] if grouped else np.asarray(columns['_yhat_'])
        expected = []
        for i, split in enumerate(grid):
            prediction = float(split % 97) + z[positions]
            if kind == 'partial':
                value = prediction.mean()
            elif kind == 'accumulated':
                value = 0. if i == 0 else 2.
            else:
                distances = np.array([np.nan if values[row] is None else float(values[row] - split) for row in positions])
                weights = np.exp(-.5 * (distances/.5)**2)
                good = np.isfinite(weights)
                value = np.average(prediction[good], weights=weights[good])
            expected.append(value)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert set(result.result['_x_']) == set(grid)
    exported = result.result_as('pyarrow', partition='x')
    assert set(exported['_x_'].to_pylist()) == set(grid)


@pytest.mark.parametrize('backend', ['polars', 'arrow'])
def test_mixed_numeric_grid_types_use_native_partitions_without_rounding(backend, monkeypatch):
    base = 2**63
    original = pl.DataFrame({'x': pl.Series([base+1, base+3, base+5, base+7], dtype=pl.UInt64),
                             'z': [2., 5., 1., 4.]})
    data = original if backend == 'polars' else original.to_arrow()
    model = ExactIntegerModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    with native_computation_only(monkeypatch, backend):
        result = exp.model_profile(variables=['x', 'z'], variable_splits={'x': [base+1, base+3], 'z': [.5, 1.5]},
                                    N=None, verbose=False)
        with pytest.raises(ValueError, match='partition'):
            result.result_as()
        assert nw.from_native(result.result_as(partition='x'))['_x_'].to_list() == [base+1, base+3]
    assert len(result.result) == 4
    assert result.result_as('pyarrow', partition='x')['_x_'].to_pylist() == [base+1, base+3]


@pytest.mark.parametrize('backend', ['polars', 'arrow'])
@pytest.mark.parametrize('kind', ['partial', 'conditional', 'accumulated'])
def test_categorical_null_and_nan_levels_remain_distinct(backend, kind, monkeypatch):
    original = pl.DataFrame({'x': [1., None, np.nan, 3.], 'z': [2., 5., 1., 4.]})
    data = original if backend == 'polars' else original.to_arrow()
    model = NullNanModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    with native_computation_only(monkeypatch, backend):
        result = exp.model_profile(type=kind, variable_type='categorical', variables=['x'],
                                    variable_splits={'x': [None, np.nan]}, N=None, center=False, verbose=False)
        native = nw.from_native(result.result_as())
        assert native['_x_'].to_list() == ['<null>', '<nan>']
        expected = {'partial': [13., 25.], 'conditional': [17., 21.], 'accumulated': [0., 12.]}[kind]
        np.testing.assert_allclose(native['_yhat_'].to_numpy(), expected, atol=1e-12, rtol=1e-12)


class MissingGroupModel:
    def predict(self, frame):
        groups = frame['group'].to_list() if isinstance(frame, pl.DataFrame) else frame['group'].to_pylist()
        offsets = np.array([7. if value is None else 19. for value in groups])
        return 2 * frame['x'].to_numpy() + offsets


@pytest.mark.parametrize('backend', ['polars', 'arrow'])
@pytest.mark.parametrize('kind', ['partial', 'conditional', 'accumulated'])
def test_native_null_and_nan_group_memberships_do_not_merge(backend, kind, monkeypatch):
    original = pl.DataFrame({'x': [0., 1., 2., 3.], 'group': [None, np.nan, None, np.nan]})
    data = original if backend == 'polars' else original.to_arrow()
    model = MissingGroupModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    with native_computation_only(monkeypatch, backend):
        result = exp.model_profile(type=kind, variables=['x'], variable_splits={'x': [0., 1.]},
                                    groups=['group'], N=None, center=False, verbose=False)
        native = nw.from_native(result.result_as())
        assert len(native) == 4
        nulls = native.filter(nw.col('group').is_null())['_yhat_'].to_numpy()
        nans = native.filter(nw.col('group').is_nan().fill_null(False))['_yhat_'].to_numpy()
        np.testing.assert_allclose(nulls, [0., 2.] if kind == 'accumulated' else [7., 9.], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(nans, [0., 2.] if kind == 'accumulated' else [19., 21.], atol=1e-12, rtol=1e-12)
    assert set(result.result['_groups_']) == {'None', 'nan'}
