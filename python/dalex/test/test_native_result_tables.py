"""Native computation storage and mutable backwards-compatible pandas views."""
import pickle
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import narwhals.stable.v2 as nw
import pytest
import dalex as dx
from dalex._dataframe import column_values
from dalex._explanation import Explanation


class LinearModel:
    def predict(self, data):
        return 2 * column_values(data, 'x') + column_values(data, 'z')


@pytest.mark.parametrize('container', [list, tuple])
def test_nested_non_tabular_shap_exports_are_independent(container):
    value = dx.wrappers.ShapWrapper(type='predict_parts')
    value.result = container([np.array([1., 2.]), np.array([3., 4.])])
    exported = value.result_as()
    exported[0][0] = 99.
    assert value.result[0][0] == 1.


def explainer(backend):
    data = pl.DataFrame({'x': [1., 2., 3., 4.], 'z': [4., 2., 5., 1.]})
    frame = data.to_pandas() if backend == 'pandas' else data if backend == 'polars' else data.to_arrow()
    model = LinearModel()
    return dx.Explainer(model, frame, model.predict(frame), model_type='regression', verbose=False)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['variable_importance', 'ratio', 'difference'])
def test_importance_results_have_native_storage_and_legacy_public_tables(backend, kind):
    exp = explainer(backend)
    result = exp.model_parts(type=kind, N=None, B=3, random_state=42, keep_raw_permutations=True)
    native = result.result_as()
    raw = result.result_as(table='permutation')
    expected_type = {'pandas': pd.DataFrame, 'polars': pl.DataFrame, 'pyarrow': pa.Table}[backend]
    assert isinstance(native, expected_type) and isinstance(raw, expected_type)
    assert '_compatibility_result' not in result.__dict__
    assert '_compatibility_permutation' not in result.__dict__
    np.testing.assert_allclose(nw.from_native(native)['dropout_loss'].to_numpy(),
                               result.result['dropout_loss'].to_numpy(), equal_nan=True)
    assert isinstance(result.result, pd.DataFrame)
    assert isinstance(result.permutation, pd.DataFrame)
    result.result.loc[0, 'dropout_loss'] = 123.
    result.permutation.iloc[0, 0] = 456.
    assert nw.from_native(result.result_as())['dropout_loss'][0] == 123.
    assert nw.from_native(result.result_as(table='permutation')).to_numpy()[0, 0] == 456.
    restored = pickle.loads(pickle.dumps(result))
    pd.testing.assert_frame_equal(restored.result, result.result)
    assert nw.from_native(restored.result_as())['dropout_loss'][0] == 123.
    result.result = result.result.iloc[:1].copy()
    assert len(result.result_as()) == 1


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_native_importance_never_materializes_pandas_results(monkeypatch, backend):
    exp = explainer(backend)
    def forbidden(*args, **kwargs):
        raise AssertionError('native computation materialized pandas')
    with monkeypatch.context() as patch:
        patch.setattr(nw.DataFrame, 'to_pandas', forbidden)
        result = exp.model_parts(N=None, B=3, random_state=42, keep_raw_permutations=True)
        native = result.result_as()
        assert len(nw.from_native(native)) == 4
        assert len(nw.from_native(result.result_as(table='permutation'))) == 3
    assert result.plot(show=False) is not None


@pytest.mark.parametrize('target', ['pandas', 'polars', 'pyarrow'])
def test_explicit_result_exports_are_independent(target):
    result = explainer('polars').model_parts(N=None, B=3, random_state=42)
    exported = result.result_as(target)
    assert nw.from_native(exported).columns == ['variable', 'dropout_loss', 'label']
    if target == 'pandas':
        exported.loc[0, 'dropout_loss'] = 999.
    elif target == 'polars':
        exported[0, 'dropout_loss'] = 999.
    assert result.result['dropout_loss'].iloc[0] != 999.
    with pytest.raises(ValueError, match='backend'):
        result.result_as('invalid')
    with pytest.raises(ValueError, match='table'):
        result.result_as(table='missing')


def test_old_pickle_state_and_non_tabular_results_remain_usable():
    class ExistingExplanation(Explanation):
        def fit(self):
            pass
        def plot(self):
            pass
    result = ExistingExplanation()
    result.__dict__['result'] = pd.DataFrame({'x': [1]}, index=pd.Index([17], name='row'))
    assert result.result.index.name == 'row'
    assert result.result_as('pandas').index[0] == 17
    assert result.result_as('polars').columns == ['x']
    result.result = np.array([1., 2.])
    np.testing.assert_array_equal(result.result_as(), [1., 2.])
    with pytest.raises(TypeError, match='dataframe'):
        result.result_as('polars')


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_nonstring_group_labels_retain_their_legacy_python_types(backend):
    result = explainer(backend).model_parts(variable_groups={1: ['x'], 'z': ['z']},
                                           N=None, B=3, random_state=42, keep_raw_permutations=True)
    assert set(result.result.variable) == {1, 'z', '_full_model_', '_baseline_'}
    assert list(result.permutation.columns) == [1, '_baseline_', '_full_model_', 'z']
