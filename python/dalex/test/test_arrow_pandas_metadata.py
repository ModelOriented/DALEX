"""Pandas-origin Arrow hints must not hide fields or undo numeric interventions."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import dalex as dx
from dalex._dataframe import column_names, set_columns, take_rows, to_numpy, to_pandas


class PandasConsumingModel:
    def predict(self, batch):
        assert isinstance(batch, pa.Table)
        frame = batch.to_pandas()
        return frame['x'].fillna(0).astype(float).to_numpy() + 2 * frame['z'].to_numpy()


@pytest.mark.parametrize('physical_index', [False, True])
@pytest.mark.parametrize('family', ['break_down', 'shap', 'kernel_shap', 'partial'])
def test_pandas_origin_arrow_fields_remain_native_features(physical_index, family):
    frame = pd.DataFrame({'x': pd.array([1, None, 4], dtype='Int64'), 'z': [0., 1., 2.]})
    frame.index = pd.Index([7, 9, 20], name='row') if physical_index else pd.RangeIndex(7, 13, 2)
    data = pa.Table.from_pandas(frame)
    schema = data.schema
    matrix = to_numpy(data)
    assert matrix.shape == data.shape
    assert list(to_pandas(data).columns) == data.column_names
    assert list(to_pandas(data).index) == list(range(len(data)))
    if physical_index:
        np.testing.assert_array_equal(matrix[:, 2], [7, 9, 20])
    model = PandasConsumingModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    observation = take_rows(data, [0])
    if family == 'partial':
        result = exp.model_profile(type='partial', N=None, variable_splits={'x': [1.25, 2.5, 3.75]},
                                   verbose=False).result
        np.testing.assert_allclose(result['_yhat_'], [3.25, 4.5, 5.75])
    elif family == 'kernel_shap':
        result = exp.predict_parts(observation, type='shap_wrapper', shap_explainer_type='KernelExplainer',
                                   N=None, random_state=42, nsamples=20, l1_reg=0, silent=True)
        np.testing.assert_allclose(np.asarray(result.result).sum() + np.asarray(result.expected_value).item(), 1.)
    else:
        result = exp.predict_parts(observation, type=family, N=None, B=3, random_state=42)
        if family == 'shap':
            np.testing.assert_allclose(result.result.loc[result.result.B == 0, 'contribution'].sum()
                                       + result.intercept, 1.)
        else:
            np.testing.assert_allclose(result.result.iloc[-1]['cumulative'], 1.)
        assert set(column_names(data)).issubset(set(result.result['variable_name']))
    assert data.schema.equals(schema, check_metadata=True)


def test_arrow_numeric_promotion_updates_only_obsolete_pandas_dtype_hints():
    frame = pd.DataFrame({'x': pd.array([1, None, 4], dtype='Int64'), 'z': [0., 1., 2.]})
    data = pa.Table.from_pandas(frame)
    data = data.replace_schema_metadata({**data.schema.metadata, b'dataset': b'keep'})
    field = data.schema.field('x').with_metadata({b'unit': b'count'})
    data = data.set_column(0, field, data['x'])
    result = set_columns(data, {'x': [1.25, 2.5, 3.75]})
    np.testing.assert_array_equal(result.to_pandas()['x'], [1.25, 2.5, 3.75])
    assert result.schema.metadata[b'dataset'] == b'keep'
    assert result.schema.field('x').metadata == {b'unit': b'count'}
    before, after = [json.loads(table.schema.metadata[b'pandas']) for table in (data, result)]
    assert before['index_columns'] == after['index_columns']
    assert before['columns'][1:] == after['columns'][1:]
    assert after['columns'][0]['numpy_type'] == 'float64'
    assert before['columns'][0]['numpy_type'] == 'Int64'


@pytest.mark.parametrize('dtype', ['int64[pyarrow]', 'uint64[pyarrow]'])
@pytest.mark.parametrize('method', ['pearson', 'spearman'])
def test_nullable_arrow_statistics_keep_model_storage_unchanged(dtype, method):
    from dalex.aspect.utils import calculate_depend_matrix
    data = pd.DataFrame({'x': pd.Series([1, None, 3, 4], dtype=dtype), 'z': [4., 2., 5., 1.]})
    original = data.copy()
    result = calculate_depend_matrix(data, 'assoc', method, 'max')
    expected = pd.DataFrame({'x': [1., np.nan, 3., 4.], 'z': [4., 2., 5., 1.]}).corr(method).abs()
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(data, original)
