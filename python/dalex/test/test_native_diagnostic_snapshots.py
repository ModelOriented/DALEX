"""Results remain snapshots when callers mutate input frames or cached vectors."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import narwhals.stable.v2 as nw
import dalex as dx
from dalex._dataframe import column_values


class SnapshotModel:
    def predict(self, frame):
        if isinstance(frame, np.ndarray):
            return frame[:, 0] * 2 + frame[:, 1]
        return column_values(frame, 'x') * 2 + column_values(frame, 'z')


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('method', ['model_performance', 'model_diagnostics'])
def test_native_diagnostic_result_snapshot_and_pandas_index(backend, method):
    source = pd.DataFrame({'x': [1., 2., 3.], 'z': [3., 2., 1.]}, index=pd.Index([20, 5, 99], name='row'))
    original = pl.from_pandas(source)
    frame = source if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    model = SnapshotModel()
    y = model.predict(frame) + 1
    exp = dx.Explainer(model, frame, y.copy(), verbose=False)
    result = getattr(exp, method)()
    expected = model.predict(frame).copy()
    exp.y_hat[0] = 999.
    exp.y[0] = 777.
    if backend == 'pandas':
        frame.iloc[0, 0] = 888.
    elif backend == 'polars':
        frame[0, 'x'] = 888.
    table = result.result_as(table='residuals') if method == 'model_performance' else result.result_as()
    np.testing.assert_allclose(nw.from_native(table)['y_hat'].to_numpy(), expected)
    if method == 'model_diagnostics':
        np.testing.assert_allclose(nw.from_native(table)['x'].to_numpy(), [1., 2., 3.])
        if backend == 'pandas':
            pd.testing.assert_index_equal(result.result.index, source.index)
        assert list(result.result.columns) == ['x', 'z', 'y', 'y_hat', 'residuals', 'abs_residuals', 'label', 'ids']
    else:
        assert list(result.result.index) == ['SnapshotModel']


@pytest.mark.parametrize('backend', ['polars', 'arrow'])
def test_native_diagnostics_retain_exact_nullable_large_integer_features(backend):
    class Model:
        def predict(self, frame):
            return column_values(frame, 'z')
    original = pl.DataFrame({'x': pl.Series([2**63 + 17, None, 2**63 + 19], dtype=pl.UInt64),
                              'z': [1., 2., 4.]})
    frame = original if backend == 'polars' else original.to_arrow()
    exp = dx.Explainer(Model(), frame, np.array([2., 3., 5.]), verbose=False, model_type='regression')
    result = exp.model_diagnostics()
    table = nw.from_native(result.result_as())
    assert table['x'].to_list() == [2**63 + 17, None, 2**63 + 19]
    assert table.schema['x'] == nw.UInt64
