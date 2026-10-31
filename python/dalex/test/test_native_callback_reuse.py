"""Reusing Explainer callbacks must retain serialization's local-function policy."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import dalex as dx
from dalex._dataframe import column_values


class SerializableModel:
    def predict(self, batch):
        if isinstance(batch, np.ndarray):
            return 2 * batch[:, 0] + batch[:, 1]
        return 2 * column_values(batch, 'x') + column_values(batch, 'z')


def native_vector(batch, values):
    return pl.Series(values) if isinstance(batch, pl.DataFrame) else pa.array(values) if isinstance(batch, pa.Table) else values


def module_prediction(model, batch):
    return native_vector(batch, model.predict(batch))


def module_residual(model, batch, y):
    return native_vector(batch, 2 * (y - model.predict(batch)))


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('local', ['prediction', 'residual', 'both'])
def test_reused_callbacks_keep_pickle_roundtrip_and_default_restoration(backend, local):
    data = pl.DataFrame({'x': [1., 2., 3.], 'z': [4., 2., 5.]})
    frame = data.to_pandas() if backend == 'pandas' else data if backend == 'polars' else data.to_arrow()
    model = SerializableModel()
    y = model.predict(frame) + 1
    def local_prediction(model, batch):
        return native_vector(batch, model.predict(batch) + 3)
    def local_residual(model, batch, y):
        return native_vector(batch, y - model.predict(batch) + 2)
    first = dx.Explainer(model, frame, y, verbose=False,
                         predict_function=local_prediction if local in ('prediction', 'both') else module_prediction,
                         residual_function=local_residual if local in ('residual', 'both') else module_residual)
    second = dx.Explainer(model, frame, y, predict_function=first.predict_function,
                          residual_function=first.residual_function, verbose=False)
    np.testing.assert_array_equal(second.predict(frame), first.predict(frame))
    np.testing.assert_array_equal(second.residual(frame, y), first.residual(frame, y))
    payload = second.dumps()
    raw = dx.Explainer.loads(payload, use_defaults=False)
    assert (raw.predict_function is None) == (local in ('prediction', 'both'))
    assert (raw.residual_function is None) == (local in ('residual', 'both'))
    restored = dx.Explainer.loads(payload)
    assert type(restored.data) is type(frame)
    np.testing.assert_array_equal(restored.predict(frame), model.predict(frame))
    expected = y - restored.predict(frame) if local in ('residual', 'both') else 2 * (y - model.predict(frame))
    np.testing.assert_array_equal(restored.residual(frame, y), expected)
