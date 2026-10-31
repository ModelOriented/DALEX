"""Importance respects user loss precision and legacy pandas reductions."""
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

import dalex as dx
from test_native_profile_workflows import native_computation_only


class Prediction:
    def __init__(self, backend):
        self.backend = backend

    def __call__(self, model, frame):
        assert type(frame) is self.backend
        return model.predict(nw.from_native(frame).to_numpy())


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_float32_losses_with_nonportable_python_group_labels(backend):
    rng = np.random.default_rng(76)
    values = rng.normal(size=(32, 3))
    target = values @ [2., -1., .5] + .1
    model = LinearRegression().fit(values, target)
    original = pl.DataFrame(values, schema=['x', 'y', 'z'])
    frame = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    exp = dx.Explainer(model, frame, target, predict_function=Prediction(type(frame)), verbose=False)
    value = exp.model_parts(variable_groups={1: ['x'], 2: ['y'], 3: ['z']},
        loss_function=lambda observed, predicted: np.float32(np.mean((observed - predicted)**2)),
        N=16, B=7, random_state=42)
    raw = value.permutation
    assert all(dtype == np.dtype('float32') for dtype in raw.dtypes)
    expected = raw.mean().sort_values()
    actual = value.result.set_index('variable').dropout_loss
    assert actual.dtype == np.dtype('float32')
    assert actual.index.tolist() == expected.index.tolist()
    np.testing.assert_allclose(actual, expected, rtol=8 * np.finfo(np.float32).eps)
    assert exp.data is frame


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['variable_importance', 'ratio', 'difference'])
@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('precision', ['float32', 'mixed', 'nullable'])
def test_loss_precision_and_missing_draws_match_legacy_mean(monkeypatch, record_testsuite_property,
                                                          backend, kind, missing, precision):
    rng = np.random.default_rng(76)
    values = rng.normal(size=(32, 3))
    target = values @ [2., -1., .5] + rng.normal(size=32) * .1
    model = LinearRegression().fit(values, target)
    original = pl.DataFrame(values, schema=['x', 'y', 'z'])
    frame = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()

    def loss(observed, predicted):
        number = np.mean((observed - predicted)**2)
        if missing and observed[0] > 0:
            return np.float32(np.nan)
        if precision == 'nullable' and observed.mean() > target.mean():
            return None
        return np.float32(number) if precision in ('float32', 'nullable') or number < 1 else np.float64(number)

    with native_computation_only(monkeypatch, backend):
        exp = dx.Explainer(model, frame, target, predict_function=Prediction(type(frame)), verbose=False)
        value = exp.model_parts(loss_function=loss, type=kind, N=16, B=7, random_state=42)
        assert '_compatibility_result' not in value.__dict__
        assert '_compatibility_permutation' not in value.__dict__
        if backend != 'pandas' and precision == 'nullable':
            native_raw = nw.from_native(value.result_as(table='permutation'))
            assert any(native_raw[name].null_count() for name in native_raw.columns)
    raw = value.permutation
    expected = raw.mean().sort_values()
    full = expected['_full_model_']
    if kind == 'ratio':
        expected = expected / full
    elif kind == 'difference':
        expected = expected - full
    actual = value.result.set_index('variable').dropout_loss
    left, right = actual.to_numpy(dtype=float), expected.to_numpy(dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    error = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.
    case = f'{backend}-{kind}-{precision}-{missing}'
    record_testsuite_property(f'float32-loss-max-absolute-error-{case}', error)
    record_testsuite_property(f'float32-loss-finite-pairs-{case}', int(finite.sum()))
    assert actual.dtype == expected.dtype
    if precision == 'float32':
        assert all(dtype == np.dtype('float32') for dtype in raw.dtypes)
        assert actual.dtype == np.dtype('float32')
    if missing:
        assert raw.isna().any(axis=None)
    if backend == 'pandas':
        pd.testing.assert_series_equal(actual, expected, check_names=False, check_exact=True)
    else:
        assert actual.index.tolist() == expected.index.tolist()
        # A native Float32 mean can round differently at its accumulator. The
        # published precision remains Float32, with a bound in that precision.
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(),
            rtol=8 * np.finfo(np.float32).eps, atol=np.finfo(np.float32).eps, equal_nan=True)
