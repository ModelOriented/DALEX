"""Normalize numeric storage at SHAP boundaries without guessing categories."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import dalex as dx
from dalex.wrappers._shap.object import model_matrix


@pytest.mark.parametrize("storage", ["nullable", "arrow"])
def test_numeric_shap_matrix_preserves_integer_precision_and_float_missing(storage):
    integer = pd.ArrowDtype(pa.int64()) if storage == "arrow" else "Int64"
    floating = pd.ArrowDtype(pa.float32()) if storage == "arrow" else "Float32"
    expected = np.array([2**60 + 1, 2**60 + 3], dtype=np.int64)
    frame = pd.DataFrame({"x": pd.Series(expected, dtype=integer)})
    values = model_matrix(frame, None, "KernelExplainer")
    assert values.dtype == np.dtype("int64")
    np.testing.assert_array_equal(values[:, 0], expected)
    frame = pd.DataFrame({"x": pd.Series([np.float32(.1), None], dtype=floating)})
    values = model_matrix(frame, None, "KernelExplainer")
    assert values.dtype == np.dtype("float32")
    np.testing.assert_array_equal(values[:, 0], np.array([.1, np.nan], dtype=np.float32))


def test_categorical_shap_matrix_remains_object_valued():
    frame = pd.DataFrame({"category": pd.Categorical(["1", "2"]), "x": [1., 2.]})
    values = model_matrix(frame, None, "KernelExplainer")
    assert values.dtype == np.dtype("object")
    assert values[:, 0].tolist() == ["1", "2"]


@pytest.mark.parametrize("storage", ["nullable", "arrow"])
@pytest.mark.parametrize("kind", ["TreeExplainer", "KernelExplainer"])
def test_nullable_pandas_sklearn_shap_backend_parity(storage, kind):
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(219)
    data = pl.DataFrame(rng.normal(size=(64, 3)), schema=["x", "z", "u"]).with_columns(
        pl.col("x").cast(pl.Float32),
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("z")).alias("z"),
    )
    target = data["x"].to_numpy() * 2 + rng.normal(size=64) * .1
    model = HistGradientBoostingRegressor(max_iter=8, random_state=42).fit(data, target)
    standard = data.to_pandas()
    if storage == "arrow":
        nullable = data.to_pandas(use_pyarrow_extension_array=True)
    else:
        nullable = standard.astype({"x": "Float32", "z": "Float64", "u": "Float64"})
    results = []
    for frame in (standard, nullable):
        # Older sklearn versions may not predict directly from pandas ArrowDtype;
        # these external explainers use the explicit numeric matrix boundary.
        exp = dx.Explainer(model, frame, target, verbose=False, precalculate=False, model_type="regression")
        kwargs = {"nsamples": 20, "l1_reg": 0, "silent": True} if kind == "KernelExplainer" else {}
        result = exp.predict_parts(frame.iloc[[0]], type="shap_wrapper", shap_explainer_type=kind,
                                   N=24, random_state=42, **kwargs)
        values = np.asarray(result.result)
        baseline = np.asarray(result.expected_value).item()
        np.testing.assert_allclose(values.sum() + baseline, model.predict(standard.iloc[[0]])[0], atol=1e-6)
        results.append(values)
    np.testing.assert_allclose(*results, rtol=1e-10, atol=1e-10)
