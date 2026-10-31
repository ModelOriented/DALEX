"""End-to-end DALEX model/explanation integration tests using a Polars backend."""
import numpy as np
import pandas as pd
import pytest

import dalex as dx

def weighted_rmse(y, prediction, sample_weight=None):
    return np.sqrt(np.average((y-prediction)**2, weights=sample_weight))


import polars as pl


class PolarsOnlyRegressor:
    """Make unexpected Polars-to-pandas conversion a hard test failure."""

    def predict(self, data):
        assert isinstance(data, pl.DataFrame), type(data)
        return data.select((2.0 * pl.col("x") - 0.5 * pl.col("z")).alias("prediction")).to_series().to_numpy()


def test_polars_native_pipeline_through_permutation_importance():
    data = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "z": [7.0, 6.0, 5.0, 4.0, 3.0]})
    y = 2 * data["x"].to_numpy() - 0.5 * data["z"].to_numpy() + 0.01
    model = PolarsOnlyRegressor()
    explainer = dx.Explainer(model, data, y, model_type="regression", verbose=False)

    assert isinstance(explainer.data, pl.DataFrame)
    np.testing.assert_allclose(explainer.predict(data), model.predict(data))
    np.testing.assert_allclose(explainer.residual(data, y), y - model.predict(data))

    result = explainer.model_parts(N=None, B=3, random_state=42, processes=1)
    assert set(result.result["variable"]) == {"x", "z", "_baseline_", "_full_model_"}
    assert np.isfinite(result.result["dropout_loss"]).all()
    assert isinstance(explainer.data, pl.DataFrame)


def test_polars_and_pandas_permutation_parity():
    frame = pd.DataFrame({"x": [1., 2., 3., 4., 5.], "z": [7., 6., 5., 4., 3.]})
    target = 2 * frame["x"].to_numpy() - 0.5 * frame["z"].to_numpy() + 0.01

    class BackendIndependentRegressor:
        def predict(self, data):
            return (data["x"] * 2 - data["z"] / 2).to_numpy()

    model = BackendIndependentRegressor()
    scores = {}
    for backend, data in (("pandas", frame), ("polars", pl.from_pandas(frame))):
        explainer = dx.Explainer(model, data, target, model_type="regression", verbose=False)
        importance = explainer.model_parts(N=None, B=5, random_state=42, processes=1)
        scores[backend] = importance.result.set_index("variable")["dropout_loss"].sort_index()

    np.testing.assert_allclose(scores["pandas"].to_numpy(), scores["polars"].to_numpy(), atol=1e-12)

class StrictRegressor:
    def __init__(self, backend):
        self.backend = backend

    def predict(self, data):
        assert isinstance(data, pd.DataFrame if self.backend == "pandas" else pl.DataFrame)
        return (data["x"] * 2 - data["z"] / 2).to_numpy()


@pytest.mark.parametrize("precalculate", [True, False])
@pytest.mark.parametrize("family", ["performance", "diagnostics", "break_down", "break_down_interactions", "shap", "predict_profile", "partial", "accumulated", "conditional"])
def test_explanation_family_parity(family, precalculate):
    frame = pd.DataFrame({"x": np.arange(12, dtype=float), "z": np.arange(12, dtype=float)[::-1]})
    y = frame.x.to_numpy() * 2 - frame.z.to_numpy() / 2 + np.sin(np.arange(12)) * .1
    results = []
    for backend in ("pandas", "polars"):
        data = frame if backend == "pandas" else pl.from_pandas(frame)
        explainer = dx.Explainer(StrictRegressor(backend), data, y, model_type="regression", label="model", verbose=False, precalculate=precalculate)
        if family == "performance":
            explanation = explainer.model_performance()
        elif family == "diagnostics":
            explanation = explainer.model_diagnostics()
        elif family in ("break_down", "break_down_interactions", "shap"):
            obs = data.iloc[[0]] if backend == "pandas" else data.head(1)
            explanation = explainer.predict_parts(obs, type=family, N=8, B=3, random_state=42, keep_distributions=True)
        elif family == "predict_profile":
            obs = data.iloc[:2] if backend == "pandas" else data.head(2)
            explanation = explainer.predict_profile(obs, grid_points=5, verbose=False)
        else:
            explanation = explainer.model_profile(type=family, N=8, grid_points=5, random_state=42, verbose=False)
        assert type(explainer.data) is type(data)
        results.append(explanation.result.select_dtypes(include=np.number).reset_index(drop=True))
        # Every supported family must also produce a plot without a backend conversion in the predictor.
        explanation.plot(show=False)
    assert results[0].columns.tolist() == results[1].columns.tolist()
    np.testing.assert_allclose(results[0].to_numpy(dtype=float), results[1].to_numpy(dtype=float), atol=1e-12, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("processes", [1, 2])
@pytest.mark.parametrize("kind", ["variable_importance", "ratio", "difference"])
def test_grouped_weighted_sampled_permutation_parity(processes, kind):
    frame = pd.DataFrame({"x": np.arange(12, dtype=float), "z": np.arange(12, dtype=float)[::-1]}, index=np.arange(12)*7+20)
    y = frame.x.to_numpy() * 2 - frame.z.to_numpy() / 2 + .1
    results = []
    for backend in ("pandas", "polars"):
        data = frame if backend == "pandas" else pl.from_pandas(frame)
        explainer = dx.Explainer(StrictRegressor(backend), data, y, weights=np.linspace(1, 2, 12), model_type="regression", verbose=False)
        result = explainer.model_parts(N=8, B=3, processes=processes, random_state=42, type=kind, variable_groups={"both": ["x", "z"]}, keep_raw_permutations=True, loss_function=weighted_rmse)
        results.append(result)
    pd.testing.assert_frame_equal(results[0].result, results[1].result)
    pd.testing.assert_frame_equal(results[0].permutation, results[1].permutation)


class SeriesRegressor:
    def predict(self, data):
        assert isinstance(data, pl.DataFrame)
        return data["x"] * 2 - data["z"] / 2


def test_series_predictions_targets_and_weights():
    data = pl.DataFrame({"x": [1., 2., 3.], "z": [3., 2., 1.]})
    y = pl.Series("y", [1., 2., 3.])
    explainer = dx.Explainer(SeriesRegressor(), data, y, weights=pl.Series("w", [1., 1., 2.]), model_type="regression", verbose=False)
    assert isinstance(explainer.y_hat, np.ndarray)
    np.testing.assert_allclose(explainer.residuals, y.to_numpy()-explainer.predict(data))
    explainer.model_parts(B=2, random_state=1, loss_function=weighted_rmse)


def test_predict_parts_error_keeps_original_data():
    data = pl.DataFrame({"x": [1., 2.], "z": [3., 4.]})
    explainer = dx.Explainer(StrictRegressor("polars"), data, model_type="regression", verbose=False)
    with pytest.raises(ValueError):
        explainer.predict_parts(pl.DataFrame({"bad": [0.]}), N=1)
    assert explainer.data is data

class StrictClassifier:
    def __init__(self, backend):
        self.backend = backend

    def predict_proba(self, data):
        assert isinstance(data, pd.DataFrame if self.backend == "pandas" else pl.DataFrame)
        p = 1 / (1 + np.exp(-data["x"].to_numpy()))
        return np.column_stack([1-p, p])


def test_classification_and_fairness_parity():
    rng = np.random.default_rng(123)
    frame = pd.DataFrame({"x": rng.normal(size=80), "z": rng.normal(size=80)})
    y = (frame.x.to_numpy()+rng.normal(size=80)*.5 > 0).astype(int)
    results = []
    for backend in ("pandas", "polars"):
        data = frame if backend == "pandas" else pl.from_pandas(frame)
        explainer = dx.Explainer(StrictClassifier(backend), data, y, model_type="classification", verbose=False)
        importance = explainer.model_parts(N=50, B=3, random_state=42)
        fairness = explainer.model_fairness(protected=np.tile(["a","b"],40), privileged="a")
        results.append((explainer.model_performance().result, importance.result, fairness.result))
    for a,b in zip(*results):
        np.testing.assert_allclose(a.select_dtypes(include=np.number), b.select_dtypes(include=np.number), atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("family", ["shap", "predict_profile"])
def test_native_multiprocessing(family):
    data = pl.DataFrame({"x": [1., 2., 3., 4.], "z": [4., 3., 2., 1.]})
    explainer = dx.Explainer(StrictRegressor("polars"), data, model_type="regression", verbose=False)
    if family == "shap":
        result = explainer.predict_parts(data.head(1), type="shap", N=None, B=2, processes=2, random_state=42)
        expected = explainer.predict_parts(data.head(1), type="shap", N=None, B=2, processes=2, random_state=42)
    else:
        result = explainer.predict_profile(data.head(2), grid_points=3, processes=2, verbose=False)
        expected = explainer.predict_profile(data.head(2), grid_points=3, processes=1, verbose=False)
    pd.testing.assert_frame_equal(result.result, expected.result)


def test_mixed_schema_profiles_and_native_observation_inputs():
    class MixedPredictor:
        def predict(self, frame):
            assert isinstance(frame, pl.DataFrame)
            assert frame.schema["category"] == pl.Categorical
            return frame["x"].to_numpy() + (frame["category"].cast(pl.String) == "a").to_numpy()
    data = pl.DataFrame({"x": [1., 2., 3.], "category": ["a","b","a"]}).with_columns(pl.col("category").cast(pl.Categorical))
    explainer = dx.Explainer(MixedPredictor(), data, np.arange(3.), model_type="regression", verbose=False)
    profiles = explainer.predict_profile(data.head(1), grid_points=3, verbose=False)
    assert set(profiles.result._vname_) == {"x","category"}
    explainer.predict_parts(data.head(1), N=None)


def test_numpy_observation_uses_reference_backend():
    data = pl.DataFrame({"x": [1., 2., 3.], "z": [3., 2., 1.]})
    explainer = dx.Explainer(StrictRegressor("polars"), data, model_type="regression", verbose=False)
    a = explainer.predict_parts(np.array([1., 3.]), N=None)
    b = explainer.predict_parts(data.head(1), N=None)
    pd.testing.assert_frame_equal(a.result, b.result)


def native_residual(model, data, y):
    assert isinstance(data, pl.DataFrame)
    return pl.Series(y)-model.predict(data)


def test_native_residual_callback_and_serialization():
    data = pl.DataFrame({"x": [1.,2.,3.], "z": [3.,2.,1.]})
    y = pl.Series("y", [1.,2.,3.])
    explainer = dx.Explainer(SeriesRegressor(), data, y, residual_function=native_residual, model_type="regression", verbose=False)
    assert isinstance(explainer.residuals, np.ndarray)
    np.testing.assert_allclose(explainer.residual(data,y),explainer.residuals)
    restored = dx.Explainer.loads(explainer.dumps())
    assert isinstance(restored.data, pl.DataFrame)
    np.testing.assert_allclose(restored.predict(data), explainer.predict(data))
    np.testing.assert_allclose(restored.residual(data,y), explainer.residuals)
