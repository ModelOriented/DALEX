"""Native estimator and preprocessing integrations on Polars-origin inputs."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx


@pytest.mark.parametrize("kind", ["sklearn", "lightgbm"])
@pytest.mark.parametrize("classification", [False, True])
def test_native_tree_shap_wrapper(kind, classification):
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from lightgbm import LGBMRegressor, LGBMClassifier

    rng = np.random.default_rng(123)
    data = pl.DataFrame(rng.normal(size=(64, 3)), schema=["x", "z", "u"])
    continuous = 2 * data["x"].to_numpy() - .5 * data["z"].to_numpy()
    y = (continuous > 0).astype(int) if classification else continuous
    if kind == "lightgbm":
        cls = LGBMClassifier if classification else LGBMRegressor
        model = cls(n_estimators=12, n_jobs=1, random_state=42, verbosity=-1).fit(data, y)
    else:
        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        model = cls(max_iter=12, random_state=42).fit(data, y)
    results = []
    for frame in (data.to_pandas(), data, data.to_arrow()):
        def predict(model, batch):
            assert type(batch) is type(frame)
            return model.predict_proba(batch)[:, 1] if classification else model.predict(batch)
        exp = dx.Explainer(model, frame, y, predict_function=predict,
                           model_type="classification" if classification else "regression", verbose=False)
        from dalex._dataframe import take_rows
        obs = take_rows(frame, [0])
        result = exp.predict_parts(obs, type="shap_wrapper", shap_explainer_type="TreeExplainer")
        values = np.asarray(result.result)
        expected = np.asarray(result.expected_value).reshape(-1)
        if values.ndim == 3:
            values = values[..., 1 if values.shape[-1] == 2 else 0]
        if classification:
            raw = model.predict(obs, raw_score=True)[0] if kind == "lightgbm" else model.decision_function(obs)[0]
        else:
            raw = model.predict(obs)[0]
        np.testing.assert_allclose(values.sum() + expected[-1], raw, atol=1e-6, rtol=1e-6)
        result.plot(show=False)
        results.append(values)
    for values in results[1:]:
        np.testing.assert_allclose(results[0], values, atol=1e-10, rtol=1e-10)


def year_features(frame):
    import narwhals as nw
    return nw.from_native(frame).select(nw.col("when").dt.year()).to_numpy()


def numeric_features(frame):
    return frame.to_numpy()


@pytest.mark.parametrize("family", ["importance", "profile", "break_down", "break_down_interactions", "shap", "ceteris_paribus"])
def test_sklearn_native_preprocessing_pipeline(family):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(123)
    data = pl.DataFrame({
        "x": rng.normal(size=64), "group": ["a", "b"] * 32,
        "when": pd.date_range("1990-01-01", periods=64, freq="365D", tz="UTC"),
    })
    y = 2 * data["x"].to_numpy() + np.arange(64) * .1
    transformer = ColumnTransformer([
        # Polars 2 removed the dataframe interchange API used by sklearn 1.7's
        # named-selector path. Positional selectors work on that supported stack
        # too, without converting training or prediction frames to pandas.
        ("number", FunctionTransformer(numeric_features, validate=False), [0]),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1]),
        ("year", FunctionTransformer(year_features, validate=False), [2]),
    ])
    model = make_pipeline(transformer, RandomForestRegressor(n_estimators=8, random_state=42)).fit(data, y)
    results = []
    for frame in (data.to_pandas(), data):
        def predict(model, batch):
            assert type(batch) is type(frame)
            return model.predict(batch)
        exp = dx.Explainer(model, frame, y, predict_function=predict, verbose=False, label="model")
        if family == "importance":
            result = exp.model_parts(N=32, B=3, random_state=42).result
        elif family in ("break_down", "break_down_interactions", "shap"):
            from dalex._dataframe import take_rows
            result = exp.predict_parts(take_rows(frame, [0]), type=family, N=32, B=3, random_state=42).result
        elif family == "ceteris_paribus":
            from dalex._dataframe import take_rows
            result = exp.predict_profile(take_rows(frame, [0, 1]), variables=["x", "group"], grid_points=5, verbose=False).result
        else:
            result = exp.model_profile(N=20, variables=["x"], grid_points=5, random_state=42, verbose=False).result
        results.append(result.select_dtypes(include=np.number))
    pd.testing.assert_frame_equal(*results, check_exact=False, atol=1e-10, rtol=1e-10)
