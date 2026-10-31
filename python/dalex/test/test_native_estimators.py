"""Models trained on Polars must retain native explanation prediction batches."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx


@pytest.mark.parametrize("estimator", ["sklearn_regression", "sklearn_classification", "lightgbm_regression", "lightgbm_classification"])
@pytest.mark.parametrize("family", ["importance", "breakdown", "shap", "profile", "performance"])
def test_polars_trained_estimator_backend_parity(estimator, family):
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

    rng = np.random.default_rng(123)
    data = pl.DataFrame(rng.normal(size=(80, 3)), schema=["x", "z", "u"])
    continuous = 2 * data["x"].to_numpy() - data["z"].to_numpy() + rng.normal(size=80) * .1
    classifier = estimator.endswith("classification")
    y = (continuous > 0).astype(int) if classifier else continuous
    if estimator.startswith("lightgbm"):
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if classifier else lgb.LGBMRegressor
        model = cls(n_estimators=12, n_jobs=1, random_state=42, verbosity=-1)
    else:
        cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        model = cls(max_iter=12, random_state=42)
    model.fit(data, y)
    original = model.predict_proba if classifier else model.predict
    results = []
    for backend in ("pandas", "polars", "arrow"):
        frame = {"pandas": data.to_pandas, "polars": lambda: data, "arrow": data.to_arrow}[backend]()
        seen = []

        def guarded(batch):
            assert isinstance(batch, type(frame)), type(batch)
            seen.append(batch)
            return original(batch)

        method = "predict_proba" if classifier else "predict"
        setattr(model, method, guarded)
        exp = dx.Explainer(model, frame, y, label="model", verbose=False)
        obs = frame.iloc[[0]] if backend == "pandas" else (frame.head(1) if backend == "polars" else frame.slice(0, 1))
        if family == "importance":
            explanation = exp.model_parts(N=40, B=3, random_state=42, keep_raw_permutations=True)
        elif family in ("breakdown", "shap"):
            explanation = exp.predict_parts(obs, type="break_down" if family == "breakdown" else "shap", N=40, B=3, random_state=42)
        elif family == "profile":
            explanation = exp.model_profile(N=20, variables=["x"], grid_points=4, random_state=42, verbose=False)
        else:
            explanation = exp.model_performance()
        assert exp.data is frame and seen
        results.append(explanation.result.select_dtypes(include=np.number).reset_index(drop=True))
        setattr(model, method, original)
    for result in results[1:]:
        pd.testing.assert_frame_equal(results[0], result, check_exact=False, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("family", ["importance", "break_down", "shap", "partial", "conditional"])
def test_lightgbm_explicit_categorical_encoding_and_missing_values(family):
    from lightgbm import LGBMRegressor

    # LightGBM's native Arrow stream rejects dictionary columns. Encode with
    # the training schema explicitly; never invent category codes inside DALEX.
    labels = ["a", "b", None, "a"] * 20
    original = pl.DataFrame({"category": labels, "value": np.arange(80, dtype=float)})
    data = original.with_columns(
        pl.col("category").cast(pl.Enum(["a", "b"])).to_physical().cast(pl.Int32),
        pl.when(pl.col("value") == 3).then(None).otherwise(pl.col("value")).alias("value"),
    )
    y = np.arange(80, dtype=float) + np.array([label == "b" for label in labels]) * 5
    model = LGBMRegressor(n_estimators=12, n_jobs=1, verbosity=-1, random_state=42).fit(
        data, y, categorical_feature=["category"],
    )
    results = []
    schema = data.schema
    for frame in (data.to_pandas(), data, data.to_arrow()):
        exp = dx.Explainer(model, frame, y, verbose=False, label="model")
        assert exp.data is frame
        if family == "importance":
            result = exp.model_parts(N=40, B=3, random_state=42).result
        elif family in ("break_down", "shap"):
            from dalex._dataframe import take_rows
            result = exp.predict_parts(take_rows(frame, [2]), type=family, N=40, B=3, random_state=42).result
        else:
            result = exp.model_profile(type=family, N=40, variables=["category"], variable_type="categorical",
                                       variable_splits={"category": [0, 1]}, random_state=42, verbose=False).result
            assert np.isfinite(result["_yhat_"]).all() and (result["_yhat_"] != 0).any()
        results.append(result)
    for result in results[1:]:
        pd.testing.assert_frame_equal(results[0], result, check_exact=False, atol=1e-10, rtol=1e-10)
    assert data.schema == schema and original["category"].to_list() == labels


@pytest.mark.parametrize("estimator", ["sklearn", "lightgbm"])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("sample", [None, 32])
def test_shared_native_estimator_seeded_threaded_importance(estimator, backend, sample):
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(123)
    data = pl.DataFrame(rng.normal(size=(64, 3)), schema=["x", "z", "u"])
    y = data["x"].to_numpy() * 2 + rng.normal(size=64) * .1
    model = (LGBMRegressor(n_estimators=8, n_jobs=1, verbosity=-1, random_state=42)
             if estimator == "lightgbm" else HistGradientBoostingRegressor(max_iter=8, random_state=42))
    model.fit(data, y)
    frame = data.to_pandas() if backend == "pandas" else data

    def calculate(seed):
        exp = dx.Explainer(model, frame, y, label="model", precalculate=False, verbose=False)
        return exp.model_parts(N=sample, B=3, random_state=seed).result

    seeds = [7, 31, 42, 123]
    expected = [calculate(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(calculate, seeds))
    for reference, result in zip(expected, actual):
        pd.testing.assert_frame_equal(reference, result)
