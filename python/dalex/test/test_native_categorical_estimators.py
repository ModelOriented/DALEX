"""Explicit categorical masks support older sklearn without dataframe interchange."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx
from dalex._dataframe import take_rows


def normalize_missing_categories(frame):
    import narwhals.stable.v2 as nw
    return nw.from_native(frame).with_columns(
        nw.when(nw.col("kind").is_null()).then(nw.lit("__MISSING__"))
        .otherwise(nw.col("kind").cast(nw.String)).alias("kind")
    # An explicit estimator matrix boundary also supports sklearn 1.6's
    # ColumnTransformer, which cannot select columns from an Arrow Table.
    ).to_numpy()


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("family", ["importance", "break_down", "break_down_interactions", "shap", "profile", "kernel_shap"])
def test_hist_gradient_boosting_native_categorical_features(classification, family):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    rng = np.random.default_rng(617)
    data = pl.DataFrame({"kind": ["a", "b", None, "a"] * 20, "x": rng.normal(size=80)}).with_columns(
        pl.col("kind").cast(pl.Enum(["a", "b", "unused"]))
    )
    continuous = data["x"].to_numpy() + np.array([kind == "b" for kind in data["kind"].to_list()]) * 2
    y = (continuous > 0).astype(int) if classification else continuous
    cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
    # OrdinalEncoder treats None as a learned level but pandas categorical NaN
    # as missing. Explicit native preprocessing makes that model contract equal.
    model = make_pipeline(FunctionTransformer(normalize_missing_categories),
                          cls(max_iter=12, random_state=42, categorical_features=[True, False])).fit(data, y)
    outputs = []
    for frame in (data.to_pandas(), data, data.to_arrow()):
        def predict(model, batch):
            assert type(batch) is type(frame)
            return model.predict_proba(batch)[:, 1] if classification else model.predict(batch)

        exp = dx.Explainer(model, frame, y, predict_function=predict, label="model", verbose=False,
                           model_type="classification" if classification else "regression")
        # Explain the missing category itself, including reconstruction by Kernel SHAP.
        observation = take_rows(frame, [2])
        if family == "importance":
            result = exp.model_parts(N=40, B=3, random_state=42).result
        elif family == "profile":
            result = exp.predict_profile(observation, variables=["kind"], verbose=False).result
            # pandas retains the user index (2); native frames use position 0.
            result = result.drop(columns="_ids_")
        elif family == "kernel_shap":
            explanation = exp.predict_parts(observation, type="shap_wrapper", shap_explainer_type="KernelExplainer",
                                             N=20, random_state=42, nsamples=20, l1_reg=0, silent=True)
            outputs.append(np.asarray(explanation.result))
            np.testing.assert_allclose(np.asarray(explanation.result).sum() + np.asarray(explanation.expected_value).item(),
                                       exp.predict(observation)[0], rtol=1e-8, atol=1e-8)
            continue
        else:
            result = exp.predict_parts(observation, type=family, N=40, B=3, random_state=42).result
        outputs.append(result.select_dtypes(include=np.number).to_numpy(dtype=float))
    for output in outputs[1:]:
        np.testing.assert_allclose(outputs[0], output, rtol=1e-10, atol=1e-10, equal_nan=True)
