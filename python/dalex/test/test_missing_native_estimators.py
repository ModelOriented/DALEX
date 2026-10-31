"""Models accepting missing features must receive useful grids and native batches."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx
from dalex._dataframe import column_values, set_columns, take_rows


@pytest.mark.parametrize("estimator", ["sklearn", "lightgbm"])
@pytest.mark.parametrize("family", ["importance", "break_down", "shap", "uniform", "quantiles",
                                    "partial_missing", "conditional_missing"])
def test_missing_features_have_cross_backend_estimator_parity(estimator, family):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(317)
    values = rng.normal(size=(80, 3))
    values[::7, 0] = np.nan
    data = pl.DataFrame(values, schema=["x", "z", "u"]).with_columns(
        pl.when(pl.int_range(pl.len()) % 11 == 0).then(None).otherwise(pl.col("z")).alias("z")
    )
    y = np.nan_to_num(values[:, 0]) * 2 - values[:, 1]
    model = (HistGradientBoostingRegressor(max_iter=12, random_state=42) if estimator == "sklearn"
             else LGBMRegressor(n_estimators=12, n_jobs=1, verbosity=-1, random_state=42)).fit(data, y)
    outputs = []
    for frame in (data.to_pandas(), data, data.to_arrow()):
        def predict(model, batch):
            assert type(batch) is type(frame)
            return model.predict(batch)

        exp = dx.Explainer(model, frame, y, predict_function=predict, label="model", verbose=False)
        if family == "importance":
            result = exp.model_parts(N=40, B=3, random_state=42).result
        elif family in ("break_down", "shap"):
            result = exp.predict_parts(take_rows(frame, [0]), type=family, N=40, B=3, random_state=42).result
        elif family.endswith("_missing"):
            kind = family.removesuffix("_missing")
            result = exp.model_profile(type=kind, N=None, variables=["x"],
                                       variable_splits={"x": [-1., np.nan, 1.]}, verbose=False).result
            intervention = set_columns(frame, {"x": np.nan}, scalar=True)
            predictions = model.predict(intervention)
            if kind == "conditional":
                predictions = predictions[np.isnan(column_values(frame, "x").astype(float))]
            missing = result.loc[result["_x_"].isna(), "_yhat_"]
            assert len(missing) == 1
            np.testing.assert_allclose(missing.to_numpy(dtype=float), predictions.mean())
        else:
            result = exp.predict_profile(take_rows(frame, [0, 1]), variables=["x", "z"],
                                         variable_splits_type=family, grid_points=5, verbose=False).result
            for variable, group in result.groupby("_vname_"):
                # The default additionally inserts observation values, including
                # their missing value; the generated finite grid must still exist.
                assert np.isfinite(group[variable]).sum() >= 5
            assert np.isfinite(result["_yhat_"]).all()
        outputs.append(result.select_dtypes(include=np.number).reset_index(drop=True))
    for output in outputs[1:]:
        pd.testing.assert_frame_equal(outputs[0], output, check_exact=False, rtol=1e-10, atol=1e-10)
