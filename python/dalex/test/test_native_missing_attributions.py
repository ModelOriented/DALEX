"""Native floating nulls and NaNs are distinct model inputs."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import dalex as dx
from dalex._dataframe import take_rows


class NullNanModel:
    def predict(self, data):
        assert isinstance(data, (pl.DataFrame, pa.Table))
        x = data["x"].to_list() if isinstance(data, pl.DataFrame) else data["x"].to_pylist()
        values = [7. if value is None else 19. if np.isnan(value) else value for value in x]
        z = data["z"].to_numpy()
        return np.array(values) + 2 * z


class ExactIntegerModel:
    def predict(self, data):
        values = data["x"].to_list() if isinstance(data, (pl.DataFrame, pd.DataFrame)) else data["x"].to_pylist()
        return np.array([7. if value is None or value is pd.NA else float(value % 97) for value in values]) + data["z"].to_numpy()


@pytest.mark.parametrize("backend", ["polars", "arrow"])
@pytest.mark.parametrize("row", [1, 2])
@pytest.mark.parametrize("family", ["break_down", "break_down_interactions", "shap", "aspect_default", "aspect_shap"])
def test_attributions_keep_native_null_distinct_from_nan(backend, row, family):
    data = pl.DataFrame({"x": [1., None, np.nan, 3.] * 4,
                         "z": [2., 5., 1., 4.] * 4})
    data = data if backend == "polars" else data.to_arrow()
    model = NullNanModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    observation = take_rows(data, [row])
    before = data.clone() if backend == "polars" else (
        data.schema, data["x"].is_null().to_pylist(), data["x"].to_numpy().copy(), data["z"].to_numpy().copy())
    if family.startswith("aspect_"):
        from dalex.aspect import PredictAspectImportance
        # Sample values and binary design are independently reconstructed to
        # check both attribution algorithms against the actual model contract.
        from dalex.aspect._predict_aspect_importance.utils import get_sample
        seed, n = 42, len(data)
        ids = np.random.RandomState(seed).randint(0, len(data), n)
        sample = take_rows(data, ids)
        base_x = model.predict(sample) - 2 * sample["z"].to_numpy()
        target_x = 7. if row == 1 else 19.
        if family == "aspect_default":
            design = get_sample(n, 2, random_state=seed)
            delta = design[:, 0] * (target_x - base_x) + design[:, 1] * (
                2 * observation["z"].to_numpy()[0] - 2 * sample["z"].to_numpy())
            from sklearn.linear_model import LinearRegression
            expected = LinearRegression().fit(design, delta).coef_[0]
        else:
            expected = target_x - base_x.mean()
        explanation = PredictAspectImportance(variable_groups={"x": ["x"], "z": ["z"]},
                                              type=family.split("_")[1], N=n, B=3, random_state=seed)
        explanation.fit(exp, observation)
        actual = explanation.result.set_index("aspect_name").loc["x", "importance"]
    else:
        explanation = exp.predict_parts(observation, type=family, N=None, B=3, random_state=42)
        result = explanation.result
        if family == "shap":
            actual = result.loc[(result.variable_name == "x") & (result.B == 0), "contribution"].item()
        else:
            actual = result.loc[result.variable_name == "x", "contribution"].item()
        expected = (7. if row == 1 else 19.) - np.mean([1., 7., 19., 3.])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    if backend == "polars":
        assert data.equals(before)
    else:
        assert data.schema.equals(before[0], check_metadata=True)
        assert data["x"].is_null().to_pylist() == before[1]
        np.testing.assert_array_equal(data["x"].to_numpy(), before[2])
        np.testing.assert_array_equal(data["z"].to_numpy(), before[3])


@pytest.mark.parametrize("backend", ["pandas", "polars", "arrow"])
@pytest.mark.parametrize("row", [0, 1])
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("family", ["break_down", "shap", "aspect_default", "aspect_shap"])
def test_nullable_large_integer_attributions_do_not_round_cells(backend, row, unsigned, family):
    base = 2**63 if unsigned else 2**60
    data = pl.DataFrame({"x": pl.Series([base + 1, None, base + 3, base + 5] * 4,
                                       dtype=pl.UInt64 if unsigned else pl.Int64),
                         "z": [2., 5., 1., 4.] * 4})
    data = data.to_pandas(use_pyarrow_extension_array=True) if backend == 'pandas' else data if backend == "polars" else data.to_arrow()
    model = ExactIntegerModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    observation = take_rows(data, [row])
    target = model.predict(observation)[0] - observation["z"].to_numpy()[0]
    if family.startswith("aspect_"):
        from dalex.aspect import PredictAspectImportance
        from dalex.aspect._predict_aspect_importance.utils import get_sample
        ids = np.random.RandomState(42).randint(0, len(data), len(data))
        sample = take_rows(data, ids)
        base_x = model.predict(sample) - sample["z"].to_numpy()
        if family == "aspect_default":
            design = get_sample(len(data), 2, random_state=42)
            delta = design[:, 0] * (target - base_x) + design[:, 1] * (
                observation["z"].to_numpy()[0] - sample["z"].to_numpy())
            from sklearn.linear_model import LinearRegression
            expected = LinearRegression().fit(design, delta).coef_[0]
        else:
            expected = target - base_x.mean()
        result = PredictAspectImportance(variable_groups={"x": ["x"], "z": ["z"]},
                                         type=family.split("_")[1], N=len(data), B=3, random_state=42)
        result.fit(exp, observation)
        actual = result.result.set_index("aspect_name").loc["x", "importance"]
    else:
        result = exp.predict_parts(observation, type=family, N=None, B=3, random_state=42).result
        mask = result.variable_name == "x"
        if family == "shap":
            mask &= result.B == 0
        actual = result.loc[mask, "contribution"].item()
        expected = target - np.mean([float((base + 1) % 97), 7., float((base + 3) % 97), float((base + 5) % 97)])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_multiprocess_shap_preserves_null_observation(backend):
    data = pl.DataFrame({"x": [1., None, np.nan, 3.], "z": [2., 5., 1., 4.]})
    data = data if backend == "polars" else data.to_arrow()
    model = NullNanModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    result = exp.predict_parts(take_rows(data, [1]), type="shap", N=None, B=2,
                               processes=2, random_state=42).result
    values = result.loc[(result.variable_name == "x") & (result.B == 0), "contribution"]
    np.testing.assert_allclose(values, [-.5], atol=1e-12)


@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_numeric_profile_explicit_null_and_nan_are_distinct_interventions(backend):
    data = pl.DataFrame({"x": [1., None, np.nan, 3.], "z": [2., 5., 1., 4.]})
    data = data if backend == "polars" else data.to_arrow()
    model = NullNanModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    result = exp.predict_profile(take_rows(data, [0, 1]), variable_splits={"x": [None, np.nan, 2.5]},
                                 verbose=False).result
    np.testing.assert_allclose(result["_yhat_"], [11., 23., 6.5, 17., 29., 12.5])


@pytest.mark.parametrize("backend", ["polars", "arrow"])
@pytest.mark.parametrize("kind", ["null", "nan", "both"])
def test_all_missing_numeric_profile_grid_retains_native_validity(backend, kind):
    values = [None] * 4 if kind == "null" else [np.nan] * 4 if kind == "nan" else [None, np.nan] * 2
    data = pl.DataFrame({"x": pl.Series(values, dtype=pl.Float64), "z": [2., 5., 1., 4.]})
    data = data if backend == "polars" else data.to_arrow()
    model = NullNanModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    result = exp.predict_profile(take_rows(data, [0]), variables=["x"], verbose=False).result
    expected = [11., 23.] if kind == "both" else [11.] if kind == "null" else [23.]
    np.testing.assert_allclose(result["_yhat_"], expected)


@pytest.mark.parametrize("backend", ["polars", "arrow"])
@pytest.mark.parametrize("family", ["partial", "conditional"])
def test_aggregated_profiles_keep_distinct_native_missing_states(backend, family):
    data = pl.DataFrame({"x": [1., None, np.nan, 3.], "z": [2., 5., 1., 4.]})
    data = data if backend == "polars" else data.to_arrow()
    model = NullNanModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    result = exp.model_profile(type=family, N=None, variable_splits={"x": [None, np.nan]}, verbose=False)
    import narwhals.stable.v2 as nw
    native = nw.from_native(result.result_as())
    assert native['_x_kind_'].to_list() == ['null', 'nan']
    assert native['_x_'].null_count() == 1
    assert native['_x_'].is_nan().sum() == 1
    np.testing.assert_allclose(native['_yhat_'].to_numpy(), [13., 25.] if family == 'partial' else [17., 21.])
    assert result.result['_x_'].iloc[0] is None
    assert np.isnan(result.result['_x_'].iloc[1])
    exported = result.result_as('pyarrow')
    assert exported['_x_'].null_count == 1
    assert np.isnan(exported['_x_'][1].as_py())


@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_partial_profile_accepts_one_native_missing_intervention(backend):
    data = pl.DataFrame({"x": [1., None, np.nan, 3.], "z": [2., 5., 1., 4.]})
    data = data if backend == "polars" else data.to_arrow()
    model = NullNanModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    for value, expected in ((None, 13.), (np.nan, 25.)):
        result = exp.model_profile(type="partial", N=None, variable_splits={"x": [value]}, verbose=False).result
        np.testing.assert_allclose(result['_yhat_'], [expected])


@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_combined_cp_objects_keep_null_and_nan_interventions(backend):
    data = pl.DataFrame({"x": [1., None, np.nan, 3.], "z": [2., 5., 1., 4.]})
    data = data if backend == "polars" else data.to_arrow()
    model = NullNanModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    profiles = [exp.predict_profile(take_rows(data, [0]), variable_splits={"x": [value]}, verbose=False)
                for value in (None, np.nan)]
    from dalex.model_explanations import AggregatedProfiles
    result = AggregatedProfiles()
    result.fit(profiles, verbose=False)
    import narwhals.stable.v2 as nw
    native = nw.from_native(result.result_as())
    assert native['_x_kind_'].to_list() == ['null', 'nan']
    np.testing.assert_allclose(native['_yhat_'].to_numpy(), [11., 23.])


@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_uint64_reconstruction_avoids_signed_integer_inference(backend):
    from dalex._dataframe import frame_from_values, set_columns
    data = pl.DataFrame({"x": pl.Series([2**63 + 1, None], dtype=pl.UInt64), "z": [2., 5.]})
    data = data if backend == "polars" else data.to_arrow()
    values = np.empty((2, 2), dtype=object)
    values[:, 0] = [2**63 + 1, None]
    values[:, 1] = [2., 5.]
    for result in (frame_from_values(values, data), set_columns(data, {"x": values[:, 0]})):
        assert result.schema == data.schema
        assert result.equals(data)
