"""Invalid positions and missing feature values must behave across backends."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from dalex._dataframe import column_values, set_columns, shuffle_columns, take_rows


@pytest.fixture(params=["pandas", "polars", "arrow"])
def frame(request):
    data = pl.DataFrame({"x": [1., None, 3.], "z": [4., 5., 6.]})
    return {"pandas": data.to_pandas, "polars": lambda: data, "arrow": data.to_arrow}[request.param]()


@pytest.mark.parametrize("positions", [np.array([2**64 - 1], dtype=np.uint64), [3], [-4]])
def test_gather_rejects_out_of_bounds_without_integer_overflow(frame, positions):
    with pytest.raises(IndexError, match="out of bounds"):
        take_rows(frame, positions)


@pytest.mark.parametrize("positions", [[1.9, 0., 2.], [True, False, True], [[1], [0], [2]]])
def test_shuffle_rejects_noninteger_positions(frame, positions):
    with pytest.raises(TypeError, match="integer sequence"):
        shuffle_columns(frame, "x", positions)


def test_negative_shuffle_positions_are_positional(frame):
    result = shuffle_columns(frame, "z", [-1, 0, -2])
    np.testing.assert_array_equal(column_values(result, "z"), [6., 4., 5.])
    np.testing.assert_array_equal(column_values(frame, "z"), [4., 5., 6.])


def test_empty_shuffle_preserves_schema(frame):
    empty = take_rows(frame, [])
    result = shuffle_columns(empty, "x", [])
    if isinstance(frame, pd.DataFrame):
        pd.testing.assert_frame_equal(result, empty)
    else:
        assert result.schema == empty.schema
        assert result.equals(empty)


@pytest.mark.parametrize("kind", ["uniform", "quantiles"])
def test_profile_grids_ignore_missing_numeric_values(frame, kind):
    from dalex.predict_explanations._ceteris_paribus.utils import calculate_variable_split

    grid = calculate_variable_split(frame, ["x"], 3, kind)["x"]
    np.testing.assert_array_equal(grid, [1., 2., 3.])


def test_null_scalar_preserves_native_feature_type(frame):
    result = set_columns(frame, {"x": None})
    if isinstance(frame, pd.DataFrame):
        assert result.dtypes.equals(frame.dtypes)
        assert result.x.isna().all()
    else:
        assert result.schema == frame.schema
        assert column_values(result, "x").shape == (3,)


@pytest.mark.parametrize("backend", ["pandas", "polars", "arrow"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_nullable_integer_and_boolean_profile_grids(backend, all_missing):
    from dalex.predict_explanations._ceteris_paribus.utils import calculate_variable_split

    data = pl.DataFrame({"x": [None, None, None] if all_missing else [1, None, 3],
                         "flag": [True, None, False]}, schema={"x": pl.Int64, "flag": pl.Boolean})
    data = {"pandas": lambda: data.to_pandas(use_pyarrow_extension_array=True),
            "polars": lambda: data, "arrow": data.to_arrow}[backend]()
    for kind in ("uniform", "quantiles"):
        grids = calculate_variable_split(data, ["x", "flag"], 3, kind)
        expected = ([np.nan] if backend == "pandas" else [None]) if all_missing else [1., 2., 3.]
        np.testing.assert_array_equal(grids["x"], expected)
        assert len(grids["flag"]) == 3
        assert any(value is True or value == np.bool_(True) for value in grids["flag"] if not pd.isna(value))


@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_scalar_and_matrix_restore_preserve_temporal_and_nested_values(backend):
    from datetime import date
    from decimal import Decimal
    from dalex._dataframe import frame_from_values

    stamp = pd.Timestamp("2025-01-01 00:00:00.123456789", tz="America/Sao_Paulo")
    data = pl.DataFrame({
        "when": pl.Series([stamp.value, stamp.value + 1], dtype=pl.Int64).cast(pl.Datetime("ns", "America/Sao_Paulo")),
        "day": [date(2025, 1, 1), date(2025, 1, 2)],
        "amount": pl.Series([Decimal("1.25"), Decimal("2.50")], dtype=pl.Decimal(20, 2)),
        "list": [[1, 2], [3]], "record": [{"a": 1}, {"a": 2}],
    })
    reference = data if backend == "polars" else data.to_arrow()
    observation = data.to_pandas().iloc[[0]]
    restored = frame_from_values(observation.to_numpy(), reference)
    expected = take_rows(reference, [0])
    assert restored.schema == expected.schema, (str(restored.schema), str(expected.schema))
    assert restored.equals(expected)
    replaced = set_columns(reference, observation.iloc[0].to_dict(), scalar=True)
    expected = take_rows(reference, [0, 0])
    assert replaced.schema == expected.schema
    assert replaced.equals(expected)


def test_arrow_field_metadata_survives_replacement_and_matrix_bridge():
    from dalex._dataframe import frame_from_values

    schema = pa.schema([
        pa.field("x", pa.int64(), metadata={b"unit": b"count"}),
        pa.field("items", pa.large_list(pa.int64()), metadata={b"role": b"sequence"}),
    ], metadata={b"dataset": b"test"})
    data = pa.Table.from_pydict({"x": [1, 2], "items": [[1], [2, 3]]}, schema=schema)
    values = np.empty((1, 2), dtype=object)
    values[0] = [2, [2, 3]]
    restored = frame_from_values(values, data)
    assert restored.schema.equals(schema, check_metadata=True)
    result = set_columns(data, {"x": [1.25, 2.75]})
    assert result.schema.metadata == schema.metadata
    assert result.schema.field("x").metadata == schema.field("x").metadata
    assert result.column("x").to_pylist() == [1.25, 2.75]
    assert result.schema.field("items").equals(schema.field("items"), check_metadata=True)


def test_arrow_permutations_preserve_metadata_for_native_model_callbacks():
    import dalex as dx

    schema = pa.schema([
        pa.field("x", pa.int64(), nullable=False, metadata={b"unit": b"count"}),
        pa.field("z", pa.float64(), metadata={b"unit": b"meter"}),
    ], metadata={b"dataset": b"metadata-sensitive-model"})
    data = pa.Table.from_pydict({"x": [1, 2, 3], "z": [4., 5., 6.]}, schema=schema)
    for columns in (["x"], ["x", "z"]):
        changed = shuffle_columns(data, columns, [2, 0, 1])
        assert changed.schema.equals(schema, check_metadata=True)
        np.testing.assert_array_equal(changed.column("x").to_numpy(), [3, 1, 2])

    def predict(model, batch):
        assert isinstance(batch, pa.Table)
        assert batch.schema.equals(schema, check_metadata=True)
        return 2 * batch.column("x").to_numpy() + batch.column("z").to_numpy()

    class MetadataModel:
        pass

    exp = dx.Explainer(MetadataModel(), data, predict(None, data), predict_function=predict,
                       model_type="regression", label="metadata", verbose=False)
    result = exp.model_parts(N=None, B=2, random_state=42).result
    assert np.isfinite(result["dropout_loss"]).all()
    assert data.schema.equals(schema, check_metadata=True)


@pytest.mark.parametrize("backend", ["polars", "arrow"])
@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("timezone", [None, "America/New_York"])
def test_temporal_bridge_preserves_units_dst_and_missing_values(backend, unit, timezone):
    from dalex._dataframe import frame_from_values

    # These instants have the same wall-clock time on the DST fall-back day.
    instants = pd.to_datetime(["2025-11-02T05:30:00.123456789Z", "2025-11-02T06:30:00.123456789Z"], utc=True)
    integers = instants.to_numpy(dtype=f"datetime64[{unit}]").view("i8").tolist() + [None]
    data = pl.DataFrame({
        "when": pl.Series(integers, dtype=pl.Int64).cast(pl.Datetime(unit, timezone)),
        "elapsed": pl.Series([1, 1000001, None], dtype=pl.Int64).cast(pl.Duration(unit)),
    })
    reference = data if backend == "polars" else data.to_arrow()
    values = data.to_pandas().to_numpy()
    restored = frame_from_values(values, reference)
    assert restored.schema == reference.schema
    assert restored.equals(reference)
    for row in range(3):
        replaced = set_columns(reference, data.to_pandas().iloc[row].to_dict(), scalar=True)
        assert replaced.equals(take_rows(reference, [row] * 3))


def test_chunked_arrow_gathers_preserve_null_and_nan_distinction():
    # Different chunk boundaries are common after Arrow dataset scans.
    data = pa.table({
        "x": pa.chunked_array([[1., None], [np.nan], [3.]]),
        "category": pa.chunked_array([
            pa.array(["a", "b"]).dictionary_encode(),
            pa.array(["b", None]).dictionary_encode(),
        ]),
    })
    expected = [None, 3., 1., np.nan, None]
    result = take_rows(data, [1, -1, 0, 2, 1])
    np.testing.assert_array_equal(result.column("x").is_null().to_numpy(), [True, False, False, False, True])
    np.testing.assert_allclose(result.column("x").to_numpy(), np.array(expected, dtype=float), equal_nan=True)
    assert result.column("category").to_pylist() == ["b", None, "a", "b", "b"]
    shuffled = shuffle_columns(data, ["x", "category"], [3, 2, 1, 0])
    expected = data.take(pa.array([3, 2, 1, 0]))
    # Arrow deliberately considers NaN unequal to itself, including Table.equals.
    assert shuffled.schema == expected.schema
    assert shuffled.column("category").equals(expected.column("category"))
    np.testing.assert_array_equal(shuffled.column("x").is_null().to_numpy(), expected.column("x").is_null().to_numpy())
    np.testing.assert_allclose(shuffled.column("x").to_numpy(), expected.column("x").to_numpy(), equal_nan=True)


def test_arrow_dictionary_numpy_boundaries_do_not_expose_masked_category():
    from dalex._dataframe import as_numpy_vector, iter_column_arrays, to_numpy

    # Arrow permits arbitrary physical index values underneath a null bitmap.
    indices = pa.array([0, 1, 2], mask=[False, False, True], type=pa.int8())
    category = pa.DictionaryArray.from_arrays(indices, pa.array(["a", "b", "unused"]))
    data = pa.table({"kind": category, "x": [1., 2., 3.]})
    assert column_values(data, "kind").tolist() == ["a", "b", None]
    assert next(iter_column_arrays(data)).tolist() == ["a", "b", None]
    assert to_numpy(data)[:, 0].tolist() == ["a", "b", None]
    assert as_numpy_vector(data.select(["kind"]), "y").tolist() == ["a", "b", None]
    assert as_numpy_vector(data.column("kind"), "y").tolist() == ["a", "b", None]


def test_pandas_replacement_series_is_positional_and_fractional_values_can_widen():
    data = pd.DataFrame({"x": [1, 2, 3]}, index=[11, 7, 3])
    result = set_columns(data, {"x": pd.Series([3, 1, 2], index=[3, 11, 7])})
    assert result.x.tolist() == [3, 1, 2]
    assert result.index.equals(data.index)
    assert set_columns(data, {"x": 1.5}).x.tolist() == [1.5] * 3
    assert data.x.tolist() == [1, 2, 3]


@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_untyped_all_null_feature_survives_observation_replacement(backend):
    from dalex._dataframe import frame_from_values

    data = pl.DataFrame({"missing": [None, None], "x": [1., 2.]})
    reference = data if backend == "polars" else data.to_arrow()
    for values in (None, [None, None]):
        result = set_columns(reference, {"missing": values})
        assert result.schema == reference.schema
        assert result.equals(reference)
    restored = frame_from_values(np.array([[None, 1.]], dtype=object), reference)
    assert restored.schema == reference.schema
    assert restored.equals(take_rows(reference, [0]))


class CategoricalModel:
    def predict(self, data):
        return np.array([12. if pd.isna(value) else (11. if value == True or value == "<NA>" else 10.)
                         for value in column_values(data, "code")])


@pytest.mark.parametrize("backend", ["pandas", "polars", "arrow"])
@pytest.mark.parametrize("kind", ["boolean", "string", "encoded"])
@pytest.mark.parametrize("family", ["partial", "conditional"])
def test_categorical_profiles_distinguish_missing_and_literal_labels(backend, kind, family):
    import dalex as dx

    if kind == "boolean":
        original = pd.DataFrame({"code": pd.Series([True, False, None], dtype="boolean")})
    elif kind == "string":
        original = pd.DataFrame({"code": ["<NA>", "nan", None]})
    else:
        original = pd.DataFrame({"code": [0., 1., np.nan]})
    data = {"pandas": lambda: original, "polars": lambda: pl.from_pandas(original),
            "arrow": lambda: pa.Table.from_pandas(original)}[backend]()
    model = CategoricalModel()
    exp = dx.Explainer(model, data, model.predict(data), verbose=False, label="model", model_type="regression")
    extra = {"variable_splits": {"code": [0., 1., np.nan]}} if kind == "encoded" else {}
    result = exp.model_profile(type=family, N=None, variables=["code"], variable_type="categorical",
                               verbose=False, **extra).result
    assert len(result) == 3 and result["_x_"].is_unique
    np.testing.assert_allclose(sorted(result["_yhat_"]), [10., 11., 12.])


def test_optional_leakage_check_does_not_require_temporal_numpy_conversion(monkeypatch):
    import dalex as dx

    data = pl.DataFrame({"x": [1., 2., 3.], "when": pd.date_range("2025-01-01", periods=3, tz="UTC")})
    original = pl.Series.to_numpy

    def guarded(series, *args, **kwargs):
        if isinstance(series.dtype, pl.Datetime):
            raise RuntimeError("this backend cannot export timezone values to NumPy")
        return original(series, *args, **kwargs)

    monkeypatch.setattr(pl.Series, "to_numpy", guarded)
    exp = dx.Explainer(CategoricalModel(), data, np.array([1.1, 2.1, 3.1]),
                       predict_function=lambda model, frame: frame["x"].to_numpy(),
                       verbose=True, model_type="regression")
    np.testing.assert_array_equal(exp.y_hat, [1., 2., 3.])


class MissingNumericModel:
    def predict(self, data):
        values = column_values(data, "x").astype(float)
        return np.where(np.isnan(values), 7., values) + column_values(data, "z")


@pytest.mark.parametrize("backend", ["pandas", "polars", "arrow"])
@pytest.mark.parametrize("family", ["partial", "conditional"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_numeric_profiles_retain_a_missing_intervention(backend, family, all_missing):
    import dalex as dx

    original = pd.DataFrame({"x": [np.nan] * 3 if all_missing else [1., np.nan, 3.],
                             "z": [10., 40., 30.]})
    data = {"pandas": lambda: original, "polars": lambda: pl.from_pandas(original),
            "arrow": lambda: pa.Table.from_pandas(original)}[backend]()
    model = MissingNumericModel()
    exp = dx.Explainer(model, data, model.predict(data), verbose=False,
                       model_type="regression", label="model")
    # The automatic all-missing grid must work as well as an explicit missing
    # intervention in a column that also contains observed numeric values.
    extra = {} if all_missing else {"variable_splits": {"x": [1., np.nan, 3.]}}
    result = exp.model_profile(type=family, N=None, variables=["x"],
                               verbose=False, **extra).result
    missing = result.loc[result["_x_"].isna(), "_yhat_"]
    assert len(missing) == 1
    expected = 7. + (original.z.mean() if family == "partial" or all_missing else 40.)
    np.testing.assert_allclose(missing.to_numpy(dtype=float), [expected])


@pytest.mark.parametrize("backend", ["pandas", "polars", "arrow"])
def test_accumulated_numeric_profiles_reject_an_unordered_missing_grid(backend):
    import dalex as dx

    original = pd.DataFrame({"x": [1., np.nan, 3.], "z": [10., 40., 30.]})
    data = {"pandas": lambda: original, "polars": lambda: pl.from_pandas(original),
            "arrow": lambda: pa.Table.from_pandas(original)}[backend]()
    model = MissingNumericModel()
    exp = dx.Explainer(model, data, model.predict(data), verbose=False,
                       model_type="regression")
    with pytest.raises(ValueError, match="missing.*numeric.*grid"):
        exp.model_profile(type="accumulated", N=None, variables=["x"],
                          variable_splits={"x": [1., np.nan, 3.]}, verbose=False)
