"""Cross-backend DALEX tests; CI installs Polars so neither backend is skipped."""
import numpy as np
import pandas as pd
import pytest

import narwhals as nw
from dalex._dataframe import as_eager_dataframe, as_numpy_vector, take_rows, shuffle_columns

import polars as pl


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_backend_preserved_during_sample_and_permutation(backend):
    original = pd.DataFrame({"age": [10, 20, 30, 40], "income": [4.0, 7.0, 2.0, 5.0]})
    data = original if backend == "pandas" else pl.from_pandas(original)
    sampled = take_rows(data, [3, 0, 3, 2])
    shuffled = shuffle_columns(sampled, ["age", "income"], [2, 0, 3, 1])
    assert type(shuffled) is type(data)
    np.testing.assert_array_equal(nw.from_native(sampled).to_numpy(), original.iloc[[3, 0, 3, 2]].to_numpy())
    np.testing.assert_array_equal(nw.from_native(shuffled).to_numpy(), original.iloc[[3, 0, 3, 2]].iloc[[2, 0, 3, 1]].to_numpy())


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_repeated_and_empty_row_selections(backend):
    original = pd.DataFrame({"x": [1, 2, 3], "y": [5, 6, 7]})
    data = original if backend == "pandas" else pl.from_pandas(original)
    assert take_rows(data, []).shape == (0, 2)
    np.testing.assert_array_equal(
        nw.from_native(take_rows(data, [2, 2, 0])).to_numpy(),
        original.iloc[[2, 2, 0]].to_numpy(),
    )


def test_pandas_nonstandard_index_preserved():
    data = pd.DataFrame({"x": [1, 2, 3]}, index=[7, 9, 11])
    assert take_rows(data, [2, 0]).index.tolist() == [11, 7]


def test_reject_unsupported_object():
    with pytest.raises(TypeError):
        as_eager_dataframe({"not": "a dataframe"})


def test_polars_lazy_is_rejected():
    with pytest.raises(TypeError):
        as_eager_dataframe(pl.DataFrame({"x": [1]}).lazy())


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_nonpermuted_columns_stay_unchanged(backend):
    original = pd.DataFrame({"x": [1, 2, 3], "y": [11, 12, 13], "z": [21, 22, 23]})
    data = original if backend == "pandas" else pl.from_pandas(original)
    result = shuffle_columns(data, ["x", "y"], [2, 0, 1])
    np.testing.assert_array_equal(nw.from_native(result).get_column("x").to_numpy(), [3, 1, 2])
    np.testing.assert_array_equal(nw.from_native(result).get_column("y").to_numpy(), [13, 11, 12])
    np.testing.assert_array_equal(nw.from_native(result).get_column("z").to_numpy(), [21, 22, 23])


def test_polars_series_and_one_column_targets():
    assert as_numpy_vector(pl.Series("y", [1.0, 2.0]), "y").tolist() == [1.0, 2.0]
    assert as_numpy_vector(pl.DataFrame({"y": [1, 2]}), "y").tolist() == [1, 2]
    with pytest.raises(ValueError):
        as_numpy_vector(pl.DataFrame({"a": [1], "b": [2]}), "y")


def test_take_zero_rows():
    data = pd.DataFrame({"x": [1, 2]})
    assert take_rows(data, []).empty

@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_dtype_index_and_collision_preservation(backend):
    from datetime import date
    original = pd.DataFrame({
        "__dalex_row_id__": [1, 2, 3], "__dalex_order__": ["a", None, "b"],
        "float": [1.0, np.nan, 3.0], "cat": pd.Categorical(["a", "b", "a"]),
        "date": [date(2020, 1, i) for i in (1, 2, 3)],
    }, index=[11, 7, 99])
    data = original if backend == "pandas" else pl.from_pandas(original)
    result = shuffle_columns(data, list(data.columns), [2, 0, 1])
    assert type(result) is type(data)
    if backend == "pandas":
        assert result.index.tolist() == data.index.tolist()
        assert result.dtypes.equals(data.dtypes)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), original.iloc[[2, 0, 1]].reset_index(drop=True))
    else:
        assert result.schema == data.schema
        assert result.equals(data[[2, 0, 1]])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_row_bounds_and_negative_positions(backend):
    data = pd.DataFrame({"x": [10, 20, 30]})
    if backend == "polars":
        data = pl.from_pandas(data)
    assert nw.from_native(take_rows(data, [-1, 0])).get_column("x").to_list() == [30, 10]
    for positions in ([3], [-4]):
        with pytest.raises(IndexError):
            take_rows(data, positions)
    for positions in ([1.5], [[1]], [True]):
        with pytest.raises(TypeError):
            take_rows(data, positions)


def test_nondefault_index_permutation_is_positional():
    data = pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]}, index=[100, 7, 100])
    result = shuffle_columns(data, "x", [2, 0, 1])
    assert result["x"].tolist() == [30, 10, 20]
    assert result.index.equals(data.index)
    pd.testing.assert_frame_equal(data, pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]}, index=[100, 7, 100]))


def test_legacy_pandas_integer_column_names():
    import dalex as dx
    data = pd.DataFrame({0: [1., 2., 3.], 1: [3., 2., 1.]}, index=[20, 5, 99])
    class Model:
        def predict(self, frame):
            return frame[0].to_numpy()+frame[1].to_numpy()/2
    explainer = dx.Explainer(Model(), data, np.arange(3.), verbose=False)
    result = explainer.model_parts(B=2, N=None, random_state=42)
    assert set(result.result.variable) == {0, 1, "_baseline_", "_full_model_"}

@pytest.mark.parametrize('backend',['pandas','polars'])
def test_masked_replacement_preserves_categorical_schema(backend):
    from dalex._dataframe import set_columns
    frame=pd.DataFrame({'category':pd.Categorical(['a','b','a']), 'value':[1.,2.,3.]})
    data=frame if backend=='pandas' else pl.from_pandas(frame)
    result=set_columns(data,{'category':np.array(['b','b','a'],dtype=object)})
    if backend=='pandas':
        assert result.dtypes.equals(data.dtypes)
        assert result.category.tolist()==['b','b','a']
    else:
        assert result.schema==data.schema
        assert result['category'].to_list()==['b','b','a']
    assert nw.from_native(data)['category'].to_list()==['a','b','a']


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_matrix_callback_bridge_preserves_mixed_feature_types(backend):
    from dalex._dataframe import frame_from_values, native_observation

    reference = pd.DataFrame({
        "category": pd.Categorical(["a", "b"], categories=["a", "b", "unused"]),
        "value": [1.0, 2.0],
    })
    if backend == "polars":
        reference = pl.from_pandas(reference)
    values = np.array([["b", 1.5], ["a", np.nan]], dtype=object)
    restored = frame_from_values(values, reference)
    assert type(restored) is type(reference)
    if backend == "pandas":
        assert restored.category.dtype == reference.category.dtype
        assert pd.api.types.is_float_dtype(restored.value)
    else:
        assert restored.schema == reference.schema
        observation = native_observation(["b", 1.5], reference, single=True)
        assert observation.schema == reference.schema
    np.testing.assert_allclose(nw.from_native(restored)["value"].to_numpy(), [1.5, np.nan])
