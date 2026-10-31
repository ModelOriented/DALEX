"""Pandas feature labels remain labels at the native adapter boundary."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import dalex as dx
from dalex._dataframe import shuffle_columns


@pytest.mark.parametrize('selector', [('g', 1), ('g',), 'g', [('g', 1), ('h', 1)], ['g']])
@pytest.mark.parametrize('reordered', [False, True])
def test_multiindex_joint_shuffle_preserves_levels_dtypes_and_row_labels(selector, reordered):
    frame = pd.DataFrame({0: pd.array([1, None, 3, 4], dtype='Int64'),
                          1: pd.Categorical(['a', 'b', 'a', 'b']),
                          2: pd.array([.5, 1.5, 2.5, None], dtype='Float64')})
    frame.columns = pd.MultiIndex.from_tuples([('g', 1), ('g', 2), ('h', 1)], names=['group', 'feature'])
    frame.index = pd.Index([101, 77, 205, 19], name='observation')
    if reordered:
        frame = frame.iloc[:, [2, 0, 1]]
    before = frame.copy(deep=True)
    permutation = [2, 0, 3, 1]
    expected = frame.copy(deep=True)
    expected.loc[:, selector] = frame.iloc[permutation, :].loc[:, selector].values
    actual = shuffle_columns(frame, selector, permutation)
    # pandas 1.5's legacy grouped assignment coerces extension columns to
    # object. The adapter deliberately retains the original extension dtypes.
    pd.testing.assert_frame_equal(actual.astype(object), expected.astype(object))
    pd.testing.assert_series_equal(actual.dtypes, frame.dtypes)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('label', [None, pd.Timestamp('2020-01-01'), np.int64(17), np.nan, pd.NA, pd.NaT])
def test_scalar_pandas_labels_are_not_interpreted_as_feature_sequences(label):
    frame = pd.DataFrame({label: [1., 2., 3.], 'other': [10., 20., 30.]}, index=[30, 10, 20])
    frame.columns = pd.Index([label, 'other'], dtype=object, tupleize_cols=False)
    result = shuffle_columns(frame, label, [2, 0, 1])
    expected = frame.copy()
    expected.loc[:, label] = frame.iloc[[2, 0, 1], :].loc[:, label].values
    pd.testing.assert_frame_equal(result, expected)


def predict(model, frame):
    assert isinstance(frame.columns, pd.MultiIndex)
    assert frame.columns.names == ['group', 'feature']
    return frame.to_numpy() @ [2., -1., .5]


def predict_numbers(model, frame):
    return frame.to_numpy() @ [2., -1., .5]


@pytest.mark.parametrize('label', [None, np.nan, pd.NA, pd.NaT])
def test_missing_feature_labels_survive_importance_label_sort_and_record_lookup(label):
    values = np.random.default_rng(93).normal(size=(24, 3))
    frame = pd.DataFrame(values, columns=pd.Index([label, 'a', 'b'], dtype=object))
    target = values @ [2., -1., .5] + .1
    exp = dx.Explainer(LinearRegression().fit(values, target), frame, target,
                       predict_function=predict_numbers, verbose=False)
    value = exp.model_parts(N=16, B=3, random_state=42)
    assert len(value.result) == 5
    assert value.result.variable.isna().sum() == 1
    assert np.isfinite(value.result.dropout_loss).all()
    assert value.permutation.shape == (3, 5)
    pd.testing.assert_frame_equal(value.result_as('pandas'), value.result)


def test_deep_multiindex_retains_exact_labels_and_legacy_best_effort_order():
    base = 2**63
    prefix = ('level',) * 13
    labels = [prefix + (base + offset,) for offset in [2, 1, 3]]
    values = np.random.default_rng(93).normal(size=(24, 3))
    frame = pd.DataFrame(values, columns=pd.MultiIndex.from_tuples(labels))
    target = values @ [2., -1., .5] + .1
    exp = dx.Explainer(LinearRegression().fit(values, target), frame, target,
                       predict_function=predict_numbers, verbose=False)
    value = exp.model_parts(N=16, B=3, random_state=42)
    actual = [name for name in value.permutation.columns if isinstance(name, tuple)]
    # Upstream's best-effort tuple sorting leaves these adjacent large levels
    # in input order. Preserve that layout without changing the label values.
    assert actual == [prefix + (base + offset,) for offset in [2, 1, 3]]


@pytest.mark.parametrize('selection', [None, ['g'], [('g', 1), ('h', 1)]])
@pytest.mark.parametrize('kind', ['variable_importance', 'ratio', 'difference'])
def test_multiindex_importance_full_and_partial_labels_keep_raw_permutations(selection, kind):
    values = np.random.default_rng(93).normal(size=(24, 3))
    columns = pd.MultiIndex.from_tuples([('g', 1), ('g', 2), ('h', 1)], names=['group', 'feature'])
    frame = pd.DataFrame(values, columns=columns, index=np.arange(24) * 3 + 7)
    target = values @ [2., -1., .5] + .1
    exp = dx.Explainer(LinearRegression().fit(values, target), frame, target, predict_function=predict,
                       model_type='regression', verbose=False)
    value = exp.model_parts(variables=selection, type=kind, N=16, B=3, random_state=42)
    expected_features = set(columns if selection is None else selection)
    assert set(value.result.variable) == expected_features | {'_baseline_', '_full_model_'}
    assert set(value.permutation.columns) == expected_features | {'_baseline_', '_full_model_'}
    assert value.permutation.shape == (3, len(expected_features) + 2)
    assert np.isfinite(value.result.dropout_loss).all()
    pd.testing.assert_frame_equal(value.result_as('pandas'), value.result)
    assert value.plot(show=False) is not None
    pd.testing.assert_frame_equal(exp.data, frame)
