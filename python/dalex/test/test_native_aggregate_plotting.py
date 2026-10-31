"""Native profile display preserves the published pandas figure contract."""
import copy
import json

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx
from test_native_plot_preparation import FormulaModel, assert_figures_equal, frame_as, no_legacy_materialization
from test_native_profile_workflows import native_computation_only


def profiles(backend, family, groups, categorical=False):
    data = pl.DataFrame({'x': [0., 1., 2., 3.], 'z': [10., 20., 30., 40.],
                         'group': ['a', None, 'b', 'a'], 'integer': [1, 2, 1, 2],
                         'floating': [1.5, 2.5, 1.5, 2.5]})
    frame = frame_as(data, backend)

    def predict(model, batch):
        assert type(batch) is type(frame)
        return nw.from_native(batch)['x'].to_numpy() + nw.from_native(batch)['z'].to_numpy()

    exp = dx.Explainer(FormulaModel(), frame, predict(None, frame), predict_function=predict,
                       model_type='regression', label='model', verbose=False)
    return exp.model_profile(type=family, N=None, variables=['integer'] if categorical else ['x', 'z'],
                             variable_type='categorical' if categorical else 'numerical',
                             groups=groups, grid_points=3, random_state=42, verbose=False)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('family', ['partial', 'conditional', 'accumulated'])
@pytest.mark.parametrize('groups', [None, ['group'], ['integer', 'floating']])
@pytest.mark.parametrize('bars', [False, True])
def test_profile_plot_matches_explicit_legacy_view(monkeypatch, backend, family, groups, bars):
    value = profiles(backend, family, groups)
    legacy = copy.deepcopy(value)
    legacy.result  # Explicit publication must remain authoritative.
    kwargs = {'geom': 'bars' if bars else 'aggregates', 'center': not bars, 'show': False}
    expected = json.loads(legacy.plot(**kwargs).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot(**kwargs).to_json())
    assert_figures_equal(actual, expected)
    assert '_compatibility_result' not in value.__dict__


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('family', ['partial', 'conditional', 'accumulated'])
@pytest.mark.parametrize('categorical', [False, True])
def test_profile_fit_to_plot_never_constructs_pandas(monkeypatch, backend, family, categorical):
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        value = profiles(backend, family, ['group'], categorical)
        assert value.plot(show=False) is not None


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_profile_plot_combines_edited_pandas_and_native(monkeypatch, backend):
    native = profiles(backend, 'partial', ['group'])
    edited = profiles('pandas', 'partial', ['group'])
    edited.result['_label_'] = 'edited'
    with no_legacy_materialization(monkeypatch):
        figure = native.plot(edited, variables=['z'], show=False)
    assert 'edited' in [trace.name for trace in figure.data]
    assert '_compatibility_result' not in native.__dict__


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_profile_categorical_plot_and_raw_overlay(backend):
    value = profiles(backend, 'partial', None, categorical=True)
    legacy = copy.deepcopy(value)
    legacy.result
    assert_figures_equal(json.loads(value.plot(show=False).to_json()), json.loads(legacy.plot(show=False).to_json()))
    numerical = profiles(backend, 'partial', None)
    assert numerical.plot(geom='profiles', show=False) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['large', 'nullable_integer', 'float_missing', 'boolean', 'categorical'])
def test_profile_display_edge_types_match_legacy(monkeypatch, backend, kind):
    from dalex.model_explanations import AggregatedProfiles
    from dalex.model_explanations._aggregated_profiles._table import AggregatedPartitions
    groups = {'nullable_integer': pl.Series([1, None, 3], dtype=pl.Int64),
              'float_missing': pl.Series([1., None, np.nan]),
              'boolean': pl.Series([True, None, False]),
              'categorical': pl.Series(['a', None, 'b']).cast(pl.Categorical),
              'large': pl.Series(['a', 'b', 'c'])}
    data = pl.DataFrame({'_vname_': ['x'] * 3, '_label_': ['model'] * 3,
                         '_x_': [2**53 + 1, 2**53 + 3, 2**53 + 5] if kind == 'large' else [0., 1., 2.],
                         '_yhat_': [1., 2., 3.], 'group': groups[kind]})
    value = AggregatedProfiles()
    dtype = {'group': 'object'} if kind in ['boolean', 'categorical', 'large'] else {}
    value.result = AggregatedPartitions([frame_as(data, backend)], None, ['group'], dtype, True)
    value.mean_prediction = 2.
    legacy = copy.deepcopy(value)
    legacy.result
    expected = json.loads(legacy.plot(show=False).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot(show=False).to_json())
    assert_figures_equal(actual, expected)


@pytest.mark.parametrize('integer_dtype', ['Int64', 'UInt64', 'Int32'])
def test_nullable_mixed_numeric_group_labels_match_pandas(integer_dtype):
    from dalex.model_explanations._aggregated_profiles._table import AggregatedPartitions
    frame = pd.DataFrame({'_vname_': ['x'] * 3, '_label_': ['model'] * 3,
                           '_x_': [0., 1., 2.], '_yhat_': [1., 2., 3.],
                           'g': pd.Series([1, None, 3], dtype=integer_dtype),
                           'h': pd.Series([1.5, 2.5, None], dtype='Float64')})
    table = AggregatedPartitions([frame], None, ['g', 'h'],
                                  {'g': frame.g.dtype, 'h': frame.h.dtype}, True)
    expected = table._dalex_pandas_view()['_groups_'].tolist()
    assert table._dalex_plot_data()['_groups_'].tolist() == expected


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('dtype', [pl.Datetime('ns'), pl.Datetime('ns', 'Europe/Warsaw'), pl.Duration('ns'), pl.Date])
def test_temporal_group_captions_keep_nanoseconds_and_missing_labels(monkeypatch, backend, dtype):
    from dalex.model_explanations._aggregated_profiles._table import AggregatedPartitions
    frame = pl.DataFrame({'_vname_': ['x'] * 3, '_label_': ['model'] * 3,
                           '_x_': [0., 1., 2.], '_yhat_': [1., 2., 3.],
                           'g': pl.Series([1, None, 3], dtype=pl.Int64).cast(dtype)})
    table = AggregatedPartitions([frame_as(frame, backend)], None, ['g'], {}, True)
    expected = table._dalex_pandas_view()['_groups_'].tolist()
    with native_computation_only(monkeypatch, backend):
        actual = nw.from_native(table._dalex_plot_data())['_groups_'].to_list()
    assert actual == expected


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_distinct_null_nan_group_does_not_promote_other_integer_captions(monkeypatch, backend):
    from dalex.model_explanations._aggregated_profiles._table import AggregatedPartitions
    frame = pl.DataFrame({'_vname_': ['x'] * 3, '_label_': ['model'] * 3,
                           '_x_': [0., 1., 2.], '_yhat_': [1., 2., 3.],
                           'g': [1., None, np.nan], 'h': [1, 2, 3]})
    table = AggregatedPartitions([frame_as(frame, backend)], None, ['g', 'h'], {}, True)
    expected = table._dalex_pandas_view()['_groups_'].tolist()
    with native_computation_only(monkeypatch, backend):
        actual = nw.from_native(table._dalex_plot_data())['_groups_'].to_list()
    assert actual == expected == ['1.0_1', 'None_2', 'nan_3']
