"""Complete CP figure parity, mutable metadata and native construction guards."""
import copy
import json
import warnings

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx
from test_native_plot_preparation import FormulaModel, frame_as, assert_figures_equal, no_legacy_materialization
from test_native_profile_workflows import native_computation_only


def explanation(backend, *, include=True, count=2, variables=('x', 'z'), index=None, nullable=False):
    data = pl.DataFrame({'x': [0., 1., 2., 3.], 'z': [10., 20., 30., 40.],
                         'group': ['a', 'b', 'a', 'b']})
    frame = frame_as(data, backend)
    if backend == 'pandas' and index is not None:
        frame.index = index
    if nullable:
        frame = frame.astype({'x': 'Float64', 'z': 'Float64'})

    def predict(model, batch):
        assert type(batch) is type(frame)
        f = nw.from_native(batch)
        return f['x'].to_numpy() + f['z'].to_numpy() * .2

    exp = dx.Explainer(FormulaModel(), frame, predict(None, frame), predict_function=predict,
                       model_type='regression', label='model', verbose=False)
    observation = nw.from_native(frame).head(count).to_native()
    return exp.predict_profile(observation, variables=list(variables), grid_points=3,
                                variable_splits_with_obs=include, verbose=False)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('include', [False, True])
@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('categorical', [False, True])
@pytest.mark.parametrize('color', ['_label_', '_ids_'])
def test_cp_plot_matches_explicit_legacy_views(monkeypatch, backend, include, count, categorical, color):
    value = explanation(backend, include=include, count=count,
                         variables=('group', 'x') if categorical else ('x', 'z'))
    legacy = copy.deepcopy(value)
    legacy.result
    legacy.new_observation
    kwargs = {'variable_type': 'categorical' if categorical else 'numerical',
              'variables': ['group', 'x'] if categorical else None, 'color': color, 'show': False}
    expected = json.loads(legacy.plot(**kwargs).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot(**kwargs).to_json())
    assert_figures_equal(actual, expected)
    assert '_compatibility_result' not in value.__dict__
    assert '_compatibility_new_observation' not in value.__dict__


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('categorical', [False, True])
def test_cp_native_fit_through_plot_constructs_no_pandas(monkeypatch, backend, categorical):
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        value = explanation(backend, variables=('group',) if categorical else ('x', 'z'))
        assert value.plot(variable_type='categorical' if categorical else 'numerical', show=False) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_cp_plot_uses_edited_observation_and_result_indexes(monkeypatch, backend):
    value = explanation(backend, index=[7, 19, 30, 45])
    value.new_observation['_yhat_'] = [100., 200.]
    value.result['_label_'] = 'edited'
    value.result = value.result.iloc[::-1].copy()
    with no_legacy_materialization(monkeypatch):
        figure = value.plot(show=False)
    text = [entry[0] for trace in figure.data for entry in trace.customdata]
    assert any('_original_yhat_: 100.0' in caption for caption in text)
    assert any('_original_yhat_: 200.0' in caption for caption in text)
    assert all(trace.name == 'edited' for trace in figure.data if trace.showlegend)


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_aggregate_raw_overlay_is_native(monkeypatch, backend):
    from dalex.model_explanations import AggregatedProfiles
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        cp = explanation(backend, count=2)
        value = AggregatedProfiles()
        value.fit(cp, verbose=False)
        assert value.plot(geom='profiles', show=False) is not None


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_cp_exact_integer_point_membership(monkeypatch, backend):
    data = pl.DataFrame({'x': pl.Series([2**63 + 1, 2**63 + 3], dtype=pl.UInt64)})
    frame = frame_as(data, backend)
    exp = dx.Explainer(FormulaModel(), frame, np.array([0., 1.]),
                       predict_function=lambda model, batch: np.zeros(len(nw.from_native(batch))),
                       model_type='regression', label='exact', verbose=False)
    value = exp.predict_profile(nw.from_native(frame).head(1).to_native(),
        variable_splits={'x': [2**63 + 1, 2**63 + 3]}, variable_splits_with_obs=True, verbose=False)
    with no_legacy_materialization(monkeypatch):
        figure = value.plot(show=False)
    points = [trace for trace in figure.data if trace.mode == 'markers']
    assert len(points) == 1 and list(points[0].x) == [2**63 + 1]


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'float32', 'Float32', 'object', 'double[pyarrow]'])
@pytest.mark.parametrize('values', [[1., 2., None], [1.5, 2.25, None]])
def test_cp_legacy_publication_retains_dtypes_without_deprecated_flag(dtype, values):
    from dalex.predict_explanations._ceteris_paribus._table import ProfileTables
    data = pd.DataFrame({'x': pd.Series(values, dtype=dtype)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = data.convert_dtypes(convert_floating=False)
    table = ProfileTables([data], [data.index], ['x'])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = table._dalex_pandas_view()
    pd.testing.assert_frame_equal(actual, expected)


def test_cp_nullable_float_captions_keep_fractional_display(monkeypatch):
    value = explanation('pandas', nullable=True)
    legacy = copy.deepcopy(value)
    legacy.result
    expected = json.loads(legacy.plot(show=False).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot(show=False).to_json())
    assert_figures_equal(actual, expected)


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['datetime', 'timezone', 'duration', 'date', 'enum', 'boolean', 'decimal', 'string'])
def test_cp_typed_categorical_axes_and_hover_match_legacy(monkeypatch, backend, kind):
    from datetime import date
    from decimal import Decimal
    types = {
        'datetime': (pl.Datetime('ns'), [pd.Timestamp(1), pd.Timestamp(3)], None),
        'timezone': (pl.Datetime('ns', 'UTC'), [pd.Timestamp(1, tz='UTC'), pd.Timestamp(3, tz='UTC')], None),
        'duration': (pl.Duration('ns'), [pd.Timedelta(1), pd.Timedelta(3)], None),
        'date': (pl.Date, [date(2020, 1, 1), date(2020, 1, 3)], None),
        'enum': (pl.Enum(['a', 'b', 'unused']), ['a', 'b'], None),
        'boolean': (pl.Boolean, [True, False], None),
        'decimal': (pl.Decimal(20, 4), [Decimal('1.0001'), Decimal('2.0003')], None),
        'string': (pl.String, ['a', 'b'], None),
    }
    dtype, splits, missing = types[kind]
    values = (pl.Series([1, 3, None], dtype=pl.Int64).cast(dtype)
              if kind in ['datetime', 'timezone', 'duration'] else pl.Series([*splits, missing], dtype=dtype))
    data = pl.DataFrame({'value': values})
    frame = frame_as(data, backend)
    exp = dx.Explainer(FormulaModel(), frame, np.zeros(3),
        predict_function=lambda model, batch: np.zeros(len(nw.from_native(batch))) + 1.25,
        model_type='regression', label='typed', verbose=False)
    value = exp.predict_profile(nw.from_native(frame).head(1).to_native(),
        variable_splits={'value': splits}, variable_splits_with_obs=True, verbose=False)
    legacy = copy.deepcopy(value)
    legacy.result
    legacy.new_observation
    expected = json.loads(legacy.plot(variable_type='categorical', variables=['value'], show=False).to_json())
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot(variable_type='categorical', variables=['value'], show=False).to_json())
    assert_figures_equal(actual, expected)
    if kind in ['datetime', 'timezone', 'duration']:
        assert any('000000001' in entry[0] for trace in actual['data'] for entry in trace['customdata'])


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_cp_mixed_models_keep_union_of_feature_columns(monkeypatch, backend):
    first = explanation(backend, count=1)
    second = explanation('pandas', count=1)
    second.result['extra'] = 'caption'
    second.result['_label_'] = 'second'
    legacy = copy.deepcopy(first)
    legacy.result
    legacy.new_observation
    expected = json.loads(legacy.plot(second, show=False).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(first.plot(second, show=False).to_json())
    assert_figures_equal(actual, expected)
