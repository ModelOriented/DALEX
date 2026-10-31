"""Aspect/triplot native plot preparation and writable legacy views."""
import copy
import json

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

import dalex as dx
from test_native_plot_preparation import assert_figures_equal, frame_as, no_legacy_materialization
from test_native_profile_workflows import native_computation_only


FAMILIES = ['model', 'predict', 'shap', 'model_triplot', 'predict_triplot', 'shap_triplot']


def fit(backend, family, label='model'):
    rng = np.random.default_rng(713)
    data = pl.DataFrame(rng.normal(size=(40, 4)), schema=['a', 'b', 'c', 'd'])
    target = data['a'].to_numpy() * 2 + data['b'].to_numpy() / 3 - data['c'].to_numpy()
    model = LinearRegression().fit(data.to_numpy(), target)
    frame = frame_as(data, backend)

    def predict(model, batch):
        assert type(batch) is type(frame)
        return model.predict(nw.from_native(batch).to_numpy())

    exp = dx.Explainer(model, frame, target, predict_function=predict, label=label, verbose=False)
    aspect = dx.Aspect(exp)
    common = {'N': 24, 'B': 3, 'random_state': 42}
    if family == 'model':
        return aspect.model_parts(variable_groups={'pair': ['a', 'b'], 'c': ['c'], 'd': ['d']}, **common)
    if family == 'model_triplot':
        return aspect.model_triplot(**common)
    observation = nw.from_native(frame).head(1).to_native()
    if family.endswith('_triplot'):
        return aspect.predict_triplot(observation, type='shap' if family == 'shap_triplot' else 'default', **common)
    return aspect.predict_parts(observation, variable_groups={'pair': ['a', 'b'], 'c': ['c'], 'd': ['d']},
                                 type='shap' if family == 'shap' else 'default', **common)


def options(family):
    return {'widget': False} if family.endswith('triplot') else {}


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('family', FAMILIES)
def test_aspect_figures_match_published_legacy_views(monkeypatch, backend, family):
    value = fit(backend, family)
    legacy = copy.deepcopy(value)
    legacy.result
    if family.endswith('triplot'):
        legacy.single_variable_importance
    expected = json.loads(legacy.plot(show=False, **options(family)).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot(show=False, **options(family)).to_json())
    assert_figures_equal(actual, expected)
    assert '_compatibility_result' not in value.__dict__
    assert '_compatibility_single_variable_importance' not in value.__dict__


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('family', FAMILIES)
def test_aspect_fit_through_plot_never_constructs_pandas(monkeypatch, backend, family):
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        value = fit(backend, family)
        assert value.plot(show=False, **options(family)) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('split', ['model', 'aspect'])
@pytest.mark.parametrize('limit', [1, None])
def test_model_aspect_multiple_models_limits_and_splits(monkeypatch, backend, split, limit):
    pair = [fit(backend, 'model', label) for label in ['second', 'first']]
    legacy = copy.deepcopy(pair)
    for value in legacy:
        value.result
    expected = json.loads(legacy[0].plot(legacy[1], split=split, max_aspects=limit, show=False).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(pair[0].plot(pair[1], split=split, max_aspects=limit, show=False).to_json())
    assert_figures_equal(actual, expected)


def test_pandas_aspect_rounding_callbacks_retain_series_contract():
    value = fit('pandas', 'predict_triplot')
    seen = []

    def rounding(values, digits):
        if not np.isscalar(values):
            assert isinstance(values, pd.Series)
            seen.append(values.index.tolist())
        return np.around(values, digits)

    value.plot(rounding_function=rounding, widget=False, show=False)
    assert seen and seen[0] == list(range(4))


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('family', ['model_triplot', 'predict_triplot', 'shap_triplot'])
def test_native_widget_callbacks_preserve_results(monkeypatch, backend, family):
    from plotly.callbacks import Points, InputDeviceState
    value = fit(backend, family)
    before = value.result_as('pyarrow')
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        box = value.plot(widget=True, show=False)
        figure = box.children[0]
        clicked = 0
        for index, trace in enumerate(figure.data):
            if trace._click_callbacks:
                trace._dispatch_on_click(Points(point_inds=[0], trace_index=index), InputDeviceState())
                clicked += 1
    assert clicked
    assert before.equals(value.result_as('pyarrow'))
    assert '_compatibility_result' not in value.__dict__


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('as_lists', [False, True])
def test_typed_aspect_hover_metadata_preserves_precision(monkeypatch, backend, as_lists):
    from dalex.aspect._table import AspectTable
    from dalex.aspect._predict_aspect_importance.object import PredictAspectImportance
    observation = pl.DataFrame({'u': pl.Series([2**63 + 1], dtype=pl.UInt64),
                                 't': pl.Series([1], dtype=pl.Int64).cast(pl.Datetime('ns', 'UTC')),
                                 'f': pl.Series([1.2345], dtype=pl.Float32),
                                 'c': pl.Series(['a']).cast(pl.Enum(['a', 'b']))})
    groups = [['u'], ['t'], ['f'], ['u', 't', 'c']]
    numbers = pl.DataFrame({'aspect_name': ['large', 'time', 'float', 'mixed'],
                             'importance': [1., 2., 3., 4.], 'min_depend': [1., 1., 1., .5],
                             'label': ['typed'] * 4})
    value = PredictAspectImportance(variable_groups=dict(zip(numbers['aspect_name'], groups)))
    value.result = AspectTable(frame_as(numbers, backend), groups, [['u', 't']] * 4,
                               frame_as(observation, backend), value_lists=as_lists)
    value.intercept = 0.
    legacy = copy.deepcopy(value)
    legacy.result
    expected = json.loads(legacy.plot(show=False).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot(show=False).to_json())
    assert_figures_equal(actual, expected)
    assert any('9223372036854775809' in text and '000000001' in text
               for trace in actual['data'] for text in trace['hovertext'])


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('family', FAMILIES)
def test_edited_aspect_tables_remain_authoritative(monkeypatch, backend, family):
    value = fit(backend, family)
    field = 'dropout_loss' if family in ['model', 'model_triplot'] else 'importance'
    value.result[field] += .123
    if family.endswith('triplot'):
        single = 'dropout_loss_change' if family == 'model_triplot' else 'importance'
        value.single_variable_importance[single] += .456
    with no_legacy_materialization(monkeypatch):
        assert value.plot(show=False, **options(family)) is not None


@pytest.mark.parametrize('family', ['predict', 'shap', 'predict_triplot', 'shap_triplot'])
def test_aspect_observation_metadata_is_a_fit_time_snapshot(family):
    data = pl.DataFrame({'a': [0., 1., 2., 3.], 'b': [2., 0., 1., 4.], 'c': [1., 4., 0., 2.]})
    target = data['a'].to_numpy() + data['b'].to_numpy()
    model = LinearRegression().fit(data.to_numpy(), target)
    aspect = dx.Aspect(dx.Explainer(model, data, target,
        predict_function=lambda model, batch: model.predict(batch.to_numpy()), verbose=False))
    observation = data.head(1)
    common = {'N': 4, 'B': 2, 'random_state': 42,
              'type': 'shap' if family.startswith('shap') else 'default'}
    value = (aspect.predict_triplot(observation, **common) if family.endswith('triplot') else
             aspect.predict_parts(observation, variable_groups={'a': ['a'], 'bc': ['b', 'c']}, **common))
    before = value.result_as('pyarrow')
    figure = json.loads(value.plot(show=False, **options(family)).to_json())
    observation.replace_column(0, pl.Series('a', [999.]))
    assert value.result_as('pyarrow').equals(before)
    assert_figures_equal(json.loads(value.plot(show=False, **options(family)).to_json()), figure)


@pytest.mark.parametrize('family', ['predict', 'model'])
@pytest.mark.parametrize('limit', [0, -1, 1.5, '1'])
def test_aspect_invalid_limits_are_actionable(family, limit):
    value = fit('polars', family)
    with pytest.raises(ValueError, match='max_aspects'):
        value.plot(max_aspects=limit, show=False)
