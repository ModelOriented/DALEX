"""Native plot preparation, mutable legacy views and real SHAP integrations."""
import json
import base64
import pickle
from contextlib import contextmanager

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import dalex as dx
from dalex._explanation import CompatibilityTable


class FormulaModel:
    """Marker for explicitly supplied backend-preserving prediction functions."""


@contextmanager
def no_legacy_materialization(monkeypatch):
    original = CompatibilityTable.__get__

    def guard(descriptor, instance, owner=None):
        if instance is not None and not isinstance(instance.__dict__.get(descriptor.name), (np.ndarray, list, tuple)):
            raise AssertionError(f'Plot read legacy attribute {descriptor.name}')
        return original(descriptor, instance, owner)

    def forbidden(*args, **kwargs):
        raise AssertionError('Plot converted a native table to pandas')

    with monkeypatch.context() as patch:
        patch.setattr(CompatibilityTable, '__get__', guard)
        patch.setattr(nw.DataFrame, 'to_pandas', forbidden)
        patch.setattr(pl.DataFrame, 'to_pandas', forbidden)
        yield


def assert_figures_equal(actual, expected):
    """Compare Plotly's numeric payloads, including binary arrays, at 1e-12."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        if 'bdata' in expected:
            left = np.frombuffer(base64.b64decode(actual['bdata']), dtype=actual['dtype'])
            right = np.frombuffer(base64.b64decode(expected['bdata']), dtype=expected['dtype'])
            np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-12)
        else:
            for name in expected:
                assert_figures_equal(actual[name], expected[name])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_figures_equal(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-12, rel=1e-12)
    else:
        assert actual == expected


def frame_as(data, backend):
    return data.to_pandas() if backend == 'pandas' else data.to_arrow() if backend == 'pyarrow' else data


def explanation_pair(backend, family):
    rng = np.random.default_rng(391)
    data = pl.DataFrame(rng.normal(size=(40, 3)), schema=['z', 'a', 'b'])
    target = data['z'].to_numpy() * 2 + data['a'].to_numpy() / 3 + np.sin(data['b'].to_numpy())
    # Train identical models once on their numerical input contract; DALEX's
    # callback is still required to receive each original dataframe backend.
    output = []
    for label, offset in [('second', 0), ('first', .2)]:
        model = LinearRegression().fit(data.to_numpy(), target + offset * data['a'].to_numpy())
        frame = frame_as(data, backend)

        def predict(model, batch, expected=type(frame)):
            assert isinstance(batch, expected)
            return model.predict(nw.from_native(batch).to_numpy())

        exp = dx.Explainer(model, frame, target, label=label, predict_function=predict, verbose=False)
        output.append(exp.model_parts(N=30, B=3, random_state=42) if family == 'importance'
                      else exp.model_diagnostics())
    return output


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('split', ['model', 'variable'])
@pytest.mark.parametrize('max_vars', [1, None])
def test_importance_plot_native_preparation_and_backend_parity(monkeypatch, backend, split, max_vars):
    reference = explanation_pair('pandas', 'importance')
    expected = json.loads(reference[0].plot(reference[1], split=split, max_vars=max_vars, show=False).to_json())
    pair = explanation_pair(backend, 'importance')
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(pair[0].plot(pair[1], split=split, max_vars=max_vars, show=False).to_json())
    assert_figures_equal(actual, expected)
    assert '_compatibility_result' not in pair[0].__dict__


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('smooth', [False, True])
@pytest.mark.parametrize('sample', [10, None])
def test_diagnostics_plot_native_preparation_and_sampling(monkeypatch, backend, smooth, sample):
    reference = explanation_pair('pandas', 'diagnostics')
    expected = json.loads(reference[0].plot(reference[1], smooth=smooth, N=sample, show=False).to_json())
    pair = explanation_pair(backend, 'diagnostics')
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(pair[0].plot(pair[1], smooth=smooth, N=sample, show=False).to_json())
    assert_figures_equal(actual, expected)
    assert '_compatibility_result' not in pair[0].__dict__


@pytest.mark.parametrize('family', ['importance', 'diagnostics'])
def test_plot_mixed_backends_and_authoritative_edited_pandas_views(monkeypatch, family):
    first = explanation_pair('polars', family)[0]
    second = explanation_pair('pyarrow', family)[1]
    # An explicit public view, including user edits, remains authoritative.
    column = 'dropout_loss' if family == 'importance' else 'residuals'
    first.result[column] += 2
    reference = explanation_pair('pandas', family)
    reference[0].result[column] += 2
    kwargs = {'show': False} if family == 'importance' else {'show': False, 'smooth': False}
    expected = json.loads(reference[0].plot(reference[1], **kwargs).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(first.plot(second, **kwargs).to_json())
    assert_figures_equal(actual, expected)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['TreeExplainer', 'KernelExplainer'])
@pytest.mark.parametrize('method', ['predict_parts', 'model_parts'])
def test_real_shap_lazy_observation_fit_export_plot_and_pickle(monkeypatch, backend, kind, method):
    import shap

    rng = np.random.default_rng(91)
    data = pl.DataFrame(rng.normal(size=(32, 3)), schema=['x', 'z', 'u'])
    target = data['x'].to_numpy() * 2 - data['z'].to_numpy()
    model = HistGradientBoostingRegressor(max_iter=3, min_samples_leaf=4, random_state=42).fit(data, target)
    frame = frame_as(data, backend)
    if backend == 'pandas':
        frame.index = pd.Index(np.arange(32) * 3 + 1, name='rows')
    exp = dx.Explainer(model, frame, target, verbose=False, label='model')
    observation = nw.from_native(frame).head(1).to_native()
    kwargs = {'shap_explainer_type': kind, 'type': 'shap_wrapper', 'N': 12, 'random_state': 42}
    if kind == 'KernelExplainer':
        kwargs.update(nsamples=8, silent=True)
    with no_legacy_materialization(monkeypatch):
        value = (exp.predict_parts(observation, **kwargs) if method == 'predict_parts'
                 else exp.model_parts(**kwargs))
        exported = value.result_as(table='new_observation')
        assert type(exported) is type(frame)
        restored = pickle.loads(pickle.dumps(value)) if kind == 'TreeExplainer' else value
        # KernelExplainer itself stores a local prediction closure and is not
        # pickleable upstream; test our metadata separately below.
        captured = {}
        monkeypatch.setattr(shap, 'force_plot', lambda **kw: captured.update(kw))
        monkeypatch.setattr(shap, 'summary_plot', lambda **kw: captured.update(kw))
        restored.plot(show=False)
        np.testing.assert_array_equal(nw.from_native(exported).to_numpy(),
                                      np.asarray(captured['features']))
        if backend == 'polars' and method == 'predict_parts':
            observation[0, 0] = -999
            np.testing.assert_array_equal(value.result_as(table='new_observation').to_numpy(), exported.to_numpy())
    assert '_compatibility_new_observation' not in value.__dict__
    native = value.result_as(table='new_observation')
    assert isinstance(value.new_observation, pd.DataFrame)
    if backend == 'pandas':
        pd.testing.assert_index_equal(value.new_observation.index, native.index)
    value.new_observation.iloc[0, 0] = 999
    assert nw.from_native(value.result_as(table='new_observation'))[0, 0] == 999
    # Assigning a replacement invalidates the cached view.
    value.new_observation = native
    assert '_compatibility_new_observation' not in value.__dict__


def test_old_shap_pickle_observation_state_remains_usable():
    value = dx.wrappers.ShapWrapper('predict_parts')
    value.__dict__['new_observation'] = pd.DataFrame({'x': [1.]}, index=pd.Index([17], name='row'))
    restored = pickle.loads(pickle.dumps(value))
    pd.testing.assert_frame_equal(restored.new_observation, value.new_observation)
    pd.testing.assert_index_equal(restored.result_as(table='new_observation').index, value.new_observation.index)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('geom', ['roc', 'ecdf', 'lift'])
@pytest.mark.parametrize('size', [8, 301])
def test_performance_plot_native_preparation_and_backend_parity(monkeypatch, backend, geom, size):
    from dalex.model_explanations import ModelPerformance
    rng = np.random.default_rng(931)
    predictions = rng.uniform(size=size)
    target = (predictions > .3).astype(int)
    data = pl.DataFrame({'y': target, 'y_hat': predictions,
                         'residuals': target - predictions, 'label': ['model'] * size})
    reference = ModelPerformance('classification')
    reference.residuals = data.to_pandas()
    value = ModelPerformance('classification')
    value.residuals = frame_as(data, backend)
    np.random.seed(93)
    expected = json.loads(reference.plot(geom=geom, show=False).to_json())
    np.random.seed(93)
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot(geom=geom, show=False).to_json())
    assert_figures_equal(actual, expected)
    assert '_compatibility_residuals' not in value.__dict__


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_lift_plot_different_model_sizes_and_mutable_residuals(monkeypatch, backend):
    from dalex.model_explanations import ModelPerformance
    objects = []
    for size in [303, 7]:
        value = ModelPerformance('classification')
        value.residuals = frame_as(pl.DataFrame({'y': np.ones(size), 'y_hat': np.linspace(0, 1, size),
                                                'residuals': np.zeros(size), 'label': [str(size)] * size}), backend)
        objects.append(value)
    objects[1].residuals['y'] = 0
    with no_legacy_materialization(monkeypatch):
        fig = objects[0].plot(objects[1], geom='lift', show=False)
    assert len(fig.data[1].x) == 7
    np.testing.assert_array_equal(fig.data[1].y[1:], np.zeros(6))


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('family', ['break_down', 'break_down_interactions', 'shap'])
@pytest.mark.parametrize('max_vars', [1, 20])
def test_native_attribution_plot_with_mixed_feature_metadata(monkeypatch, backend, family, max_vars):
    data = pl.DataFrame({'x': [1., 4., 3., 2.], 'group': ['a', 'b', 'b', 'a'],
                         'integer': [2**60 + 1, 2**60 + 3, 2**60 + 5, 2**60 + 7]})

    def predict(model, batch):
        frame = nw.from_native(batch)
        return frame['x'].to_numpy() + np.array([1. if group == 'b' else 0. for group in frame['group'].to_list()])

    figures = []
    for selected in ['pandas', backend]:
        frame = frame_as(data, selected)
        exp = dx.Explainer(FormulaModel(), frame, predict(None, frame), predict_function=predict,
                           model_type='regression', label='model', verbose=False)
        value = exp.predict_parts(nw.from_native(frame).head(1).to_native(), type=family, B=3, random_state=42)
        with no_legacy_materialization(monkeypatch):
            figures.append(json.loads(value.plot(max_vars=max_vars, show=False).to_json()))
        assert '_compatibility_result' not in value.__dict__
    assert_figures_equal(*figures)


@pytest.mark.parametrize('family', ['break_down', 'break_down_interactions', 'shap'])
def test_attribution_plot_respects_edited_pandas_view(monkeypatch, family):
    from dalex._dataframe import column_values
    data = pl.DataFrame({'x': [1., 4., 3., 2.], 'z': [4., 1., 2., 3.]})
    exp = dx.Explainer(FormulaModel(), data, data['x'].to_numpy(), predict_function=lambda model, batch: column_values(batch, 'x'),
                       model_type='regression', label='model', verbose=False)
    value = exp.predict_parts(data.head(1), type=family, B=3, random_state=42)
    value.result['variable'] = 'edited caption'
    with no_legacy_materialization(monkeypatch):
        fig = value.plot(show=False)
    assert all(caption == 'edited caption' for caption in fig.data[0].y)


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('family', ['importance', 'diagnostics', 'performance', 'break_down', 'shap'])
def test_fit_through_plot_never_constructs_pandas_tables(monkeypatch, backend, family):
    from test_native_profile_workflows import native_computation_only
    data = pl.DataFrame({'x': [1., 4., 3., 2.], 'z': [4., 1., 2., 3.]})
    frame = frame_as(data, backend)

    def predict(model, batch):
        assert type(batch) is type(frame)
        return nw.from_native(batch)['x'].to_numpy()

    exp = dx.Explainer(FormulaModel(), frame, data['x'].to_numpy(), predict_function=predict,
                       model_type='regression', label='model', verbose=False)
    observation = nw.from_native(frame).head(1).to_native()
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        if family == 'importance':
            value = exp.model_parts(B=2, random_state=42)
        elif family == 'diagnostics':
            value = exp.model_diagnostics()
        elif family == 'performance':
            value = exp.model_performance()
        else:
            value = exp.predict_parts(observation, type=family, B=2, random_state=42)
        kwargs = {'smooth': False} if family == 'diagnostics' else {}
        assert value.plot(show=False, **kwargs) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('boolean', [False, True])
def test_roc_missing_predictions_and_boolean_targets(backend, boolean):
    from dalex.model_explanations import ModelPerformance
    target = np.array([1, 0, 1, 0, 1, 0], dtype=bool if boolean else int)
    data = pl.DataFrame({'y': target, 'y_hat': [.9, .8, .8, .1, None, np.nan],
                         'residuals': np.zeros(6), 'label': ['model'] * 6})
    value = ModelPerformance('classification')
    value.residuals = frame_as(data, backend)
    figure = value.plot(geom='roc', show=False)
    np.testing.assert_allclose(figure.data[0].x, [0, 0, 1/3, 2/3])
    np.testing.assert_allclose(figure.data[0].y, [0, 1/3, 2/3, 2/3])


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_roc_missing_targets_skip_in_denominators_and_groups(backend):
    from dalex.model_explanations import ModelPerformance
    data = pl.DataFrame({'y': [1., None, 0., 1.], 'y_hat': [.8, .9, .8, .1],
                         'residuals': np.zeros(4), 'label': ['model'] * 4})
    value = ModelPerformance('classification')
    value.residuals = frame_as(data, backend)
    figure = value.plot(geom='roc', show=False)
    np.testing.assert_allclose(figure.data[0].x, [0, 0, 1, 1])
    np.testing.assert_allclose(figure.data[0].y, [0, 0, .5, 1])


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_diagnostics_plot_combines_different_numeric_schemas(backend):
    from dalex.model_explanations import ResidualDiagnostics
    objects = []
    for floating in [False, True]:
        data = pl.DataFrame({'y_hat': np.array([1, 2, 3], dtype=np.float64 if floating else np.int32),
                             'residuals': [1., 2., 3.], 'ids': [1, 2, 3], 'label': [str(floating)] * 3})
        value = ResidualDiagnostics()
        value.result = frame_as(data, backend)
        objects.append(value)
    figure = objects[0].plot(objects[1], smooth=False, show=False)
    assert len(figure.data) == 2
    for trace in figure.data:
        np.testing.assert_allclose(trace.x, [1., 2., 3.])


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_roc_retains_exact_nullable_uint64_decision_scores(backend):
    from dalex.model_explanations import ModelPerformance
    data = pl.DataFrame({'y': [1, 0, 1, 1],
                         'y_hat': pl.Series([2**63 + 1, 2**63 + 3, None, 2**63 + 5], dtype=pl.UInt64),
                         'residuals': np.zeros(4), 'label': ['model'] * 4})
    value = ModelPerformance('classification')
    value.residuals = frame_as(data, backend)
    figure = value.plot(geom='roc', show=False)
    np.testing.assert_allclose(figure.data[0].x, [0, 0, 1, 1])
    np.testing.assert_allclose(figure.data[0].y, [0, 1/3, 1/3, 2/3])


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_native_plot_schema_promotion_preserves_missing_values_and_rejects_rounding(backend):
    from dalex._plotting import concat_frames
    first = frame_as(pl.DataFrame({'x': [1., 2.]}), backend)
    empty = frame_as(pl.DataFrame({'x': [None, None]}), backend)
    joined = concat_frames([nw.from_native(first), nw.from_native(empty)])
    assert joined['x'].to_list() == [1., 2., None, None]
    large = frame_as(pl.DataFrame({'x': pl.Series([2**63 + 1], dtype=pl.UInt64)}), backend)
    with pytest.raises(ValueError, match='round large integers'):
        concat_frames([nw.from_native(first), nw.from_native(large)])
