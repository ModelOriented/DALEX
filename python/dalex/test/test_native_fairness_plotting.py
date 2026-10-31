"""Native fairness figures retain presentation and respect edited public views."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import polars as pl
import pyarrow as pa
import dalex as dx

from dalex.fairness import GroupFairnessClassification, GroupFairnessRegression
from dalex.fairness._basics.exceptions import ParameterCheckError
from test_native_plot_preparation import assert_figures_equal, no_legacy_materialization
from test_native_profile_workflows import native_computation_only, fitted_pipeline


TYPES = ['fairness_check', 'metric_scores', 'stacked', 'radar', 'performance_and_fairness',
         'heatmap', 'ceteris_paribus_cutoff', 'density']


def objects(backend, regression=False):
    rng = np.random.default_rng(842)
    protected = np.repeat(['a', 'b', 'c'], 40)
    if regression:
        y = rng.normal(size=120)
        prediction = .8 * y + rng.normal(size=120) * .4
        cls = GroupFairnessRegression
    else:
        y = np.tile(np.arange(40) % 2, 3)
        prediction = np.tile(np.linspace(.05, .95, 40), 3)
        cls = GroupFairnessClassification
    return [cls(y, prediction + offset, protected, 'a', label, _backend=backend)
            for label, offset in [('second', 0), ('first', .025)]]


def options(kind):
    return ({'verbose': False, 'subgroup': 'b', 'grid_points': 7, 'metrics': ['TPR', 'ACC']}
            if kind == 'ceteris_paribus_cutoff' else {} if kind == 'density' else {'verbose': False})


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('kind', TYPES)
@pytest.mark.parametrize('multiple', [False, True])
def test_fairness_plots_match_explicit_public_views(monkeypatch, backend, kind, multiple):
    pair = objects(backend)
    legacy = copy.deepcopy(pair)
    for value in legacy:
        value.result
        value.metric_scores
        value.parity_loss
    expected = json.loads(legacy[0].plot(legacy[1] if multiple else None, type=kind,
                                        show=False, **options(kind)).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(pair[0].plot(pair[1] if multiple else None, type=kind,
                                         show=False, **options(kind)).to_json())
    assert_figures_equal(actual, expected)
    assert all(not key.startswith('_compatibility_') for value in pair for key in value.__dict__)


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('kind', TYPES)
def test_fairness_construction_through_plot_never_constructs_pandas(monkeypatch, backend, kind):
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        pair = objects(backend)
        assert pair[0].plot(pair[1], type=kind, show=False, **options(kind)) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['fairness_check', 'density'])
def test_regression_fairness_plot_native_preparation(monkeypatch, backend, kind):
    pair = objects(backend, regression=True)
    expected_pair = copy.deepcopy(pair)
    for value in expected_pair:
        value.result
    expected = json.loads(expected_pair[0].plot(expected_pair[1], type=kind, show=False, **options(kind)).to_json())
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(pair[0].plot(pair[1], type=kind, show=False, **options(kind)).to_json())
    assert_figures_equal(actual, expected)


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_fairness_plots_respect_edits_without_constructing_pandas(monkeypatch, backend):
    value = objects(backend)[0]
    value.parity_loss.loc['TPR'] = 123.
    value.metric_scores.loc['b', 'TPR'] = .123
    value.result.loc['b', 'TPR'] = .456
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        radar = value.plot(type='radar', metrics=['TPR'], show=False, verbose=False)
        scores = value.plot(type='metric_scores', show=False, verbose=False)
        ratios = value.plot(type='fairness_check', show=False, verbose=False)
    assert 123. in list(radar.data[0].r)
    assert any(np.any(np.isclose(trace.x, .123)) for trace in scores.data)
    assert any(np.any(np.isclose(trace.x, -.544)) for trace in ratios.data)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_cutoff_plot_does_not_mutate_settings(backend):
    value = objects(backend)[0]
    before = dict(value.cutoff)
    value.plot(type='ceteris_paribus_cutoff', show=False, **options('ceteris_paribus_cutoff'))
    assert value.cutoff == before


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_heatmap_aligns_reordered_metric_views(backend):
    pair = objects(backend)
    pair[0].parity_loss.loc['TPR'] = 11.
    pair[1].parity_loss.loc['TPR'] = 22.
    pair[1].parity_loss = pair[1].parity_loss.iloc[::-1]
    figure = pair[0].plot(pair[1], type='heatmap', metrics=['TPR', 'ACC'], show=False, verbose=False)
    column = list(figure.data[0].x).index('TPR')
    np.testing.assert_allclose(np.asarray(figure.data[0].z)[:, column], [11., 22.])


@pytest.mark.parametrize('kind', ['fairness_check', 'metric_scores'])
def test_all_missing_fairness_scores_have_finite_axes(kind):
    value = objects('polars')[0]
    table = value.result if kind == 'fairness_check' else value.metric_scores
    table.loc[:, :] = np.nan
    figure = value.plot(type=kind, show=False, verbose=False)
    assert all(np.isfinite(axis) for axis in figure.layout.xaxis.range)


def test_legacy_fairness_objects_without_backend_metadata_plot():
    value = objects('pandas')[0]
    value.result = value.result.copy()
    value.parity_loss = value.parity_loss.copy()
    del value._result_backend
    assert value.plot(type='fairness_check', show=False, verbose=False) is not None
    assert value.plot(type='radar', show=False, verbose=False) is not None


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['fairness_check', 'density'])
def test_regression_fairness_construction_to_plot_is_native(monkeypatch, backend, kind):
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        pair = objects(backend, regression=True)
        assert pair[0].plot(pair[1], type=kind, show=False, **options(kind)) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_heatmap_fills_missing_metric_with_nan(backend):
    pair = objects(backend)
    pair[1].parity_loss = pair[1].parity_loss.drop('ACC')
    figure = pair[0].plot(pair[1], type='heatmap', metrics=['TPR', 'ACC'], show=False, verbose=False)
    column = list(figure.data[0].x).index('ACC')
    assert np.isnan(np.asarray(figure.data[0].z)[1, column])


@pytest.mark.parametrize('grid_points', [0, 1, 2.5])
def test_cutoff_grid_validation(grid_points):
    with pytest.raises(ParameterCheckError, match='grid_points'):
        objects('polars')[0].plot(type='ceteris_paribus_cutoff', subgroup='b', grid_points=grid_points,
                                  show=False, verbose=False)


def test_cutoff_all_invalid_metrics_reports_actionable_error_without_mutation():
    protected = np.repeat(['a', 'b', 'c'], 40)
    y = np.tile(np.arange(40) % 2, 3)
    y[protected == 'b'] = 0
    prediction = np.tile(np.linspace(.05, .95, 40), 3)
    value = GroupFairnessClassification(y, prediction, protected, 'a', 'missing', _backend='polars')
    before = dict(value.cutoff)
    with pytest.raises(ParameterCheckError, match='No cutoff has finite parity loss'):
        value.plot(type='ceteris_paribus_cutoff', subgroup='b', grid_points=4,
                    metrics=['TPR'], show=False, verbose=False)
    assert value.cutoff == before


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_legacy_parity_adapter_preserves_columns_values_and_indexes(backend):
    from dalex.fairness._group_fairness import utils
    pair = objects(backend)
    expected = []
    metrics = ['TPR', 'ACC']
    for value in pair:
        series = value.parity_loss
        data = pd.DataFrame({'score': series.to_numpy(), 'label': value.label, 'metric': series.index})
        expected.append(data.loc[data.metric.isin(metrics)])
    expected = pd.concat(expected)
    actual = utils._unwrap_parity_loss_data(pair[0], [pair[1]], metrics, False)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_legacy_ratio_adapter_preserves_row_order_and_indexes(backend):
    from dalex.fairness._group_fairness import utils
    value = objects(backend)[0]
    matrix = value.result
    expected = pd.DataFrame({'subgroup': np.repeat(matrix.index.to_numpy(), matrix.shape[1]),
        'metric': np.tile(matrix.columns.to_numpy(), len(matrix)), 'score': matrix.to_numpy().reshape(-1)})
    expected = expected.loc[expected.metric.isin(['TPR', 'ACC', 'PPV', 'FPR', 'STP']) &
                            (expected.subgroup != value.privileged)].copy()
    expected['score'] -= 1
    expected['label'] = value.label
    pd.testing.assert_frame_equal(utils._metric_ratios_2_df(value), expected)


def test_confusion_matrix_dataframe_adapters_match_legacy_contract():
    value = objects('polars')[0]._subgroup_confusion_matrix_metrics_object
    rows = value.subgroup_confusion_matrix_metrics
    expected = pd.DataFrame(rows).transpose()
    pd.testing.assert_frame_equal(value.to_horizontal_DataFrame(), expected)
    vertical = pd.concat([pd.DataFrame({'metric': list(metrics), 'subgroup': subgroup,
                                       'score': list(metrics.values())}) for subgroup, metrics in rows.items()], ignore_index=True)
    pd.testing.assert_frame_equal(value.to_vertical_DataFrame(), vertical)


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['fairness_check', 'density'])
def test_real_sklearn_lightgbm_pipeline_through_native_fairness_plot(fitted_pipeline, monkeypatch, backend, kind):
    model, original, target, classification = fitted_pipeline
    frame = original if backend == 'polars' else original.to_arrow()
    seen = []

    def predict(model, batch):
        assert type(batch) is type(frame)
        seen.append(type(batch))
        return model.predict_proba(batch)[:, 1] if classification else model.predict(batch)

    exp = dx.Explainer(model, frame, target, predict_function=predict,
                       model_type='classification' if classification else 'regression', verbose=False)
    membership = np.where(np.arange(64) % 4 < 2, 'a', 'b')
    protected = pl.Series('group', membership) if backend == 'polars' else pa.array(membership)
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        value = exp.model_fairness(protected, 'a', verbose=False)
        assert value.plot(type=kind, show=False, **options(kind)) is not None
    assert seen and exp.data is frame


@pytest.mark.parametrize('metrics', [[], None, 'TPR'])
def test_empty_or_ambiguous_metric_selection_is_rejected(metrics):
    with pytest.raises(ParameterCheckError, match='nonempty sequence'):
        objects('polars')[0].plot(type='radar', metrics=metrics, show=False, verbose=False)


def test_heatmap_keeps_model_with_all_selected_metrics_missing():
    pair = objects('polars')
    pair[1].parity_loss = pair[1].parity_loss.drop(['TPR', 'ACC'])
    figure = pair[0].plot(pair[1], type='heatmap', metrics=['TPR', 'ACC'], show=False, verbose=False)
    assert list(figure.data[0].y) == ['second', 'first']
    assert np.isnan(np.asarray(figure.data[0].z)[1]).all()
