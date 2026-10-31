"""Base Aspect dendrograms keep native metadata through plotting and pickle."""
import json
import pickle

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from threadpoolctl import threadpool_limits

import dalex as dx
from test_native_plot_preparation import (FormulaModel, assert_figures_equal,
                                           frame_as, no_legacy_materialization)
from test_native_profile_workflows import fitted_pipeline, native_computation_only


def predict_formula(model, batch):
    return np.nan_to_num(nw.from_native(batch).to_numpy()).sum(axis=1)


class DifferenceResidual:
    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self, model, batch, target):
        return target - self.predictor(model, batch)


class PipelinePrediction:
    def __init__(self, backend, classification):
        self.backend = backend
        self.classification = classification

    def __call__(self, model, batch):
        assert type(batch) is self.backend
        return model.predict_proba(batch)[:, 1] if self.classification else model.predict(batch)


def aspect_for(backend, kind):
    values = np.random.default_rng(41).normal(size=(16, 3))
    if kind == 'tied':
        values[:] = np.arange(16)[:, None]
    elif kind == 'constant':
        values[:] = 0
    elif kind == 'missing':
        values[:] = np.nan
    data = frame_as(pl.DataFrame(values, schema=['z', 'a', 'b']), backend)
    exp = dx.Explainer(FormulaModel(), data, np.zeros(16), predict_function=predict_formula,
                       residual_function=DifferenceResidual(predict_formula),
                       label='formula', model_type='regression', verbose=False)
    return dx.Aspect(exp)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['random', 'tied', 'constant', 'missing'])
def test_base_dendrogram_payload_and_pickle_preserve_native_storage(monkeypatch, backend, kind):
    reference = aspect_for('pandas', kind)
    expected = json.loads(reference.plot_dendrogram(show=False, digits=2).to_json())
    value = aspect_for(backend, kind)
    before = value.result_as('pyarrow', table='_dendrogram_aspects_ordered')
    with no_legacy_materialization(monkeypatch):
        actual = json.loads(value.plot_dendrogram(show=False, digits=2).to_json())
        restored = pickle.loads(pickle.dumps(value))
        assert_figures_equal(json.loads(restored.plot_dendrogram(show=False, digits=2).to_json()), actual)
    assert_figures_equal(actual, expected)
    assert before.equals(value.result_as('pyarrow', table='_dendrogram_aspects_ordered'))
    assert '_compatibility__dendrogram_aspects_ordered' not in value.__dict__
    assert '_compatibility_depend_matrix' not in value.__dict__


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['random', 'tied', 'constant', 'missing'])
def test_base_dendrogram_fit_to_plot_has_no_pandas_construction(monkeypatch, backend, kind):
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        value = aspect_for(backend, kind)
        assert value.plot_dendrogram(show=False) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_explicit_dendrogram_metadata_edits_are_authoritative(monkeypatch, backend):
    value = aspect_for(backend, 'random')
    legacy = value._dendrogram_aspects_ordered
    legacy.index = pd.Index(['same'] * len(legacy))
    legacy['min_depend'] = pd.Series(np.repeat(.12345, len(legacy)), index=legacy.index, dtype='float32')
    legacy['variable_names'] = [['edited', 'names']] * len(legacy)
    legacy['vars_min_depend'] = [['edited', 'pair']] * len(legacy)
    seen = []

    def rounding(number, digits):
        seen.append(type(number))
        return np.around(number, digits)

    with no_legacy_materialization(monkeypatch):
        figure = value.plot_dendrogram(show=False, digits=2, rounding_function=rounding)
    assert seen[0] is np.float32
    assert any('0.12' in trace.hovertext and 'edited, pair' in trace.hovertext and
               'edited<br>names' in trace.hovertext for trace in figure.data)
    restored = pickle.loads(pickle.dumps(value))
    assert_figures_equal(json.loads(restored.plot_dendrogram(show=False, digits=2).to_json()),
                         json.loads(figure.to_json()))


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
def test_real_pipeline_base_dendrogram_fit_export_plot_pickle(fitted_pipeline, backend, monkeypatch):
    model, original, target, classification = fitted_pipeline
    native = original if backend == 'polars' else original.to_arrow()

    def explain(data):
        predict = PipelinePrediction(type(data), classification)
        exp = dx.Explainer(model, data, target, predict_function=predict, verbose=False,
                           residual_function=DifferenceResidual(predict),
                           model_type='classification' if classification else 'regression')
        return dx.Aspect(exp)

    with threadpool_limits(limits=1):
        reference = explain(original.to_pandas())
        expected = json.loads(reference.plot_dendrogram(show=False).to_json())
        with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
            value = explain(native)
            actual = json.loads(value.plot_dendrogram(show=False).to_json())
            exported = value.result_as('pyarrow', table='_dendrogram_aspects_ordered')
            restored = pickle.loads(pickle.dumps(value))
            assert restored.result_as('pyarrow', table='_dendrogram_aspects_ordered').equals(exported)
            assert_figures_equal(json.loads(restored.plot_dendrogram(show=False).to_json()), actual)
    assert_figures_equal(actual, expected)
