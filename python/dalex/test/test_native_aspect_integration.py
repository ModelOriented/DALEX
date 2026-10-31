"""Real sklearn/LightGBM preprocessing through native Aspect fits and figures."""
import json
import pickle

import narwhals.stable.v2 as nw
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import dalex as dx
from test_native_aspect_plotting import FAMILIES, options
from test_native_plot_preparation import assert_figures_equal, no_legacy_materialization
from test_native_profile_workflows import fitted_pipeline, native_computation_only


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('family', FAMILIES)
def test_real_pipeline_aspect_fit_export_plot_and_pickle(fitted_pipeline, backend, family, monkeypatch):
    model, original, target, classification = fitted_pipeline
    frame = original if backend == 'polars' else original.to_arrow()
    groups = {'numeric': ['x', 'z'], 'kind': ['kind'], 'when': ['when'], 'group': ['group']}
    seen = []

    def explain(data):
        def predict(model, batch):
            assert type(batch) is type(data)
            seen.append(type(batch))
            return model.predict_proba(batch)[:, 1] if classification else model.predict(batch)
        exp = dx.Explainer(model, data, target, predict_function=predict, label='pipeline',
                           model_type='classification' if classification else 'regression', verbose=False)
        aspect = dx.Aspect(exp)
        settings = dict(N=16, B=2, random_state=42)
        if family == 'model':
            value = aspect.model_parts(variable_groups=groups, **settings)
        elif family == 'model_triplot':
            value = aspect.model_triplot(**settings)
        else:
            observation = nw.from_native(data).head(1).to_native()
            settings['type'] = 'shap' if family.startswith('shap') else 'default'
            value = (aspect.predict_triplot(observation, **settings) if family.endswith('triplot') else
                     aspect.predict_parts(observation, variable_groups=groups, **settings))
        assert exp.data is data
        return value

    with threadpool_limits(limits=1):
        reference = explain(original.to_pandas())
        expected_numbers = reference.result.select_dtypes(include=np.number).to_numpy(dtype=float)
        expected_figure = json.loads(reference.plot(show=False, **options(family)).to_json())
        with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
            value = explain(frame)
            exported = value.result_as('pyarrow')
            actual_figure = json.loads(value.plot(show=False, **options(family)).to_json())
            restored = pickle.loads(pickle.dumps(value))
            assert restored.result_as('pyarrow').equals(exported)
            assert_figures_equal(json.loads(restored.plot(show=False, **options(family)).to_json()), actual_figure)
    np.testing.assert_allclose(value.result.select_dtypes(include=np.number).to_numpy(dtype=float),
                               expected_numbers, atol=1e-12, rtol=1e-12, equal_nan=True)
    assert_figures_equal(actual_figure, expected_figure)
    assert type(frame) in seen
