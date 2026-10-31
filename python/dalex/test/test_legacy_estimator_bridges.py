"""Native DALEX workflows with an explicit model-owned numerical input bridge.

This supports estimators predating native dataframe input and is separate from
the direct native-estimator integrations. DALEX still passes the original frame
to the predictor; the fitted pipeline controls its own matrix conversion.
"""
import json
import pickle

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

import dalex as dx
from test_native_plot_preparation import assert_figures_equal, no_legacy_materialization
from test_native_profile_workflows import native_computation_only


def numerical_model_input(frame):
    return np.ascontiguousarray(nw.from_native(frame, eager_only=True).to_numpy())


class Prediction:
    def __init__(self, backend, classification):
        self.backend = backend
        self.classification = classification

    def __call__(self, model, frame):
        assert type(frame) is self.backend
        return model.predict_proba(frame)[:, 1] if self.classification else model.predict(frame)


class Residual:
    def __init__(self, prediction):
        self.prediction = prediction

    def __call__(self, model, frame, target):
        return target - self.prediction(model, frame)


def explanations(exp, frame):
    observation = nw.from_native(frame, eager_only=True).head(1).to_native()
    settings = dict(N=16, B=2, random_state=42)
    values = {'importance': exp.model_parts(**settings),
              'breakdown': exp.predict_parts(observation, type='break_down', N=16, random_state=42),
              'shap': exp.predict_parts(observation, type='shap', **settings),
              'cp': exp.predict_profile(observation, variables=['x'], grid_points=3, verbose=False),
              'performance': exp.model_performance(), 'diagnostics': exp.model_diagnostics()}
    for kind in ['partial', 'conditional', 'accumulated']:
        values[kind] = exp.model_profile(type=kind, variables=['x'], N=16, grid_points=3,
                                          random_state=42, verbose=False)
    aspect = dx.Aspect(exp)
    groups = {'pair': ['x', 'z'], 'q': ['q']}
    values['model_aspect'] = aspect.model_parts(variable_groups=groups, **settings)
    values['predict_aspect'] = aspect.predict_parts(observation, variable_groups=groups, **settings)
    values['model_triplot'] = aspect.model_triplot(**settings)
    values['predict_triplot'] = aspect.predict_triplot(observation, type='shap', **settings)
    return aspect, values


def figure(value, name):
    options = {'widget': False} if name.endswith('triplot') else {'smooth': False} if name == 'diagnostics' else {}
    return json.loads(value.plot(show=False, **options).to_json())


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('classification', [False, True])
def test_legacy_lightgbm_matrix_bridge_complete_native_lifecycle(monkeypatch, backend, classification):
    rng = np.random.default_rng(857)
    original = pl.DataFrame(rng.normal(size=(64, 3)), schema=['x', 'z', 'q'])
    response = original['x'].to_numpy() * 2 - original['z'].to_numpy() + original['q'].to_numpy()**2
    target = (response > np.median(response)).astype(int) if classification else response
    native = original if backend == 'polars' else original.to_arrow()
    estimator = (LGBMClassifier if classification else LGBMRegressor)(
        n_estimators=8, n_jobs=1, random_state=42, verbosity=-1)
    with threadpool_limits(limits=1):
        model = make_pipeline(FunctionTransformer(numerical_model_input, validate=False), estimator).fit(original, target)

        def explainer(frame):
            prediction = Prediction(type(frame), classification)
            return dx.Explainer(model, frame, target, predict_function=prediction,
                residual_function=Residual(prediction), model_type='classification' if classification else 'regression',
                label='bridge', verbose=False)

        pandas_frame = original.to_pandas()
        _, reference = explanations(explainer(pandas_frame), pandas_frame)
        expected_figures = {name: figure(value, name) for name, value in reference.items()}
        with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
            exp = explainer(native)
            assert exp.data is native
            aspect, actual = explanations(exp, native)
            assert aspect.plot_dendrogram(show=False) is not None
            for name, value in actual.items():
                exported = value.result_as('pyarrow')
                assert_figures_equal(figure(value, name), expected_figures[name])
                restored = pickle.loads(pickle.dumps(value))
                assert restored.result_as('pyarrow').equals(exported)
                assert_figures_equal(figure(restored, name), expected_figures[name])
                assert '_compatibility_result' not in value.__dict__
    for name, value in actual.items():
        left, right = value.result.select_dtypes(include=np.number), reference[name].result.select_dtypes(include=np.number)
        assert list(left.columns) == list(right.columns)
        np.testing.assert_allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=1e-12, atol=1e-12)
        value.result.loc[value.result.index[0], left.columns[0]] += .125
        pd.testing.assert_frame_equal(value.result_as('pandas'), value.result)
