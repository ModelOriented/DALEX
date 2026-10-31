"""Real sklearn/LightGBM -> local/global surrogates -> exports/pickle/plots."""
import pickle
import numpy as np
import pandas as pd
import polars as pl
import narwhals.stable.v2 as nw
import pytest
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from lime.explanation import Explanation as LimeExplanation
from threadpoolctl import threadpool_limits
import dalex as dx
from test_native_profile_workflows import fitted_pipeline, native_computation_only


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('kind', ['tree', 'linear', 'lime'])
def test_native_surrogates_real_models(fitted_pipeline, backend, kind, monkeypatch):
    pipeline, original, y, classification = fitted_pipeline
    numeric = original.select('x', 'z')
    model = clone(pipeline.steps[-1][1])
    data = numeric.to_pandas() if backend == 'pandas' else numeric if backend == 'polars' else numeric.to_arrow()
    with threadpool_limits(limits=1):
        model.fit(numeric, y)
        exp = dx.Explainer(model, data, y, model_type='classification' if classification else 'regression', verbose=False)
        reference = dx.Explainer(model, numeric.to_pandas(), y,
                                 model_type='classification' if classification else 'regression', verbose=False)
        with native_computation_only(monkeypatch, backend):
            if kind == 'lime':
                observation = data.head(1) if backend != 'arrow' else data.slice(0, 1)
                result = exp.predict_surrogate(observation, random_state=42, num_samples=80, num_features=2)
                assert isinstance(result, LimeExplanation)
                if backend != 'pandas':
                    assert isinstance(result.result_as(), type(data))
                    assert '_compatibility_result' not in result.__dict__
            else:
                kwargs = {'random_state': 42} if kind == 'tree' or classification else {}
                result = exp.model_surrogate(type=kind, max_vars=1, **kwargs)
                expected_class = (DecisionTreeClassifier if classification else DecisionTreeRegressor) if kind == 'tree' else (
                    LogisticRegression if classification else LinearRegression)
                assert isinstance(result, expected_class)
                if backend == 'pandas':
                    assert type(result) is expected_class
                else:
                    assert isinstance(result.result_as(), type(data))
                    assert '_compatibility_performance' not in result.__dict__
                    selected = nw.from_native(data).select(result.feature_names).to_native()
                    assert np.isfinite(result.predict(selected)).all()
            restored = pickle.loads(pickle.dumps(result))
        if kind == 'lime':
            expected = reference.predict_surrogate(numeric.to_pandas().head(1), random_state=42,
                                                     num_samples=80, num_features=2)
            assert type(expected) is LimeExplanation
            pd.testing.assert_frame_equal(result.result, expected.result, check_exact=False, atol=1e-10, rtol=1e-10)
            pd.testing.assert_frame_equal(restored.result, result.result)
            assert result.plot(return_figure=True) is not None
            result.result.loc[0, 'effect'] = 9.
            if backend != 'pandas':
                assert nw.from_native(result.result_as())['effect'][0] == 9.
        else:
            expected = reference.model_surrogate(type=kind, max_vars=1, **kwargs)
            pd.testing.assert_frame_equal(result.performance, expected.performance, check_exact=False, atol=1e-10, rtol=1e-10)
            pd.testing.assert_frame_equal(restored.performance, result.performance)
            if kind == 'tree':
                assert result.plot(return_figure=True) is not None
            result.performance.iloc[0, 0] = 8.
            if backend != 'pandas':
                assert nw.from_native(result.result_as()).drop('label').to_numpy()[0, 0] == 8.
