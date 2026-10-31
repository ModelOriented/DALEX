"""Custom display labels retain native computation and legacy table layouts."""
import pickle
import pandas as pd
import narwhals.stable.v2 as nw
import pytest
from threadpoolctl import threadpool_limits
import dalex as dx
from test_native_profile_workflows import fitted_pipeline, native_computation_only


def explain(exp, method, label=None):
    options = {'label': label} if label else {}
    observation = exp.data.head(1) if hasattr(exp.data, 'head') else exp.data.slice(0, 1)
    if method == 'importance':
        return exp.model_parts(N=12, B=2, random_state=42, **options)
    if method in ('breakdown', 'shap'):
        return exp.predict_parts(observation, type='break_down' if method == 'breakdown' else 'shap',
                                  N=12, B=2, random_state=42, **options)
    if method == 'cp':
        return exp.predict_profile(observation, variables=['x'], grid_points=3, verbose=False, **options)
    if method in ('partial', 'conditional', 'accumulated'):
        return exp.model_profile(type=method, variables=['x'], grid_points=3, N=12,
                                  random_state=42, verbose=False, **options)
    return exp.model_performance(**options) if method == 'metrics' else exp.model_diagnostics(**options)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('method', ['importance', 'breakdown', 'shap', 'cp', 'partial',
                                    'conditional', 'accumulated', 'metrics', 'diagnostics'])
def test_real_model_custom_labels_preserve_lazy_results(fitted_pipeline, backend, method, monkeypatch):
    model, original, y, classification = fitted_pipeline
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with threadpool_limits(limits=1):
        exp = dx.Explainer(model, data, y, label='fitted', verbose=False,
                           model_type='classification' if classification else 'regression')
        reference = dx.Explainer(model, original.to_pandas(), y, label='fitted', verbose=False,
                                 model_type=exp.model_type)
        with native_computation_only(monkeypatch, backend):
            result = explain(exp, method, label='custom')
            assert '_compatibility_result' not in result.__dict__
            partition = 'x' if method in ('shap', 'cp', 'partial', 'conditional', 'accumulated') else None
            native = nw.from_native(result.result_as(partition=partition))
            name = '_label_' if method in ('cp', 'partial', 'conditional', 'accumulated') else 'label'
            assert native[name].unique().to_list() == ['custom']
        expected = explain(reference, method).result.copy()
        expected[name] = 'custom'  # Historical label override keeps the old index.
        pd.testing.assert_frame_equal(result.result, expected, check_exact=False, rtol=1e-10, atol=1e-10)
        pd.testing.assert_frame_equal(pickle.loads(pickle.dumps(result)).result, expected,
                                      check_exact=False, rtol=1e-10, atol=1e-10)
