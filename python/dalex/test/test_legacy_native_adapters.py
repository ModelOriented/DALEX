"""Golden upstream regressions discovered while consolidating private adapters."""
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
import sys
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression

import dalex as dx


@pytest.mark.parametrize('kind,names,expected', [
    ('default', ['za', 'b'], [1.5486937380839365, 1.1311803886687215]),
    ('shap', ['b', 'za'], [.3610502815084848, .24887295092506165]),
])
def test_pandas_aspect_retains_historical_callback_indexes(kind, names, expected):
    data = pd.DataFrame(np.random.default_rng(682).normal(size=(32, 3)), columns=['z', 'a', 'b'],
                         index=np.arange(32) * 3 + 7)
    target = data.z.to_numpy() * 2 + data.a.to_numpy() / 3 + np.sin(data.b.to_numpy())
    model = LinearRegression().fit(data, target)
    indexes = []

    def predict(model, frame):
        indexes.append(frame.index.to_numpy().copy())
        return frame.to_numpy().sum(axis=1) + frame.index.to_numpy() / 10

    exp = dx.Explainer(model, data, target, predict_function=predict, model_type='regression', verbose=False)
    result = dx.Aspect(exp).predict_parts(data.iloc[[2]], type=kind, N=20, B=3, random_state=42,
        variable_groups={'za': np.array(['z', 'a']), 'b': np.array(['b'])}).result
    assert result['aspect_name'].tolist() == names
    np.testing.assert_allclose(result['importance'], expected, rtol=1e-12, atol=1e-12)
    assert any(np.array_equal(index, np.arange(20)) for index in indexes)


class NonlinearModel:
    def predict(self, data):
        values = nw.from_native(data, eager_only=True).to_numpy()
        return values[:, 0] * values[:, 1] + np.sin(values[:, 2] * values[:, 3]) + np.cos(values[:, 4]) * values[:, 5]**2


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_nonlinear_triplot_advances_one_serial_rng_across_cuts(backend):
    rng = np.random.default_rng(123)
    rng.normal(size=(40, 3))  # Same preceding fixture draw as the upstream probe.
    data = pl.DataFrame(rng.normal(size=(40, 6)), schema=[f'k{i}' for i in range(6)])
    frame = data.to_pandas() if backend == 'pandas' else data if backend == 'polars' else data.to_arrow()
    model = NonlinearModel()
    exp = dx.Explainer(model, frame, model.predict(frame), verbose=False)
    explanation = dx.Aspect(exp).predict_triplot(nw.from_native(frame).head(1).to_native(),
                                                 type='shap', N=20, B=3, random_state=42)
    # Untouched upstream, seed 42. Resetting the RNG at each cut changes a
    # contribution by 0.45745 while linear-model parity tests still pass.
    expected = [-.8159792175992177, .3017737920101344, .5522206190638511,
                -.9388428131081086, -.15789490567471964]
    np.testing.assert_allclose(explanation.result['importance'], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('aggregation,score', [('min', .2), ('max', .8), ('mean', .5)])
def test_pps_external_boundary_restores_native_matrix(monkeypatch, backend, aggregation, score):
    from dalex.aspect.utils import calculate_pps_matrix
    data = pl.DataFrame({'b': [1., 2., 3.], 'a': [3., 1., 2.]})
    frame = data.to_pandas() if backend == 'pandas' else data if backend == 'polars' else data.to_arrow()
    seen = []

    def external_matrix(batch, sample):
        assert isinstance(batch, pd.DataFrame) and sample is None
        seen.append(batch)
        return pd.DataFrame({'x': ['b', 'a', 'b', 'a'], 'y': ['b', 'b', 'a', 'a'],
                              'ppscore': [1., .2, .8, 1.]})

    monkeypatch.setitem(sys.modules, 'ppscore', SimpleNamespace(matrix=external_matrix))
    result = calculate_pps_matrix(frame, aggregation, native=True)
    assert type(result.data) is type(frame)
    legacy = result._dalex_pandas_view()
    pd.testing.assert_frame_equal(legacy, pd.DataFrame([[1., score], [score, 1.]],
                                                      columns=['a', 'b'], index=['a', 'b']))
    assert seen
