"""Typed heterogeneous Aspect metadata, legacy edits and cache invalidation."""
import pickle
import numpy as np
import pandas as pd
import polars as pl
import pytest
import dalex as dx
from dalex.aspect import PredictAspectImportance
from test_native_profile_workflows import native_computation_only


class TypedModel:
    def predict(self, data):
        values = data['big'].to_list() if isinstance(data, (pl.DataFrame, pd.DataFrame)) else data['big'].to_pylist()
        return np.asarray([0. if value is None or value is pd.NA else float(int(value) % 97) for value in values])


@pytest.mark.parametrize('backend', ['polars', 'arrow'])
@pytest.mark.parametrize('kind', ['default', 'shap'])
def test_native_aspect_structs_retain_mixed_scalar_precision(backend, kind, monkeypatch):
    original = pl.DataFrame({'big': pl.Series([2**63 + 13, 2**63 + 29, None, 2**63 + 31], dtype=pl.UInt64),
                             'time': pd.date_range('2020', periods=4, freq='ns', tz='UTC'),
                             'kind': ['a', 'b', 'b', 'a']})
    data = original if backend == 'polars' else original.to_arrow()
    observation = data.slice(1, 1)
    exp = dx.Explainer(TypedModel(), data, TypedModel().predict(data), model_type='regression', verbose=False)
    groups = {'mixed': ['big', 'time', 'kind']}
    matrix = pl.DataFrame(np.eye(3), schema=original.columns)
    matrix = matrix if backend == 'polars' else matrix.to_arrow()
    with native_computation_only(monkeypatch, backend):
        result = PredictAspectImportance(groups, type=kind, N=4, B=3, random_state=42, _depend_matrix=matrix)
        result.fit(exp, observation)
        native = result.result_as('pyarrow')
        assert native['variable_values'].chunk(0).field('big')[0].as_py() == 2**63 + 29
        assert native['variable_values'].chunk(0).field('time').cast('int64')[0].as_py() == original['time'].cast(pl.Int64)[1]
        restored = pickle.loads(pickle.dumps(result))
        assert restored.result_as('pyarrow')['variable_values'].equals(native['variable_values'])
    result.result.loc[0, 'importance'] = 9.
    edited = result.result_as('pyarrow')
    assert edited['importance'][0].as_py() == 9.
    assert edited['variable_values'].chunk(0).field('time').cast('int64')[0].as_py() == original['time'].cast(pl.Int64)[1]
    assert edited['variable_values'].chunk(0).field('big')[0].as_py() == 2**63 + 29


def test_model_triplot_cache_requires_same_random_seed(monkeypatch):
    from sklearn.linear_model import LinearRegression
    from dalex.aspect._model_aspect_importance.object import ModelAspectImportance
    rng = np.random.default_rng(52)
    data = pd.DataFrame(rng.normal(size=(32, 3)), columns=['a', 'b', 'c'])
    y = data.a.to_numpy() + data.b.to_numpy() * 2
    model = LinearRegression().fit(data, y)
    aspect = dx.Aspect(dx.Explainer(model, data, y, verbose=False))
    aspect.model_triplot(N=20, B=3, random_state=42)
    def forbidden(*args, **kwargs):
        raise AssertionError('Expected matching triplot cache to be reused')
    with monkeypatch.context() as patch:
        patch.setattr(ModelAspectImportance, 'fit', forbidden)
        aspect.model_parts(N=20, B=3, random_state=42)
    calls = []
    original = ModelAspectImportance.fit
    def record(self, explainer):
        calls.append(self.random_state)
        return original(self, explainer)
    monkeypatch.setattr(ModelAspectImportance, 'fit', record)
    aspect.model_parts(N=20, B=3, random_state=43)
    assert calls == [43]
