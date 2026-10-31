"""Association algorithms and Aspect lifecycles against pandas/SciPy references."""
import pickle
from types import SimpleNamespace
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from scipy.stats import chi2_contingency, kruskal
import dalex as dx
from dalex.aspect import utils
from test_native_profile_workflows import native_computation_only


@pytest.mark.parametrize('constant', [False, True])
def test_dendrogram_hover_interpolation_preserves_vertices_without_renderer(constant, monkeypatch):
    import plotly.graph_objects as go
    from dalex.aspect.plot import _add_points_on_dendrogram_traces
    def forbidden(*args, **kwargs):
        raise AssertionError('Hover geometry invoked a browser renderer')
    monkeypatch.setattr(go.Figure, 'full_figure_for_development', forbidden)
    x = [1.] * 5 if constant else [0., 1., 1., 1., 0.]
    y = [0.] * 5 if constant else [0., 0., .5, 1., 1.]
    original = list(zip(x, y))
    fig = go.Figure(go.Scatter(x=x, y=y, text=[None, None, 'middle', None, None]))
    result = _add_points_on_dendrogram_traces(fig)
    actual = list(zip(result.data[0].x, result.data[0].y))
    assert all(vertex in actual for vertex in original)
    assert actual[0] == original[0] and actual[-1] == original[-1]
    assert sum(text == 'middle' for text in result.data[0].text) == 1


def reference(data, method):
    numeric = list(data.corr(method, numeric_only=True).columns)
    names = numeric + list(set(data.columns) - set(numeric))
    output = pd.DataFrame(np.nan, index=names, columns=names)
    output.loc[numeric, numeric] = abs(data[numeric].corr(method))
    n = len(data)
    for left in names[len(numeric):]:
        for right in names[len(numeric):]:
            if left == right:
                value = 1.
            else:
                table = pd.crosstab(data[left], data[right])
                r, c = table.shape
                phi = max(0, chi2_contingency(table)[0] / n - ((r-1)*(c-1))/(n-1))
                value = np.sqrt(phi / (min(r-(r-1)**2/(n-1), c-(c-1)**2/(n-1))-1))
            output.loc[left, right] = np.clip(value, 0, 1)
        for right in numeric:
            samples = [part[right].values for _, part in data.groupby(left)]
            value = (kruskal(*samples, nan_policy='omit').statistic - len(samples) + 1) / (n-len(samples))
            output.loc[left, right] = output.loc[right, left] = np.clip(value, 0, 1)
    output = output.fillna(0.)
    for name in names:
        output.loc[name, name] = 1.
    return output


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
def test_native_aspect_statistics_and_lifecycle(backend, method, monkeypatch):
    rng = np.random.default_rng(27)
    original = pl.DataFrame({'x': rng.normal(size=36), 'y': rng.normal(size=36),
                             'binary': np.tile([True, False], 18),
                             'kind': ['a', 'a', 'b', None] * 9,
                             'other': ['d', 'c', 'c', 'd'] * 9})
    expected = reference(original.to_pandas(), method)
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with native_computation_only(monkeypatch, backend):
        aspect = dx.Aspect(SimpleNamespace(data=data), corr_method=method)
        native = aspect.result_as(table='depend_matrix')
        assert isinstance(native, type(data))
        groups = aspect.get_aspects(h=.5)
        assert sorted(name for group in groups.values() for name in group) == sorted(original.columns)
        assert '_compatibility_depend_matrix' not in aspect.__dict__
        assert '_compatibility__dendrogram_aspects_ordered' not in aspect.__dict__
    pd.testing.assert_frame_equal(aspect.depend_matrix, expected, check_exact=False, atol=1e-12, rtol=1e-12)
    import plotly.graph_objects as go
    def forbidden_renderer(*args, **kwargs):
        raise AssertionError('Aspect dendrogram must not require an external renderer')
    monkeypatch.setattr(go.Figure, 'full_figure_for_development', forbidden_renderer)
    assert aspect.plot_dendrogram(show=False) is not None
    restored = pickle.loads(pickle.dumps(aspect))
    pd.testing.assert_frame_equal(restored.depend_matrix, expected, check_exact=False, atol=1e-12, rtol=1e-12)
    # Public pandas edits remain authoritative in later exports and minima.
    aspect.depend_matrix.loc['x', 'y'] = aspect.depend_matrix.loc['y', 'x'] = .8
    assert utils.get_min_depend_from_matrix(aspect._native_depend_matrix(), [['x', 'y']])[1] == [.8]
    assert aspect.result_as('pandas', table='depend_matrix').loc['x', 'y'] == .8


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_native_aspect_constant_and_nanosecond_categories(backend, monkeypatch):
    original = pl.DataFrame({'x': [1., 1., 1., 1., 1., 1.], 'y': [1., 2., 3., 4., 5., 6.],
                             'time': pd.date_range('2020', periods=6, freq='ns', tz='UTC'),
                             'kind': ['a', 'b'] * 3})
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with native_computation_only(monkeypatch, backend), pytest.warns(UserWarning, match='NaNs'):
        aspect = dx.Aspect(SimpleNamespace(data=data))
        values = aspect.result_as('pandas', table='depend_matrix') if backend == 'pandas' else aspect.result_as(table='depend_matrix')
    # No datetime conversion is allowed to collapse the six categories into one.
    assert aspect.depend_matrix.loc['time', 'kind'] == 0.
    np.testing.assert_array_equal(np.diag(aspect.depend_matrix), np.ones(4))
    assert np.isfinite(aspect.linkage_matrix).all()


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_unused_categories_follow_legacy_groupby_policy(backend, monkeypatch):
    original = pl.DataFrame({'x': [1., 2., 3., 4., 5., 6.],
                             'kind': pl.Series(['a', 'b'] * 3).cast(pl.Enum(['a', 'b', 'unused']))})
    legacy = original.to_pandas()
    expected = reference(legacy, 'spearman')
    data = legacy if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with native_computation_only(monkeypatch, backend):
        aspect = dx.Aspect(SimpleNamespace(data=data))
    pd.testing.assert_frame_equal(aspect.depend_matrix, expected, check_exact=False, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_high_cardinality_categorical_association_avoids_dense_crosstab(backend, monkeypatch):
    values = [str(i) for i in range(4000)] * 3
    original = pl.DataFrame({'left': values, 'right': ['v' + value for value in values]})
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with native_computation_only(monkeypatch, backend):
        aspect = dx.Aspect(SimpleNamespace(data=data))
    assert aspect.depend_matrix.loc['left', 'right'] == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_all_missing_category_has_no_join_keys(backend, monkeypatch):
    original = pl.DataFrame({'x': [1., 2., 3.], 'absent': [None, None, None]})
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with native_computation_only(monkeypatch, backend), pytest.warns(UserWarning, match='NaNs'):
        aspect = dx.Aspect(SimpleNamespace(data=data))
    assert aspect.depend_matrix.loc['x', 'absent'] == 0.
    assert aspect.depend_matrix.loc['absent', 'absent'] == 1.


def test_enum_levels_do_not_call_removed_polars_category_api(monkeypatch):
    import narwhals.stable.v2 as nw
    from dalex._dataframe import categorical_levels
    series = nw.from_native(pl.Series('value', ['a', 'a'], dtype=pl.Enum(['a', 'unused'])), series_only=True)
    def forbidden(*args, **kwargs):
        raise AttributeError('Removed in Polars 2')
    monkeypatch.setattr(type(series.cat), 'get_categories', forbidden)
    assert categorical_levels(series).to_list() == ['a', 'unused']
