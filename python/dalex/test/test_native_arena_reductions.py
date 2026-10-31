"""Analytical Arena reductions for missing values and typed memberships."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import polars as pl
import narwhals.stable.v2 as nw
import pytest
import dalex as dx
from dalex.arena._native_eda import distribution, against
from dalex.arena.resources import ShapleyValuesResource
from dalex.predict_explanations._native_contributions import ShapTable
from test_native_profile_workflows import native_computation_only


def convert(frame, backend):
    return frame.to_pandas() if backend == 'pandas' else frame if backend == 'polars' else frame.to_arrow()


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_shap_resource_skips_nan_contributions_and_missing_display_memberships(backend, monkeypatch):
    observation = convert(pl.DataFrame({'x': [2.], 'z': [np.nan]}), backend)
    raw = convert(pl.DataFrame({'variable': ['x = 2', 'z = nan'] * 3,
                                'variable_name': ['x', 'z'] * 3,
                                'contribution': [1., 10., np.nan, 20., 3., 30.],
                                'sign': [1., 1., np.nan, 1., 1., 1.],
                                'label': ['model'] * 6, 'B': [1, 1, 2, 2, 3, 3]}), backend)
    shap = SimpleNamespace(result=ShapTable(raw, observation), intercept=np.float64(0))
    resource = ShapleyValuesResource(dx.Arena(precalculate=False))
    with native_computation_only(monkeypatch, backend):
        resource.add_to_data(shap)
        stats = nw.from_native(resource.data.native('stats').data)
        assert stats['variable_name'].to_list() == ['x']
        for name, expected in {'mean': 2., 'min': 1., 'max': 3., 'q1': 1.5, 'q3': 2.5}.items():
            assert stats[name][0] == expected
    assert list(resource.data['stats'].index) == [('x', 2.)]


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_eda_empty_histograms_missing_groups_and_unused_categories(backend, monkeypatch):
    data = convert(pl.DataFrame({'x': pl.Series([np.nan, None, np.inf], dtype=pl.Float64),
                                 'kind': pl.Series(['a', None, 'a'], dtype=pl.Enum(['a', 'unused'])),
                                 'absent': [None, None, None]}), backend)
    with native_computation_only(monkeypatch, backend):
        component, histogram = distribution(data, 'x', [5])
        assert component == 'DistributionHistogram'
        assert histogram[5]['counts'] == [0] * 5 and histogram[5]['density'] == [0.] * 5
        _, counts = distribution(data, 'kind', [5])
        assert counts == {'names': ['a', 'unused'], 'count': [2, 0], 'density': [1., 0.]}
        assert all(isinstance(count, int) for count in counts['count'])
        _, missing = distribution(data, 'absent', [5])
        assert missing == {'names': [], 'count': [], 'density': []}


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_eda_boxplot_quantiles_and_fences(backend, monkeypatch):
    data = convert(pl.DataFrame({'x': [1., 2., 3., 4., 100., None], 'group': ['a']*5 + [None]}), backend)
    with native_computation_only(monkeypatch, backend):
        result = against(data, 'x', 100)['group']
        assert result['secondary'] == ['a']
        assert result['first'] == [{'q1': 2., 'q3': 4., 'mean': 22., 'median': 3.,
                                    'lf': 1., 'uf': 7., 'outliers': [100.]}]
