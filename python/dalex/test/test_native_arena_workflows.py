"""Real models -> native Arena registration -> plots -> strict JSON."""
import json
import pickle
import numpy as np
import pandas as pd
import polars as pl
import narwhals.stable.v2 as nw
import pytest
from threadpoolctl import threadpool_limits
import dalex as dx
from dalex.arena.server import serialize_json
from dalex.arena.plots import (
    MetricsContainer, BreakDownContainer, CeterisParibusContainer, FeatureImportanceContainer,
    PartialDependenceContainer, AccumulatedDependenceContainer, ROCContainer,
    FairnessCheckContainer, VariableDistributionContainer, VariableAgainstAnotherContainer,
    ShapleyValuesContainer, ShapleyValuesVariableImportanceContainer, ShapleyValuesDependenceContainer,
)
from test_native_profile_workflows import fitted_pipeline, native_computation_only


@pytest.fixture(autouse=True)
def preserve_global_rng():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def assert_json_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_json_equal(actual[name], expected[name])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_json_equal(left, right)
    elif isinstance(expected, (int, float)):
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)
    else:
        assert actual == expected


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_native_arena_model_dataset_observation_and_plots(fitted_pipeline, backend, monkeypatch):
    model, original, y, classification = fitted_pipeline
    frame = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    dataset = nw.from_native(frame).with_columns(nw.new_series('target', y, backend=nw.get_native_namespace(nw.from_native(frame)))).to_native()
    observations = frame.head(2) if backend != 'arrow' else frame.slice(0, 2)
    with threadpool_limits(limits=1):
        exp = dx.Explainer(model, frame, y, label='pipeline', verbose=False)
        arena = dx.Arena(precalculate=False)
        for option in ['PartialDependence', 'AccumulatedDependence', 'CeterisParibus']:
            arena.set_option(option, 'grid_points', 4)
        for option in ['PartialDependence', 'AccumulatedDependence']:
            arena.set_option(option, 'N', 12)
        arena.set_option('VariableImportance', 'N', 12)
        arena.set_option('VariableImportance', 'B', 3)
        observed = []
        with native_computation_only(monkeypatch, backend):
            arena.push_model(exp)
            arena.push_observations(observations)
            arena.push_dataset(dataset, target='target', label='data')
            params = {'model': arena.get_params('model')[0], 'dataset': arena.get_params('dataset')[0],
                      'observation': arena.get_params('observation')[0],
                      'variable': arena.find_param_value('variable', 'x')}
            assert isinstance(params['dataset'].dataset, type(frame))
            assert params['variable'].type == 'numeric'
            assert 'when' in [entry['name'] for entry in arena.get_params_attributes()['observation']]
            classes = [MetricsContainer, BreakDownContainer, CeterisParibusContainer, FeatureImportanceContainer,
                       PartialDependenceContainer, AccumulatedDependenceContainer, ROCContainer,
                       VariableDistributionContainer, VariableAgainstAnotherContainer]
            for cls in classes:
                np.random.seed(42)
                plot = cls(arena).fit(params)
                serialized = json.loads(serialize_json(plot.serialize()))
                assert serialized['data']
                if cls != ROCContainer or classification:
                    assert serialized['plotComponent'] != 'Message'
                observed.append((cls, 'x', serialized['data']))
            category = arena.find_param_value('variable', 'kind')
            params['variable'] = category
            for cls in [FairnessCheckContainer, VariableDistributionContainer, VariableAgainstAnotherContainer]:
                plot = cls(arena).fit(params)
                serialized = json.loads(serialize_json(plot.serialize()))
                assert serialized['data'] and serialized['plotComponent'] != 'Message'
                observed.append((cls, 'kind', serialized['data']))
        legacy = dx.Arena(precalculate=False)
        expected_exp = dx.Explainer(model, original.to_pandas(), y, label='pipeline', verbose=False)
        legacy.push_model(expected_exp)
        legacy.push_dataset(dataset if backend == 'pandas' else nw.from_native(dataset).to_pandas(), 'target', 'data')
        legacy.push_observations(original.to_pandas().head(2))
        for option in ['PartialDependence', 'AccumulatedDependence', 'CeterisParibus']:
            legacy.set_option(option, 'grid_points', 4)
        for option in ['PartialDependence', 'AccumulatedDependence']:
            legacy.set_option(option, 'N', 12)
        legacy.set_option('VariableImportance', 'N', 12)
        legacy.set_option('VariableImportance', 'B', 3)
        expected_params = {'model': legacy.get_params('model')[0], 'dataset': legacy.get_params('dataset')[0],
                           'observation': legacy.get_params('observation')[0]}
        for cls, name, actual in observed:
            expected_params['variable'] = legacy.find_param_value('variable', name)
            np.random.seed(42)
            expected = cls(legacy).fit(expected_params)
            assert_json_equal(actual, json.loads(serialize_json(expected.data)))
        # Historical pandas display dataset/row access is lazy and remains valid.
        observation = params['observation']
        assert isinstance(observation.dataset, pd.DataFrame)
        assert isinstance(observation.get_row(), pd.Series)
        assert len(observation.get_attributes()) == len(original.columns)


def test_observation_labels_remain_unique_after_string_conversion():
    arena = dx.Arena(precalculate=False)
    with pytest.raises(Exception, match='unique'):
        arena.push_observations(pd.DataFrame({'x': [1, 2]}, index=[1, '1']))


def test_continuous_variable_attributes_do_not_revert_to_levels():
    from dalex.arena.params import VariableParam
    variable = VariableParam('x')
    variable.update_attributes(pl.Series('x', range(30)))
    variable.update_attributes(pl.Series('x', [1, 2]))
    assert variable.levels is None
    assert variable.min == 0 and variable.max == 29


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('kind', ['ShapleyValues', 'DatasetShapleyValues'])
def test_native_arena_background_shap_resources(fitted_pipeline, backend, kind, monkeypatch):
    model, original, y, classification = fitted_pipeline
    frame = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with threadpool_limits(limits=1):
        exp = dx.Explainer(model, frame, y, label='pipeline', verbose=False)
        arena = dx.Arena(precalculate=False)
        arena.set_option(kind, 'B', 3)
        arena.set_option(kind, 'cpus', 1)
        if kind == 'DatasetShapleyValues':
            arena.set_option(kind, 'N', 3)
        with native_computation_only(monkeypatch, backend):
            arena.push_model(exp)
            arena.push_observations(frame.iloc[[1]] if backend == 'pandas' else frame.slice(1, 1))
            params = {'model': arena.get_params('model')[0], 'observation': arena.get_params('observation')[0],
                      'variable': arena.find_param_value('variable', 'x')}
            np.random.seed(42)
            resource = arena.resource_manager.get_resource(kind, params)
            resource.thread.join(timeout=30)
            data, progress, done = resource.get_result()
            assert done and progress == 1
            native = resource.result_as(partition='when')
            assert isinstance(native, type(frame))
            assert len(nw.from_native(native)) == (3 if kind == 'ShapleyValues' else 9)
            classes = [ShapleyValuesContainer] if kind == 'ShapleyValues' else [
                ShapleyValuesVariableImportanceContainer, ShapleyValuesDependenceContainer]
            for cls in classes:
                plot = cls(arena).fit(params)
                assert plot.plot_component != 'Message'
                assert json.loads(serialize_json(plot.serialize()))['data']
            assert not data._legacy
            restored_data = pickle.loads(pickle.dumps(data))
            assert not restored_data._legacy
            assert type(restored_data.native('result')) is type(data.native('result'))
        published = data['result']
        assert isinstance(data, dict)
        assert isinstance(dict(data)['result'], pd.DataFrame)
        assert isinstance(published, pd.DataFrame)
        assert len(published) == len(original.columns) * (3 if kind == 'ShapleyValues' else 9)
        reference_data = original.to_pandas()
        reference = dx.Explainer(model, reference_data, y, label='pipeline', verbose=False)
        np.random.seed(42)
        expected_parts = []
        if kind == 'ShapleyValues':
            for _ in range(3):
                value = reference.predict_parts(reference_data.iloc[[1]], type='shap', B=1, processes=1).result
                expected_parts.append(value.loc[value.B != 0])
        else:
            ids = np.random.choice(len(original), 3, replace=False)
            for row, index in enumerate(ids):
                value = reference.predict_parts(reference_data.iloc[[index]], type='shap', B=3, processes=1).result
                value = value.loc[value.B != 0].copy()
                value['row'] = row
                expected_parts.append(value)
        expected = pd.concat(expected_parts)
        pd.testing.assert_frame_equal(published, expected, check_exact=False, rtol=1e-10, atol=1e-10)
        if kind == 'ShapleyValues':
            assert isinstance(data['stats'], pd.DataFrame)
            from dalex.arena.resources._shapley_values_resource import q1, q3
            # Each feature has one value for this observation. Grouping its
            # typed metadata alongside unrelated timestamps/numbers made the
            # old pandas resource fail while sorting heterogeneous values.
            valid = expected.loc[expected.variable_value.notna()]
            expected_stats = valid.groupby('variable_name').agg(
                {'contribution': ['mean', 'max', 'min', q1, q3]}).contribution
            expected_stats['abs'] = expected_stats['mean'].abs()
            expected_stats['variable_value'] = valid.groupby('variable_name').variable_value.first()
            expected_stats = expected_stats.set_index('variable_value', append=True)
            pd.testing.assert_frame_equal(data['stats'], expected_stats, check_exact=False, rtol=1e-10, atol=1e-10)
        published.loc[:, 'contribution'] = 5.
        assert (nw.from_native(resource.result_as(partition='x'))['contribution'].to_numpy() == 5.).all()
        times = nw.from_native(resource.result_as('pyarrow', partition='when'))['variable_value'].cast(nw.Int64).to_list()
        expected_times = [value.value for value in expected.loc[expected.variable_name == 'when', 'variable_value']]
        assert times == expected_times
