"""Arena's public JSON stays valid when display metadata contains missing values."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx


def strict_json(text):
    def reject(value):
        raise AssertionError(f'Invalid JSON numeric literal: {value}')
    return json.loads(text, parse_constant=reject)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_arena_http_missing_attributes_are_json_null_without_mutating_inputs(backend):
    import requests
    from dalex.arena.static import get_free_port
    original = pl.DataFrame({'x': [np.nan, None, 3.], 'z': [np.inf, -np.inf, 2.]})
    frame = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    arena = dx.Arena(precalculate=False)
    arena.push_observations(frame)
    port = get_free_port()
    assert port is not None
    arena.run_server(host='127.0.0.1', port=port)
    try:
        for row in [0, 1]:
            response = requests.get(f'http://127.0.0.1:{port}/attribute/observation/{row}', timeout=10)
            assert response.status_code == 200
            assert strict_json(response.text) == {'x': None, 'z': None}
    finally:
        arena.stop_server()
    if backend == 'polars':
        assert frame['x'].null_count() == 1 and frame['x'].is_nan().sum() == 1
    elif backend == 'arrow':
        assert frame['x'].null_count == 1 and np.isnan(frame['x'].to_numpy()[0])
    else:
        pd.testing.assert_frame_equal(frame, original.to_pandas())


def test_static_arena_export_sanitizes_nested_numpy_display_values():
    from dalex.arena.static import get_json
    values = np.array([1., np.nan, np.inf, -np.inf])
    plot = SimpleNamespace(serialize=lambda: {'values': values, 'integer': np.int64(2), 'extra': (pd.NA, pd.NaT)})
    arena = SimpleNamespace(plots_manager=SimpleNamespace(fill_cache=lambda: None, cache=[plot]),
                            list_available_params=lambda: {}, get_params_attributes=lambda: {})
    result = strict_json(get_json(arena))
    assert result['data'] == [{'values': [1., None, None, None], 'integer': 2, 'extra': [None, None]}]
    assert np.isnan(values[1]) and np.isposinf(values[2]) and np.isneginf(values[3])
