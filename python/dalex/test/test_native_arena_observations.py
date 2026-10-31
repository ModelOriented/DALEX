"""Arena display tables must not replace original native observation cells."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import dalex as dx
from dalex._dataframe import take_rows


class CellSensitiveModel:
    def __init__(self, unsigned=False):
        self.unsigned = unsigned

    def predict(self, batch):
        cells = batch["x"].to_list() if isinstance(batch, pl.DataFrame) else batch["x"].to_pylist()
        scores = [7. if cell is None else 19. if np.isnan(cell) else float(cell % 97) if self.unsigned else cell
                  for cell in cells]
        return np.array(scores) + batch["z"].to_numpy()


@pytest.mark.parametrize("observation_backend", ["polars", "arrow"])
@pytest.mark.parametrize("model_backend", ["polars", "arrow"])
@pytest.mark.parametrize("unsigned", [False, True])
def test_arena_retains_native_observations_across_model_backends(observation_backend, model_backend, unsigned):
    data = pl.DataFrame({"x": pl.Series([2**63 + 1, None, 2**63 + 3], dtype=pl.UInt64) if unsigned
                         else pl.Series([1., None, np.nan]), "z": [2., 5., 1.]})
    frame = data if model_backend == "polars" else data.to_arrow()
    observations = data.select("z", "x").with_columns(pl.lit("display").alias("extra"))
    observations = observations if observation_backend == "polars" else observations.to_arrow()
    model = CellSensitiveModel(unsigned)
    exp = dx.Explainer(model, frame, model.predict(frame), model_type="regression", verbose=False)
    arena = dx.Arena(precalculate=False, enable_attributes=False)
    arena.push_model(exp)
    arena.push_observations(observations)
    model_param = arena.get_params("model")[0]
    for i, observation in enumerate(arena.get_params("observation")):
        native = observation.get_row_for_model(model_param)
        assert type(native) is type(frame)
        np.testing.assert_array_equal(model.predict(native), model.predict(take_rows(frame, [i])))
        assert list(native.columns if isinstance(native, pl.DataFrame) else native.column_names) == ["x", "z"]
        attributes = observation.get_attributes()
        if i == 1:
            assert attributes['x'] is None
        elif unsigned:
            assert attributes['x'] == 2**63 + (1 if i == 0 else 3)


def test_arena_nullable_pandas_observation_keeps_exact_unsigned_cells():
    data = pl.DataFrame({"x": pl.Series([2**63 + 1, None], dtype=pl.UInt64), "z": [2., 5.]})
    model = CellSensitiveModel(unsigned=True)
    exp = dx.Explainer(model, data, model.predict(data), model_type="regression", verbose=False)
    observations = pd.DataFrame({"x": pd.array([2**63 + 1, None], dtype="UInt64"), "z": [2., 5.]})
    arena = dx.Arena(precalculate=False, enable_attributes=False)
    arena.push_model(exp)
    arena.push_observations(observations)
    model_param = arena.get_params("model")[0]
    for i, observation in enumerate(arena.get_params("observation")):
        native = observation.get_row_for_model(model_param)
        assert isinstance(native, pl.DataFrame)
        np.testing.assert_array_equal(model.predict(native), model.predict(data.slice(i, 1)))


def test_arena_model_registration_does_not_export_a_full_feature_matrix(monkeypatch):
    class NestedModel:
        def predict(self, frame):
            return frame['x'].to_numpy() + np.array([sum(items) for items in frame['items'].to_list()])
    data = pl.DataFrame({'x': [1., 2., 3.], 'items': [[1, 2], [3], [4, 5]]})
    model = NestedModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    def reject(*args, **kwargs):
        raise AssertionError('Arena model registration must inspect columns independently')
    monkeypatch.setattr(pl.DataFrame, 'to_numpy', reject)
    arena = dx.Arena(precalculate=False, enable_attributes=False)
    arena.push_model(exp)
    assert list(arena.get_params('model')[0].variables) == ['x', 'items']


def test_arena_native_observations_are_buffer_sharing_snapshots(monkeypatch):
    import dalex.arena.params as params
    data = pl.DataFrame({'x': [1., 2.], 'z': [3., 4.]})
    model = CellSensitiveModel()
    exp = dx.Explainer(model, data, model.predict(data), model_type='regression', verbose=False)
    arena = dx.Arena(precalculate=False, enable_attributes=False)
    arena.push_model(exp)
    observations = data.clone()
    arena.push_observations(observations)
    observations.replace_column(0, pl.Series('x', [100., 200.]))
    def reject(*args, **kwargs):
        raise AssertionError('Same-backend native observations must bypass matrix reconstruction')
    monkeypatch.setattr(params, 'frame_from_values', reject)
    row = arena.get_params('observation')[0].get_row_for_model(arena.get_params('model')[0])
    assert isinstance(row, pl.DataFrame)
    assert row['x'].to_list() == [1.]
    assert arena.get_params('observation')[0].get_attributes()['x'] == 1.


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_arena_still_excludes_a_feature_identical_to_the_target(backend):
    data = pl.DataFrame({'x': [1., 2., 3.], 'z': [4., 3., 2.], 'target': [5., 5., 5.]})
    frame = data.to_pandas() if backend == 'pandas' else data if backend == 'polars' else data.to_arrow()
    class ConstantModel:
        def predict(self, batch):
            return np.full(len(batch), 5.)
    exp = dx.Explainer(ConstantModel(), frame, np.full(3, 5.), verbose=False)
    arena = dx.Arena(precalculate=False, enable_attributes=False)
    arena.push_model(exp)
    assert list(arena.get_params('model')[0].variables) == ['x', 'z']


@pytest.mark.parametrize('backend', ['polars', 'arrow'])
@pytest.mark.parametrize('unsigned', [False, True])
def test_custom_arena_http_observation_avoids_full_feature_conversion(backend, unsigned, monkeypatch):
    import json
    import requests
    import narwhals.stable.v2 as nw
    from dalex.arena.static import get_free_port
    data = pl.DataFrame({'x': pl.Series([2**63 + 1, None, 2**63 + 3] * 40, dtype=pl.UInt64) if unsigned
                         else pl.Series([1., None, np.nan] * 40), 'z': [2., 5., 1.] * 40})
    frame = data if backend == 'polars' else data.to_arrow()
    model = CellSensitiveModel(unsigned)
    exp = dx.Explainer(model, frame, model.predict(frame), model_type='regression', verbose=False, label='model')
    arena = dx.Arena(precalculate=False, enable_attributes=False)
    arena.push_model(exp)
    def forbidden(*args, **kwargs):
        raise AssertionError('Native HTTP observations must not construct a pandas display table')
    monkeypatch.setattr(nw.DataFrame, 'to_pandas', forbidden)
    port = get_free_port()
    assert port is not None
    arena.run_server(host='127.0.0.1', port=port)
    try:
        for value in ([None, 2**63 + 1] if unsigned else [None, 2.5]):
            response = requests.get(f'http://127.0.0.1:{port}/Breakdown',
                                    params={'model': 'model', 'observation': json.dumps({'x': value, 'z': 5.})},
                                    timeout=10)
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload['plotComponent'] == 'Breakdown'
            expected = 12. if value is None else float(value % 97) + 5. if unsigned else 7.5
            np.testing.assert_allclose(payload['data']['prediction'], expected)
    finally:
        arena.stop_server()
