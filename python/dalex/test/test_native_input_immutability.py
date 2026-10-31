"""Failed explanations must not leave partial interventions in caller data."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import dalex as dx
from dalex._dataframe import column_values, take_rows


class FailingModel:
    def __init__(self):
        self.armed = False
        self.calls = 0

    def predict(self, batch):
        if self.armed:
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError('deliberate inference failure')
        if isinstance(batch, np.ndarray):
            return 2 * batch[:, 0] + batch[:, 1]
        return 2 * column_values(batch, 'x') + column_values(batch, 'z')


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('family', ['break_down', 'shap', 'aspect_default', 'aspect_shap'])
def test_failed_interventions_leave_explainer_and_input_unchanged(backend, family):
    original = pd.DataFrame({'x': [1., 2., 3., 4.], 'z': [4., 2., 5., 1.]}, index=[7, 9, 20, 41])
    data = original.copy() if backend == 'pandas' else pl.from_pandas(original) if backend == 'polars' else pa.Table.from_pandas(original, preserve_index=False)
    model = FailingModel()
    y = 2 * original.x.to_numpy() + original.z.to_numpy()
    exp = dx.Explainer(model, data, y, model_type='regression', precalculate=False, verbose=False)
    observation = take_rows(data, [0])
    model.armed = True
    with pytest.raises(RuntimeError, match='deliberate inference failure'):
        if family.startswith('aspect_'):
            from dalex.aspect import PredictAspectImportance
            explanation = PredictAspectImportance(variable_groups={'x': ['x'], 'z': ['z']},
                                                  type=family.split('_')[1], N=4, B=3, random_state=42)
            explanation.fit(exp, observation)
        else:
            exp.predict_parts(observation, type=family, N=None, B=3, random_state=42)
    assert exp.data is data
    np.testing.assert_array_equal(exp.y, y)
    if backend == 'pandas':
        pd.testing.assert_frame_equal(data, original)
    elif backend == 'polars':
        assert data.equals(pl.from_pandas(original))
    else:
        assert data.equals(pa.Table.from_pandas(original, preserve_index=False))
