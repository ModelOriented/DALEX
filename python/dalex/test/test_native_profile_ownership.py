"""Read-only native reductions retain independent public results and snapshots."""
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx


class Model:
    def predict(self, data):
        frame = nw.from_native(data)
        assert frame.columns == ['x', 'z']
        return frame.select(2 * nw.col('x') + nw.col('z')).to_numpy().ravel()


def make(backend):
    data = pl.DataFrame({'x': [0., 1., 2., 3.], 'z': [0., 2., 4., 6.]})
    frame = data.to_pandas() if backend == 'pandas' else data if backend == 'polars' else data.to_arrow()
    model = Model()
    exp = dx.Explainer(model, frame, model.predict(frame), verbose=False)
    split = {'x': np.array([-1., 0., 1.])}
    profiles = [exp.predict_profile(nw.from_native(frame)[ids, :].to_native(), variables=['x'],
                variable_splits=split, verbose=False) for ids in [[0, 1], [2, 3]]]
    return profiles


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('kind', ['partial', 'conditional', 'accumulated'])
@pytest.mark.parametrize('count', [1, 2])
def test_reductions_do_not_alias_or_modify_cp_inputs(backend, kind, count):
    profiles = make(backend)[:count]
    before = [value.result_as('pyarrow', partition='x') for value in profiles]
    aggregate = dx.model_explanations.AggregatedProfiles(type=kind, variables=['x'])
    aggregate.fit(profiles[0] if count == 1 else profiles, verbose=False)
    first = aggregate.result_as('pyarrow')
    repeated = dx.model_explanations.AggregatedProfiles(type=kind, variables=['x'])
    repeated.fit(profiles[0] if count == 1 else profiles, verbose=False)
    assert repeated.result_as('pyarrow').equals(first)
    aggregate.result['_yhat_'] += 100.
    assert repeated.result_as('pyarrow').equals(first)
    for value, expected in zip(profiles, before):
        assert value.result_as('pyarrow', partition='x').equals(expected)
    if count == 1:
        saved = aggregate.raw_profiles.result_as('pyarrow', partition='x')
        profiles[0].result['_yhat_'] += 200.
        assert aggregate.raw_profiles.result_as('pyarrow', partition='x').equals(saved)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_edited_multi_label_cp_uses_all_rows_without_mutation(backend):
    profile = make(backend)[0]
    profile.result.loc[profile.result['_ids_'] == 1, '_label_'] = 'edited'
    before = profile.result.copy(deep=True)
    aggregate = dx.model_explanations.AggregatedProfiles(type='partial', variables=['x'])
    aggregate.fit(profile, verbose=False)
    result = aggregate.result
    expected = {'edited': [0., 2., 4.], 'Model': [-2., 0., 2.]}
    assert set(result['_label_']) == set(expected)
    for label, predictions in expected.items():
        selected = result.loc[result['_label_'] == label].sort_values('_x_')
        np.testing.assert_array_equal(selected['_yhat_'], predictions)
    pd.testing.assert_frame_equal(profile.result, before)
