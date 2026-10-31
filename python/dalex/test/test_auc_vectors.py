"""Independent ROC-AUC checks, including exact large-integer decision scores."""
import numpy as np
import pandas as pd
import pytest
from dalex.model_explanations._model_performance.utils import auc


@pytest.mark.parametrize('seed', range(10))
@pytest.mark.parametrize('size', [2, 13, 100])
def test_auc_matches_pairwise_rank_probability(seed, size, monkeypatch):
    rng = np.random.default_rng(seed)
    observed = rng.integers(0, 2, size)
    observed[:2] = [0, 1]
    scores = rng.choice([0., .25, .5, .75, 1.], size)
    positive, negative = scores[observed == 1], scores[observed == 0]
    expected = ((positive[:, None] > negative[None, :])
                + .5 * (positive[:, None] == negative[None, :])).mean()
    def forbidden(*args, **kwargs):
        raise AssertionError('AUC constructed a dataframe')
    monkeypatch.setattr(pd, 'DataFrame', forbidden)
    monkeypatch.setattr(pd, 'Series', forbidden)
    np.testing.assert_allclose(auc(scores, observed), expected, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize('dtype', ['uint64', 'int64', 'object'])
def test_auc_retains_integer_score_order_beyond_float_precision(dtype):
    values = np.array([2**63 - 2, 2**63 - 3, 2**63 - 1, 2**63 - 4], dtype=dtype)
    assert auc(values, [1, 0, 1, 0]) == 1.
    assert auc(values, [0, 1, 0, 1]) == 0.
    assert auc(np.array([2**64-2, 2**64-3, 2**64-1, 2**64-4], dtype='uint64'), [1, 0, 1, 0]) == 1.


def test_auc_preserves_legacy_missing_score_denominators():
    assert auc([1., 0., np.nan, .5], [1., 0., 1., 0.]) == .5
    assert auc([np.nan, np.nan], [0., 1.]) == 0.
    assert auc([], []) == 0.
    assert np.isnan(auc([0., 1.], [np.nan, np.nan]))
    with pytest.raises(ValueError, match='Only one class'):
        auc([0., 1.], [1., 1.])
