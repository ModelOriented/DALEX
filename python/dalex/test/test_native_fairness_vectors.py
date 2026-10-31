"""Fairness metadata may originate in the same native backend as features."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import dalex as dx
from dalex._dataframe import column_values
from dalex.fairness import reweight, resample, roc_pivot
from dalex.fairness._basics.exceptions import ParameterCheckError


def vector(values, kind):
    if kind == "polars-series":
        return pl.Series("value", values)
    if kind == "polars-frame":
        return pl.DataFrame({"value": values})
    if kind == "arrow-array":
        return pa.array(values)
    if kind == "arrow-dictionary":
        return pa.chunked_array([pa.array(values[:20]).dictionary_encode(),
                                 pa.array(values[20:]).dictionary_encode()])
    return pa.table({"value": values})


class NativeFairnessModel:
    def __init__(self, classification):
        self.classification = classification

    def predict(self, frame):
        assert isinstance(frame, pl.DataFrame)
        score = column_values(frame, "x") + .3 * column_values(frame, "z")
        return 1 / (1 + np.exp(-score)) if self.classification else score


KINDS = ["polars-series", "polars-frame", "arrow-array", "arrow-dictionary", "arrow-table"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("classification", [False, True])
def test_native_protected_vectors_match_numpy_fairness_and_pivot(kind, classification):
    rng = np.random.default_rng(123)
    frame = pl.DataFrame({"x": rng.normal(size=80), "z": rng.normal(size=80)})
    model = NativeFairnessModel(classification)
    prediction = model.predict(frame)
    y = rng.binomial(1, prediction) if classification else prediction + rng.normal(size=80)
    exp = dx.Explainer(model, frame, y, predict_function=lambda m, f: m.predict(f),
                       model_type="classification" if classification else "regression", verbose=False)
    protected = np.tile(["a", "b"], 40)
    native = vector(protected, kind)
    expected = exp.model_fairness(protected, "a")
    actual = exp.model_fairness(native, "a")
    pd.testing.assert_frame_equal(actual.result, expected.result)
    np.testing.assert_array_equal(actual.protected, protected)
    assert exp.data is frame
    if classification:
        np.testing.assert_array_equal(roc_pivot(exp, native, "a", verbose=False).y_hat,
                                      roc_pivot(exp, protected, "a", verbose=False).y_hat)


@pytest.mark.parametrize("kind", KINDS)
def test_native_mitigation_targets_and_memberships_match_numpy(kind):
    protected = np.array(["a"] * 30 + ["b"] * 30)
    y = np.array([0] * 20 + [1] * 10 + [0] * 10 + [1] * 20)
    native_protected, native_y = vector(protected, kind), vector(y, kind)
    np.testing.assert_array_equal(reweight(native_protected, native_y, verbose=False),
                                  reweight(protected, y, verbose=False))
    probs = np.linspace(.1, .9, len(y))
    np.testing.assert_array_equal(
        resample(native_protected, native_y, type="preferential", probs=probs, verbose=False),
        resample(protected, y, type="preferential", probs=probs, verbose=False),
    )


@pytest.mark.parametrize("protected", [pl.DataFrame({"a": ["a"], "b": ["b"]}),
                                       pa.table({"a": ["a"], "b": ["b"]})])
def test_multicolumn_membership_is_rejected_clearly(protected):
    with pytest.raises(ParameterCheckError, match="one-dimensional|one column"):
        reweight(protected, np.array([0]), verbose=False)
