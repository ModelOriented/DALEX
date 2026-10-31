"""Structured feature columns must survive gathers/permutations unmodified."""
from decimal import Decimal

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import dalex as dx
from dalex._dataframe import column_names, column_values, shuffle_columns, take_rows


class StructuredModel:
    def __init__(self, reference):
        self.reference = reference

    def predict(self, data):
        assert type(data) is type(self.reference)
        if isinstance(data, pd.DataFrame):
            assert data.dtypes.equals(self.reference.dtypes)
        else:
            assert data.schema == self.reference.schema
        return column_values(data, "x") * 2


@pytest.mark.parametrize("backend", ["pandas", "polars", "arrow"])
def test_extended_schema_through_sampling_and_importance(backend):
    frame = pd.DataFrame({
        "x": [1., 2., 3., 4.],
        "when": pd.date_range("2025-01-01", periods=4, tz="America/Sao_Paulo"),
        "amount": [Decimal("1.25"), None, Decimal("2.50"), Decimal("3.75")],
        "list": [[1, 2], None, [3], []],
        "record": [{"a": 1}, None, {"a": 2}, {"a": 3}],
    })
    data = frame if backend == "pandas" else (pl.from_pandas(frame) if backend == "polars" else pa.Table.from_pandas(frame))
    order = [3, 0, 2, 1]
    sampled = take_rows(data, order)
    shuffled = shuffle_columns(data, list(column_names(data)), order)
    if backend == "pandas":
        pd.testing.assert_frame_equal(shuffled.reset_index(drop=True), sampled.reset_index(drop=True))
    else:
        assert shuffled.schema == data.schema
        assert shuffled.equals(sampled)
    model = StructuredModel(data)
    exp = dx.Explainer(model, data, np.array([2.1, 3.9, 6.1, 7.9]), verbose=False)
    result = exp.model_parts(N=None, B=3, random_state=42).result
    assert np.isfinite(result.dropout_loss).all()
    assert exp.data is data
    if backend == "pandas":
        pd.testing.assert_frame_equal(data, frame)
    elif backend == "polars":
        assert data.equals(pl.from_pandas(frame))
    else:
        assert data.equals(pa.Table.from_pandas(frame))
