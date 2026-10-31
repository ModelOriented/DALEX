"""Bundled CSV loaders preserve feature values and offer explicit native indexes."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import dalex as dx
from dalex._dataframe import to_pandas
from test_native_profile_workflows import native_computation_only

NAMES = ['titanic', 'fifa', 'apartments', 'apartments_test', 'dragons', 'dragons_test', 'hr', 'hr_test', 'german']


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('name', NAMES)
def test_native_dataset_values_match_legacy_csv(name, backend, monkeypatch):
    loader = getattr(dx.datasets, 'load_' + name)
    expected = loader().reset_index(drop=True)
    with native_computation_only(monkeypatch, backend):
        result = loader(backend=backend)
    assert isinstance(result, pl.DataFrame if backend == 'polars' else pa.Table)
    actual = to_pandas(result)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('name', NAMES[1:-1])
def test_native_dataset_row_identifiers_are_explicit(name, backend):
    loader = getattr(dx.datasets, 'load_' + name)
    expected = loader(include_index=True)
    actual = to_pandas(loader(backend=backend, include_index=True))
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def test_dataset_backend_is_validated():
    with pytest.raises(ValueError, match='backend'):
        dx.datasets.load_titanic(backend='unknown')
