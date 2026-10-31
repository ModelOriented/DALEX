"""Preserve the duplicate-label pandas APIs that worked before Narwhals."""
import pickle

import numpy as np
import pandas as pd
import pytest

import dalex as dx
from dalex._dataframe import shuffle_columns, frame_from_values


class NumericModel:
    def predict(self, frame):
        values = np.asarray(frame, dtype=float)
        return values @ [1., 2., 3.]


class BinaryModel(NumericModel):
    def predict(self, frame):
        return 1 / (1 + np.exp(-super().predict(frame)))


def labels(case):
    if case in ['multiindex', 'reserved-multiindex']:
        names = [('g', 1), ('g', 1), ('h', 1)] if case == 'multiindex' else [('y_hat', 1), ('y_hat', 1), ('label', 1)]
        return pd.MultiIndex.from_tuples(names, names=['group', 'feature'])
    return {'strings': ['x', 'x', 'z'], 'integers': [10, 10, 20],
            'missing': [None, None, 'z'], 'reserved': ['y_hat', 'y_hat', 'label']}[case]


def fixture(case, kind='regression'):
    values = np.random.default_rng(102).normal(size=(30, 3))
    data = pd.DataFrame(values, columns=labels(case), index=pd.Index(np.arange(30)*3+7, name='row'))
    model = BinaryModel() if kind == 'classification' else NumericModel()
    target = ((values[:, 0] > 0).astype(int) if kind == 'classification'
              else model.predict(data) + np.sin(values[:, 0]) / 10)
    return dx.Explainer(model, data, target, model_type=kind, label='model', verbose=False)


@pytest.mark.parametrize('case', ['strings', 'integers', 'missing', 'reserved', 'multiindex', 'reserved-multiindex'])
@pytest.mark.parametrize('kind', ['regression', 'classification'])
def test_metrics_do_not_require_unique_feature_labels(case, kind):
    exp = fixture(case, kind)
    renamed = exp.data.copy()
    renamed.columns = ['a', 'b', 'c']
    reference = dx.Explainer(exp.model, renamed, exp.y, model_type=kind, label='model', verbose=False)
    value = exp.model_performance()
    pd.testing.assert_frame_equal(value.result, reference.model_performance().result)
    native = value.result_as().set_index('label').rename_axis(None)
    pd.testing.assert_frame_equal(native, value.result_as('pandas'))
    np.testing.assert_allclose(exp.predict(exp.data), reference.predict(renamed), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('case', ['strings', 'integers', 'missing', 'reserved', 'multiindex', 'reserved-multiindex'])
@pytest.mark.parametrize('kind', ['variable_importance', 'ratio', 'difference'])
def test_importance_shuffles_duplicate_labels_jointly_and_preserves_raw_draws(case, kind):
    exp = fixture(case)
    before = exp.data.copy(deep=True)
    value = exp.model_parts(type=kind, N=20, B=4, random_state=42, keep_raw_permutations=True)
    assert value.permutation.shape == (4, 4)  # Two label groups and the two baselines.
    assert len(value.result) == 4
    assert np.isfinite(value.result.dropout_loss).all()
    pd.testing.assert_frame_equal(value.result_as(), value.result)
    pd.testing.assert_frame_equal(value.result_as(table='permutation'), value.permutation)
    pd.testing.assert_frame_equal(exp.data, before)


@pytest.mark.parametrize('case', ['strings', 'integers', 'missing', 'reserved', 'multiindex', 'reserved-multiindex'])
@pytest.mark.parametrize('selection', ['all', 'first', 'unique'])
def test_diagnostics_keep_legacy_duplicate_columns_index_and_assignment(case, selection):
    exp = fixture(case)
    variables = None if selection == 'all' else [exp.data.columns[0 if selection == 'first' else 2]]
    value = exp.model_diagnostics(variables=variables)
    source = exp.data if variables is None else exp.data.loc[:, variables]
    expected = source.assign(y=exp.y, y_hat=exp.y_hat, residuals=exp.residuals,
                             abs_residuals=np.abs(exp.residuals), label=exp.label,
                             ids=np.arange(len(source))+1)
    assert '_compatibility_result' not in value.__dict__
    pd.testing.assert_frame_equal(value.result_as(), expected)
    assert '_compatibility_result' not in value.__dict__
    pd.testing.assert_frame_equal(value.result, expected)
    pd.testing.assert_frame_equal(pickle.loads(pickle.dumps(value)).result, expected)
    if not expected.columns.is_unique:
        for backend in ['polars', 'pyarrow']:
            with pytest.raises(ValueError, match='Duplicate pandas column labels'):
                value.result_as(backend)


@pytest.mark.parametrize('multiindex', [False, True])
@pytest.mark.parametrize('joint', [False, True])
def test_duplicate_shuffle_is_positional_and_retains_extension_dtypes(multiindex, joint):
    data = pd.DataFrame({0: pd.array([1, None, 3, 4], dtype='Int64'),
                         1: pd.Categorical(['a', 'b', 'a', 'b']),
                         2: pd.array([.5, 1.5, None, 3.5], dtype='Float64')})
    data.columns = pd.MultiIndex.from_tuples([('g', 1), ('g', 1), ('h', 1)]) if multiindex else ['x', 'x', 'z']
    data.index = pd.Index([101, 72, 29, 413], name='row')
    first, last = data.columns[0], data.columns[2]
    selector = [first, last] if joint else first
    order = [2, 0, 3, 1]
    before = data.copy(deep=True)
    actual = shuffle_columns(data, selector, order)
    for position in range(3):
        expected = data.iloc[order, position].set_axis(data.index) if position < 2 or joint else data.iloc[:, position]
        pd.testing.assert_series_equal(actual.iloc[:, position], expected)
    pd.testing.assert_frame_equal(data, before)


def test_duplicate_diagnostic_views_are_authoritative_and_old_state_still_exports():
    value = fixture('strings').model_diagnostics()
    exported = value.result_as()
    exported.iloc[0, -1] = -900
    assert value.result_as().iloc[0, -1] == 1
    public = value.result
    public.iloc[0, -1] = 777
    pd.testing.assert_frame_equal(value.result_as(), public)
    # Renaming the writable projection permits an explicit unique-column export.
    public.columns = ['left', 'right', *list(public.columns[2:])]
    pd.testing.assert_frame_equal(value.result_as('polars').to_pandas(), public.reset_index(drop=True))
    restored = pickle.loads(pickle.dumps(value))
    pd.testing.assert_frame_equal(restored.result_as(), public)
    legacy = fixture('strings').model_diagnostics()
    legacy.__dict__['result'] = exported  # State shape of an actual old pandas object.
    pd.testing.assert_frame_equal(legacy.result_as(), exported)
    pd.testing.assert_frame_equal(legacy.result_as('pandas'), exported)


def test_grouped_duplicate_importance_retains_serial_and_process_paths():
    exp = fixture('strings')
    options = dict(N=20, B=4, random_state=42, variable_groups={'both': ['x', 'z']})
    serial = exp.model_parts(**options)
    parallel = exp.model_parts(**options, processes=2)
    # Existing process streams differ from serial streams. Each joint group
    # contains all physical columns, so it has the same observed/baseline layout.
    assert set(serial.result.variable) == set(parallel.result.variable) == {'both', '_baseline_', '_full_model_'}
    assert np.isfinite(parallel.result.dropout_loss).all()
    assert parallel.permutation.shape == serial.permutation.shape == (4, 3)


@pytest.mark.parametrize('case', ['strings', 'reserved-multiindex'])
def test_duplicate_diagnostics_label_override_does_not_publish_the_view(case):
    exp = fixture(case)
    value = exp.model_diagnostics(label='custom')
    assert '_compatibility_result' not in value.__dict__
    native = value.result_as()
    # pandas 1.5's partial MultiIndex __getitem__ reindexes the unrelated
    # duplicate features. Inspect the label's physical column directly.
    name = ('label', 1) if isinstance(native.columns, pd.MultiIndex) else 'label'
    position = list(native.columns).index(name)
    assert (native.iloc[:, position].to_numpy() == 'custom').all()
    assert '_compatibility_result' not in value.__dict__
    pd.testing.assert_frame_equal(value.result, native)


@pytest.mark.parametrize('case', ['strings', 'integers', 'missing', 'reserved', 'multiindex', 'reserved-multiindex'])
@pytest.mark.parametrize('kind', ['regression', 'classification'])
def test_fairness_backend_metadata_does_not_validate_feature_uniqueness(case, kind):
    exp = fixture(case, kind)
    renamed = exp.data.copy()
    renamed.columns = ['a', 'b', 'c']
    reference = dx.Explainer(exp.model, renamed, exp.y, model_type=kind, label='model', verbose=False)
    protected = np.array(['u', 'v'] * 15)
    actual = exp.model_fairness(protected, 'u')
    expected = reference.model_fairness(protected, 'u')
    pd.testing.assert_frame_equal(actual.result, expected.result, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(pickle.loads(pickle.dumps(actual)).result, actual.result)


@pytest.mark.parametrize('kind', ['regression', 'classification'])
@pytest.mark.parametrize('discretize', [False, True])
def test_lime_preserves_duplicate_model_columns_and_original_external_class(kind, discretize):
    from lime.explanation import Explanation as LimeExplanation
    exp = fixture('strings', kind)
    renamed = exp.data.copy()
    renamed.columns = ['a', 'b', 'c']
    reference = dx.Explainer(exp.model, renamed, exp.y, model_type=kind, label='model', verbose=False)
    options = dict(random_state=42, num_samples=128, num_features=3, discretize_continuous=discretize)
    observation = exp.data.iloc[0].to_numpy()
    actual = exp.predict_surrogate(observation, **options)
    expected = reference.predict_surrogate(observation, **options)
    assert type(actual) is LimeExplanation
    assert isinstance(actual.result, pd.DataFrame)
    np.testing.assert_allclose(actual.local_exp[1], expected.local_exp[1], rtol=1e-12, atol=1e-12)
    assert np.isfinite(actual.result.effect).all()


@pytest.mark.parametrize('case', ['strings', 'duplicate-multiindex', 'unique-multiindex'])
@pytest.mark.parametrize('single', [False, True])
def test_external_matrix_reconstruction_keeps_physical_columns_and_typed_cells(case, single):
    reference = pd.DataFrame({
        0: pd.Categorical(['b', 'a', 'b'], categories=['b', 'a', 'unused'], ordered=True),
        1: pd.array([2**63+1, 2**63+3, 2**63+5], dtype='UInt64'),
        2: pd.date_range('2020-01-01', periods=3, freq='ns', tz='UTC'),
    }, index=[19, 71, 43])
    if case == 'strings':
        reference.columns = pd.Index(['x', 'x', 'when'], name='features')
    else:
        pairs = [('g', 1), ('g', 1 if case.startswith('duplicate') else 2), ('h', 1)]
        reference.columns = pd.MultiIndex.from_tuples(pairs, names=['group', 'feature'])
    positions = [1] if single else [1, 0]
    matrix = reference.iloc[positions].to_numpy()
    actual = frame_from_values(matrix[0] if single else matrix, reference)
    assert isinstance(actual.index, pd.RangeIndex)
    pd.testing.assert_index_equal(actual.columns, reference.columns, exact=True)
    expected = reference.iloc[positions].reset_index(drop=True)
    pd.testing.assert_series_equal(actual.iloc[:, 0], expected.iloc[:, 0])
    assert actual.iloc[:, 1].tolist() == expected.iloc[:, 1].tolist()
    pd.testing.assert_series_equal(actual.iloc[:, 2], expected.iloc[:, 2])


def test_external_matrix_explicit_subset_retains_requested_order_and_categorical_dtype():
    reference = pd.DataFrame({'kind': pd.Categorical(['b', 'a'], categories=['b', 'a', 'unused']),
                              'x': [.5, 1.5], 'unused': [9., 7.]})
    reference.columns.name = 'feature'
    expected = reference.loc[:, ['x', 'kind']]
    actual = frame_from_values(expected.to_numpy(), reference, columns=['x', 'kind'])
    pd.testing.assert_frame_equal(actual, expected)
