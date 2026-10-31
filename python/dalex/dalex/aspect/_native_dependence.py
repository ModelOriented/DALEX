"""Backend-neutral association statistics; only scalar vectors enter SciPy."""
import warnings
import numpy as np
import narwhals.stable.v2 as nw
from .._explanation import IndexedTable
from .._dataframe import column_values, categorical_levels


def matrix_values(table):
    """Return values and labels from native storage or an edited legacy view."""
    if isinstance(table, IndexedTable):
        frame = nw.from_native(table.data, eager_only=True)
        return frame.to_numpy(), list(frame.columns)
    frame = nw.from_native(table, eager_only=True)
    # A user-supplied pandas matrix may have a different row ordering.
    if nw.get_native_namespace(frame).__name__ == 'pandas':
        table = table.loc[frame.columns, frame.columns]
        return table.to_numpy(), list(frame.columns)
    return frame.to_numpy(), list(frame.columns)


def matrix_table(values, names, reference):
    backend = nw.get_native_namespace(nw.from_native(reference, eager_only=True))
    # Physical non-string labels remain an explicit pandas compatibility case.
    if not all(isinstance(name, str) for name in names):
        import pandas as pd
        return IndexedTable(pd.DataFrame(values, columns=names), index=names)
    data = nw.from_dict({name: np.asarray(values[:, i], dtype=float)
                         for i, name in enumerate(names)}, backend=backend).to_native()
    return IndexedTable(data, index=names)


def valid_mask(series):
    mask = ~series.is_null()
    if series.dtype.is_float():
        mask = mask & ~series.is_nan()
    return np.asarray(mask.to_numpy(), dtype=bool)


def category_codes(series):
    """Factorization retains scalar types and excludes missing memberships."""
    mask = valid_mask(series)
    if not mask.any():
        return np.full(len(series), -1, dtype=int), 0
    # A native hash join avoids one full-column comparison per unique value.
    # Typed keys retain temporal precision; no feature frame is edited.
    source = series.alias('__value').to_frame().with_row_index('__row')
    unique = source.filter(nw.new_series('__valid', mask, backend=nw.get_native_namespace(series)))
    unique = unique.select('__value').unique().with_row_index('__code')
    joined = source.join(unique, on='__value', how='left').sort('__row')
    codes = joined.select(nw.col('__code').fill_null(-1).cast(nw.Int64))['__code'].to_numpy().copy()
    codes[~mask] = -1
    return codes, len(unique)


def cramers_value(a, na, b, nb, n):
    from scipy.stats import chi2_contingency
    valid = (a >= 0) & (b >= 0)
    row_counts = np.bincount(a[valid], minlength=na)
    col_counts = np.bincount(b[valid], minlength=nb)
    r, c = np.count_nonzero(row_counts), np.count_nonzero(col_counts)
    if min(r, c) <= 1 or n <= 1:
        return np.nan
    if na * nb <= 1_000_000:
        counts = np.zeros((na, nb), dtype=int)
        np.add.at(counts, (a[valid], b[valid]), 1)
        counts = counts[row_counts > 0][:, col_counts > 0]
        chi = chi2_contingency(counts)[0]
    else:
        # Include zero cells without allocating a potentially huge crosstab.
        pairs, counts = np.unique(np.column_stack((a[valid], b[valid])), axis=0, return_counts=True)
        expected = (row_counts[pairs[:, 0]].astype(float) * col_counts[pairs[:, 1]]) / valid.sum()
        chi = max(0., np.sum(counts.astype(float)**2 / expected) - valid.sum())
    phi = max(0., chi / n - (r - 1) * (c - 1) / (n - 1))
    denominator = min(r - (r - 1)**2 / (n - 1), c - (c - 1)**2 / (n - 1)) - 1
    return np.sqrt(phi / denominator) if denominator > 0 else np.nan


def category_sample_count(series, count):
    # Retain the legacy pandas default for declared but unobserved categories.
    # This reads dtype metadata without constructing any pandas objects.
    from narwhals.dependencies import get_pandas
    pandas = get_pandas()
    if (isinstance(series.dtype, (nw.Categorical, nw.Enum)) and pandas is not None
            and int(pandas.__version__.split('.')[0]) < 3):
        return max(count, len(categorical_levels(series)))
    return count


def eta_value(membership, count, vector, n):
    from scipy.stats import kruskal
    if count < 2 or n == count:
        return np.nan
    samples = [vector[membership == k] for k in range(count)]
    try:
        statistic = kruskal(*samples, nan_policy='omit').statistic
    except ValueError as exc:
        if 'All numbers are identical' not in str(exc):
            raise
        statistic = np.nan
    return (statistic - count + 1) / (n - count)


def associations(data, method):
    from scipy.stats import spearmanr, kendalltau
    frame = nw.from_native(data, eager_only=True)
    numeric = [name for name, dtype in frame.schema.items()
               if dtype.is_numeric() or dtype == nw.Boolean]
    # Retain DALEX's historical categorical ordering (including its tie order).
    categorical = list(set(frame.columns) - set(numeric))
    names = numeric + categorical
    n = len(frame)
    result = np.full((len(names), len(names)), np.nan)
    # Correlation statistics historically use float64 vectors. Convert only
    # these detached numerical inputs: Arrow's strict dataframe cast rejects
    # integers above 2**53, while feature interventions must retain them exactly.
    vectors = {name: np.asarray(column_values(data, name, preserve_nulls=True), dtype=float) for name in numeric}
    codes = {name: category_codes(frame[name]) for name in categorical}
    for i, left in enumerate(names):
        for j in range(i, len(names)):
            right = names[j]
            if left in vectors and right in vectors:
                a, b = vectors[left], vectors[right]
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() < 2 or np.ptp(a[valid]) == 0 or np.ptp(b[valid]) == 0:
                    value = np.nan
                elif i == j:
                    value = 1.
                elif method == 'pearson':
                    value = abs(np.corrcoef(a[valid], b[valid])[0, 1])
                elif method == 'spearman':
                    value = abs(spearmanr(a[valid], b[valid]).statistic)
                else:
                    value = abs(kendalltau(a[valid], b[valid]).statistic)
            elif left in codes and right in codes:
                if i == j:
                    value = 1.
                else:
                    a, na = codes[left]
                    b, nb = codes[right]
                    value = cramers_value(a, na, b, nb, n)
            else:
                category, number = (left, right) if left in codes else (right, left)
                membership, count = codes[category]
                count = category_sample_count(frame[category], count)
                value = eta_value(membership, count, vectors[number], n)
            result[i, j] = result[j, i] = np.clip(value, 0., 1.)
    return matrix_table(result, names, data)


def normalize_missing(table):
    values, names = matrix_values(table)
    if np.isnan(values).any():
        warnings.warn('There were NaNs in `depend_matrix`. This is possibly because there is a feature in the data with only one unique value. Replacing NaN values on the diagonal with 1 and others with 0.')
        values = np.where(np.isnan(values), 0., values)
        np.fill_diagonal(values, 1.)
        reference = table.data if isinstance(table, IndexedTable) else table
        table = matrix_table(values, names, reference)
    return table
