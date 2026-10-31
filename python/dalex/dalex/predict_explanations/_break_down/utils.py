"""Legacy pandas-returning adapters over native contribution kernels."""
import numpy as np
import narwhals.stable.v2 as nw
from narwhals.dependencies import get_pandas
from ..._dataframe import column_names, observation_values, observation_display_values, to_pandas


def local_interactions(explainer, new_observation, interaction_preference, type, order, keep_distributions):
    from .._native_contributions import break_down
    result, distributions = break_down(explainer, new_observation, interaction_preference,
                                       type, order, keep_distributions)
    return (to_pandas(result), distributions._dalex_pandas_view() if distributions is not None else None)


def calculate_1d_changes(explainer, new_observation, values=None):
    from .._native_contributions import one_dimensional_changes
    cells = observation_values(new_observation) if values is None else values
    return get_pandas().Series(one_dimensional_changes(explainer, cells), index=column_names(explainer.data))


def calculate_2d_changes(explainer, new_observation, inds, diffs_1d, values=None):
    from .._native_contributions import two_dimensional_changes
    cells = observation_values(new_observation) if values is None else values
    indices = nw.from_native(inds, eager_only=True).to_numpy()
    differences = np.asarray(diffs_1d)
    averages, normalized = two_dimensional_changes(explainer, cells, indices, differences)
    names = list(column_names(explainer.data))
    labels = [':'.join(names[int(i)] for i in pair) for pair in indices]
    pd = get_pandas()
    return pd.Series(averages, index=labels), pd.Series(normalized, index=labels)


def create_ordered_path(feature_path, average_yhats_index, type, order):
    from .._native_contributions import path_order
    frame = nw.from_native(feature_path, eager_only=True)
    score = 'adiff_norm' if type == '2d' else 'diff'
    positions = path_order(frame[score].to_numpy(), list(feature_path.index), average_yhats_index, order)
    # The private pandas contract retains its caller's index, including labels
    # used to identify interactions. This is an index-publication boundary.
    return feature_path.iloc[positions, :]


def calculate_contributions_along_path(explainer, new_observation, feature_path,
        baseline_yhat, target_yhat, type, keep_distributions, values=None):
    from .._native_contributions import contributions_along_path
    from ..._explanation import IndexedTable
    cells = observation_values(new_observation) if values is None else values
    display = observation_display_values(new_observation, cells)
    frame = nw.from_native(feature_path, eager_only=True)
    rows = []
    for name, record in zip(feature_path.index, frame.iter_rows(named=True)):
        candidates = [record['ind1']]
        if type == '2d' and record['ind2'] is not None:
            candidates.append(record['ind2'])
        rows.append((name, tuple(int(value) for value in candidates), None))
    result, distributions = contributions_along_path(explainer, cells, display, rows,
        float(np.asarray(baseline_yhat).item()), float(np.asarray(target_yhat).item()), keep_distributions)
    published = to_pandas(result)
    columns = {name: published[name].to_numpy() for name in published.columns}
    columns['variable'] = published['variable'].tolist()
    columns['label'] = explainer.label
    if keep_distributions:
        data = nw.concat([nw.from_native(data) for data in distributions]).to_native()
        index = np.concatenate([np.arange(len(nw.from_native(data))) for data in distributions])
        predictions = IndexedTable(data, index)._dalex_pandas_view()
    else:
        predictions = None
    return columns, predictions


def nice_pair(df, i1, i2):
    return nice_format(df.iloc[0, i1]) if i2 is None else ':'.join(
        (nice_format(df.iloc[0, i1]), nice_format(df.iloc[0, i2])))


def nice_format(x):
    return str(float(signif(x))) if isinstance(x, (int, float, np.integer, np.floating)) else str(x)


#:# https://stackoverflow.com/a/59888924
def signif(x, p=4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
