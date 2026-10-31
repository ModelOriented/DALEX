"""Native contribution computation, with legacy labels at scalar boundaries."""
import multiprocessing as mp
import numpy as np
import narwhals.stable.v2 as nw
from numpy.random import SeedSequence, default_rng

from .._dataframe import (as_eager_dataframe, column_names, copy_for_edit, set_columns,
                         observation_values, observation_display_values, take_rows)
from .._explanation import IndexedTable, export_result_frame
from .._random import get_random_state
from ._ceteris_paribus._table import concat_compatible


def _format(value):
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    number = np.asarray(value)
    positive = np.where(np.isfinite(number) & (number != 0), np.abs(number), 1000)
    magnitude = 10 ** (3 - np.floor(np.log10(positive)))
    return str(float(np.round(number * magnitude) / magnitude))


def _snapshot(data):
    return as_eager_dataframe(copy_for_edit(data)).clone().to_native()


def _frame(columns, namespace):
    return nw.from_dict(columns, backend=namespace).to_native()


def _distribution(names, caption, predicted, label, namespace):
    size = len(predicted)
    return _frame({'variable_name': [names] * size, 'variable': [caption] * size,
                   'id': np.arange(size), 'prediction': np.array(predicted, copy=True),
                   'label': [label] * size}, namespace)


def one_dimensional_changes(explainer, cells):
    names = list(column_names(explainer.data))
    averages = np.empty(len(names), dtype=object)
    for i, name in enumerate(names):
        working = set_columns(copy_for_edit(explainer.data), {name: cells[name]}, scalar=True, copy=False)
        averages[i] = explainer.predict(working).mean()
    return averages


def two_dimensional_changes(explainer, cells, indices, differences):
    names = list(column_names(explainer.data))
    averages, normalized = np.empty(len(indices)), np.empty(len(indices))
    for index, candidates in enumerate(indices):
        working = set_columns(copy_for_edit(explainer.data),
                              {names[int(i)]: cells[names[int(i)]] for i in candidates}, scalar=True, copy=False)
        averages[index] = explainer.predict(working).mean()
        value = averages[index]
        for candidate in candidates:
            value = value - differences[int(candidate)]
        normalized[index] = value
    return averages, normalized


def path_order(scores, labels, feature_names, order):
    if order is None:
        return np.argsort(np.asarray(scores))[::-1]
    if np.issubdtype(order.dtype, np.integer):
        return order
    if np.isin(order, feature_names).all():
        return [labels.index(name) for name in order]
    raise ValueError('Wrong order!')


def break_down(explainer, observation, preference, kind, order, keep_distributions):
    namespace = nw.get_native_namespace(as_eager_dataframe(explainer.data))
    names = list(column_names(explainer.data))
    cells = observation_values(observation)
    display = observation_display_values(observation, cells)
    target = explainer.predict(observation)[0]
    baseline = np.mean(explainer.predict(explainer.data))
    averages = one_dimensional_changes(explainer, cells)
    differences = averages - baseline
    # Rows are (public feature name, positions, sort score). Keep object-score
    # argsort, matching the original pandas object-Series tie ordering.
    rows = [(str(name), (i,), differences[i] if kind == '1d' else abs(differences[i])) for i, name in enumerate(names)]
    if kind == '2d':
        indices = [(higher, lower) for lower in range(len(names) - 1) for higher in range(lower + 1, len(names))]
        _, normalized = two_dimensional_changes(explainer, cells, indices, differences)
        rows.extend((f'{names[higher]}:{names[lower]}', (higher, lower), abs(value - baseline) * preference)
                    for (higher, lower), value in zip(indices, normalized))
    positions = path_order(np.asarray([row[2] for row in rows], dtype=object), names, names, order)
    result, distributions = contributions_along_path(explainer, cells, display,
        [rows[int(position)] for position in positions], baseline, target, keep_distributions)
    if not keep_distributions:
        return result, None
    baseline_distribution = _distribution('all data', 'all data', explainer.predict(explainer.data), explainer.label, namespace)
    frames = [baseline_distribution, *distributions]
    table = nw.concat([nw.from_native(frame) for frame in frames], how='vertical').to_native()
    indexes = np.concatenate([np.arange(len(nw.from_native(frame))) for frame in frames])
    return result, IndexedTable(table, indexes)


def contributions_along_path(explainer, cells, display, rows, baseline, target, keep_distributions):
    namespace = nw.get_native_namespace(as_eager_dataframe(explainer.data))
    names = list(column_names(explainer.data))
    available = set(range(len(names)))
    working = copy_for_edit(explainer.data)
    selected, means, distributions = [], [], []
    for name, candidates, _ in rows:
        if not available.issuperset(candidates):
            continue
        for candidate in candidates:
            working = set_columns(working, {names[candidate]: cells[names[candidate]]}, scalar=True, copy=False)
        prediction = explainer.predict(working)
        means.append(np.mean(prediction))
        value = ':'.join(_format(display[names[i]]) for i in candidates)
        selected.append((name, value))
        available.difference_update(candidates)
        if keep_distributions:
            distributions.append(_distribution(name, name + '=' + value, prediction, explainer.label, namespace))
    cumulative = np.hstack((baseline, means, target))
    contribution = np.hstack((cumulative[0], np.diff(cumulative)))
    contribution[-1] = cumulative[-1]
    result = _frame({'variable_name': ['intercept', *[name for name, _ in selected], ''],
                      'variable_value': ['', *[value for _, value in selected], ''],
                      'variable': ['intercept', *[name + ' = ' + value for name, value in selected], 'prediction'],
                      'cumulative': cumulative, 'contribution': contribution, 'sign': np.sign(contribution),
                      'position': np.arange(len(selected) + 1, -1, -1),
                      'label': [explainer.label] * len(cumulative)}, namespace)
    return result, distributions


def _shap_path(predict, model, data, names, cells, path):
    working = copy_for_edit(data)
    predictions = np.empty(len(path) + 1)
    predictions[0] = predict(model, working).mean()
    for i, candidate in enumerate(path):
        name = names[candidate]
        working = set_columns(working, {name: cells[name]}, scalar=True, copy=False)
        predictions[i + 1] = predict(model, working).mean()
    return np.diff(predictions)


def _random_shap_path(predict, model, data, names, cells, rng):
    path = rng.choice(len(names), len(names), replace=False)
    return path, _shap_path(predict, model, data, names, cells, path)


class ShapTable(IndexedTable):
    """Typed numeric contributions plus one-row original-value metadata."""

    def __init__(self, data, observation):
        super().__init__(data)
        self.observation = observation

    def _dalex_plot_data(self):
        # The plot uses formatted captions and numeric contributions only.
        # Flattening mixed original feature values would lose their types and
        # unnecessarily materialize the legacy pandas sidecar.
        return self.data

    def _dalex_pandas_view(self):
        import pandas as pd
        from .._dataframe import to_pandas
        result = to_pandas(self.data)
        observation = to_pandas(self.observation)
        pieces = []
        for batch in result.B.unique():
            piece = result.loc[result.B == batch].copy()
            names = piece.variable_name.to_list()
            piece.insert(3, 'variable_value', observation.loc[:, names].values.reshape(-1))
            piece.index = np.arange(len(piece))
            pieces.append(piece)
        return pd.concat(pieces) if pieces else result.assign(variable_value=[])

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        namespace = nw.get_native_namespace(nw.from_native(self.data)).__name__
        if backend == 'pandas' or pandas_view is not None or backend == 'native' and namespace == 'pandas':
            frame = pandas_view.copy() if pandas_view is not None else self._dalex_pandas_view()
            if partition is not None:
                frame = frame.loc[frame.variable_name == partition].copy()
                reference = as_eager_dataframe(self.observation).get_column(partition)
                if isinstance(reference.dtype, (nw.Datetime, nw.Duration)):
                    frame = nw.from_native(frame).with_columns(nw.col('variable_value').cast(reference.dtype)).to_native()
            if backend == 'pandas':
                return frame
            target = nw.get_native_namespace(nw.from_native(self.data)).__name__ if backend == 'native' else backend
            return export_result_frame(frame, target)
        frame = nw.from_native(self.data).with_row_index('__position__')
        source = as_eager_dataframe(self.observation)
        names = frame['variable_name'].unique(maintain_order=True).to_list()
        if partition is not None:
            if partition not in names:
                raise ValueError(f'Unknown contribution partition {partition!r}')
            names = [partition]
        pieces = []
        for name in names:
            piece = frame.filter(nw.col('variable_name') == name)
            original = source.select(source.get_column(name)).to_native()
            repeated = nw.from_native(take_rows(original, np.zeros(len(piece), dtype=np.int64)))
            piece = piece.with_columns(repeated.get_column(name).alias('variable_value'))
            pieces.append(piece.to_native())
        result = nw.from_native(concat_compatible(pieces)).sort('__position__').drop('__position__')
        result = result.select(['variable', 'contribution', 'variable_name', 'variable_value', 'sign', 'label', 'B'])
        return export_result_frame(result.to_native(), backend)


def shap(explainer, observation, path, keep_distributions, B, processes, random_state):
    if B < 1:
        raise ValueError('B must be positive')
    names = list(column_names(explainer.data))
    cells = observation_values(observation)
    display = observation_display_values(observation, cells)
    captions = [str(name) + ' = ' + _format(display[name]) for name in names]
    namespace = nw.get_native_namespace(as_eager_dataframe(explainer.data))
    if processes == 1:
        rng = get_random_state(random_state)
        paths = [_random_shap_path(explainer.predict_function, explainer.model, explainer.data, names, cells, rng) for _ in range(B)]
    else:
        generators = [default_rng(seed) for seed in SeedSequence(random_state).spawn(B)]
        arguments = [(explainer.predict_function, explainer.model, explainer.data, names, cells, rng) for rng in generators]
        with mp.get_context('spawn').Pool(processes) as pool:
            paths = pool.starmap(_random_shap_path, arguments)
    records = []
    for batch, (positions, contributions) in enumerate(paths, 1):
        for candidate, value in zip(positions, contributions):
            records.append((captions[candidate], value, names[candidate], np.sign(value), explainer.label, batch))
    if path is not None:
        if isinstance(path, str) and path == 'average':
            matrix = np.empty((B, len(names)))
            for i, (positions, contributions) in enumerate(paths):
                matrix[i, positions] = contributions
            means = np.nanmean(matrix, axis=0)
            # pandas pivot sorts its caption index before abs-sort. Preserve
            # this input ordering for its default quicksort tie behavior.
            alphabetic = np.argsort(np.asarray(captions, dtype=object))
            magnitudes = np.abs(means[alphabetic])
            valid = np.flatnonzero(~np.isnan(magnitudes))[::-1]
            descending = valid[np.argsort(magnitudes[valid], kind='quicksort')][::-1]
            positions = alphabetic[np.r_[descending, np.flatnonzero(np.isnan(magnitudes))]]
            values = means[positions]
        else:
            positions = path
            values = _shap_path(explainer.predict_function, explainer.model, explainer.data, names, cells, path)
        for candidate, value in zip(positions, values):
            records.append((captions[candidate], value, names[candidate], np.sign(value), explainer.label, 0))
    columns = {name: [record[i] for record in records] for i, name in enumerate(
        ['variable', 'contribution', 'variable_name', 'sign', 'label', 'B'])}
    result = ShapTable(_frame(columns, namespace), _snapshot(observation))
    if keep_distributions:
        distribution = _distribution('all_data', 'all data', explainer.predict(explainer.data), explainer.label, namespace)
    else:
        distribution = None
    prediction = explainer.predict(observation)[0]
    baseline = explainer.predict(explainer.data).mean()
    return result, prediction, baseline, distribution
