"""Native Aspect plot numbers with separate, small display metadata records."""
from dataclasses import dataclass, replace

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.dependencies import get_pandas

from ._table import AspectTable
from .._dataframe import column_values
from .._plotting import concat_frames
from ..predict_explanations._ceteris_paribus._plotting import _display_column


METADATA = ['variable_names', 'variable_values', 'vars_min_depend']


@dataclass
class PlotData:
    frame: object
    metadata: list
    key: str
    pandas_rounding: bool
    reset_rounding_index: bool = False

    def rows(self):
        result = []
        for row in self.frame.iter_rows(named=True):
            position = row.pop(self.key)
            row.update(self.metadata[position])
            result.append(row)
        return result

    def with_frame(self, frame):
        return replace(self, frame=frame)

    def take(self, positions):
        return self.with_frame(self.frame[list(positions), :])

    def rounded(self, name, function, digits):
        series = self.frame[name]
        values = series.to_native().copy() if self.pandas_rounding else series.to_numpy().copy()
        if self.pandas_rounding and self.reset_rounding_index:
            values = values.reset_index(drop=True)
        values = np.asarray(function(values, digits))
        if values.ndim == 0:
            values = np.repeat(values, len(self.frame))
        return self.with_frame(self.frame.with_columns(nw.new_series(
            name, values, backend=nw.get_native_namespace(self.frame))))


def _group_values(observation, names):
    if names is None or len(names) == 0:
        return np.asarray([])
    frame = nw.from_native(observation, eager_only=True).select(list(names))
    namespace = nw.get_native_namespace(frame).__name__
    if namespace == 'pandas':
        return frame.to_numpy()[0]
    dtypes = list(frame.schema.values())
    temporal = all(isinstance(dtype, nw.Datetime) and dtype.time_zone is None or
                   dtype == nw.Date and namespace == 'polars' for dtype in dtypes)
    durations = all(isinstance(dtype, nw.Duration) for dtype in dtypes)
    if temporal or durations:
        arrays = [column_values(frame.to_native(), name) for name in names]
        arrays = [array.astype('datetime64[ms]') if dtype == nw.Date else array
                  for array, dtype in zip(arrays, dtypes)]
        unit = np.result_type(*[array.dtype for array in arrays])
        if int(get_pandas().__version__.split('.')[0]) < 2:
            unit = np.dtype('datetime64[ns]' if temporal else 'timedelta64[ns]')
        return np.asarray([array[0] for array in arrays], dtype=unit)
    values = [_display_column(frame, name, False, False)[0] for name in names]
    numeric = all(dtype.is_numeric() and not isinstance(dtype, nw.Decimal) or dtype == nw.Boolean for dtype in dtypes)
    if numeric and not (any(dtype == nw.Boolean for dtype in dtypes) and any(dtype != nw.Boolean for dtype in dtypes)):
        physical = [np.dtype('float64' if dtype.is_integer() and frame[name].null_count() else
                             'bool' if dtype == nw.Boolean else str(dtype).lower())
                    for name, dtype in zip(names, dtypes)]
        if not any(dtype == nw.Boolean and frame[name].null_count() for name, dtype in zip(names, dtypes)):
            return np.asarray(values, dtype=np.result_type(*physical))
    return np.asarray(values, dtype=object)


def as_plot_data(value):
    if isinstance(value, PlotData):
        return value
    if isinstance(value, AspectTable):
        original = nw.from_native(value.data, eager_only=True)
        if nw.get_native_namespace(original).__name__ == 'pandas':
            # Exact legacy dtype/scalar publication is a pandas display
            # boundary. No compatibility attribute is published as a side effect.
            return as_plot_data(value._dalex_pandas_view())
        columns = [name for name in value._columns() if name not in METADATA]
        frame = original.select(columns)
        metadata = []
        for index, names in enumerate(value.names):
            row = {'variable_names': names[0] if value.single else names,
                   'vars_min_depend': value.pairs[index]}
            if value.observation is not None:
                values = _group_values(value.observation, names)
                row['variable_values'] = (values[0] if value.single else values.tolist()
                                          if value.value_lists[index] else values)
            metadata.append(row)
        pandas_rounding = False
    else:
        original = nw.from_native(value, eager_only=True)
        meta = {name: original[name].to_list() for name in METADATA if name in original.columns}
        metadata = [dict(zip(meta, values)) for values in zip(*meta.values())] if meta else [{} for _ in range(len(original))]
        frame = original.drop(list(meta)) if meta else original
        pandas_rounding = nw.get_native_namespace(original).__name__ == 'pandas'
    key = '__dalex_aspect_row__'
    while key in frame.columns:
        key += '_'
    return PlotData(frame.with_row_index(key), metadata, key, pandas_rounding)


def plot_data(obj, table='result'):
    return as_plot_data(obj.__dict__.get('_compatibility_' + table, obj.__dict__.get(table)))


def ordered(data, order):
    data = as_plot_data(data)
    ranks = {name: i for i, name in enumerate(order)}
    rows = data.rows()
    positions = sorted(range(len(data.frame)), key=lambda i: ranks.get(rows[i]['variable_names'], np.inf))
    return replace(data.take(positions), reset_rounding_index=True)


def concat_data(values):
    values = [as_plot_data(value) for value in values]
    frames = [value.frame.drop(value.key) for value in values]
    metadata = [value.metadata[position] for value in values for position in value.frame[value.key].to_list()]
    frame = concat_frames(frames, how='diagonal')
    key = '__dalex_aspect_row__'
    while key in frame.columns:
        key += '_'
    return PlotData(frame.with_row_index(key), metadata, key,
                    nw.get_native_namespace(frame).__name__ == 'pandas')


def complete(data):
    pd = get_pandas()
    positions = []
    for index, row in enumerate(data.rows()):
        if not any(value is None or value is pd.NA or value is pd.NaT or
                   isinstance(value, (float, np.floating)) and np.isnan(value) for value in row.values()):
            positions.append(index)
    return data.take(positions)
