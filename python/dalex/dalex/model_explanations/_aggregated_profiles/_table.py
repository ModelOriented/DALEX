"""Typed aggregate partitions with a compatible combined pandas publication."""
import numpy as np
import narwhals.stable.v2 as nw
from ..._explanation import export_result_frame
from ...predict_explanations._ceteris_paribus._table import concat_compatible
from ._native import AggregatedTable, _large_integer


class AggregatedPartitions:
    def __init__(self, frames, index, groups, group_dtypes, numerical):
        self.frames = frames
        self.index = index
        self.groups = groups
        self.group_dtypes = group_dtypes
        self.numerical = numerical

    def _dalex_set_column(self, name, value):
        self.frames = [nw.from_native(frame, eager_only=True).with_columns(nw.lit(value).alias(name)).to_native()
                       for frame in self.frames]

    def _dalex_plot_data(self):
        """Prepare display columns without publishing the pandas result view."""
        from ..._plotting import concat_frames
        pieces = []
        for data in self.frames:
            frame = nw.from_native(data, eager_only=True)
            namespace = nw.get_native_namespace(frame)
            if self.groups:
                columns = []
                numeric = []
                for name in self.groups:
                    series = frame[name]
                    dtype = self.group_dtypes.get(name)
                    values = series.to_list()
                    # Match the legacy table's scalar display, including its
                    # row-wise numeric promotion and missing group spellings.
                    if namespace.__name__ != 'pandas' and (
                        isinstance(series.dtype, (nw.Datetime, nw.Duration)) or
                        series.dtype == nw.Date and namespace.__name__ == 'polars'
                    ):
                        # Scalar display is an intentional compatibility
                        # boundary. Integer epochs avoid Python datetime's
                        # microsecond truncation without creating pandas tables.
                        from narwhals.dependencies import get_pandas
                        pd = get_pandas()
                        counts = series.cast(nw.Int64).to_list()
                        if isinstance(series.dtype, nw.Duration):
                            values = [pd.NaT if value is None else pd.Timedelta(value, unit=series.dtype.time_unit)
                                      for value in counts]
                        else:
                            unit = 'D' if series.dtype == nw.Date else series.dtype.time_unit
                            zone = None if series.dtype == nw.Date else series.dtype.time_zone
                            values = [pd.NaT if value is None else pd.Timestamp(value, unit=unit, tz=zone)
                                      for value in counts]
                    elif namespace.__name__ == 'pandas':
                        values = list(frame.to_native()[name].astype(dtype).array) if dtype is not None else values
                        if dtype == 'object':
                            values = [np.nan if missing else value for value, missing in zip(values, series.is_null().to_list())]
                    elif dtype == 'object':
                        values = [np.nan if value is None else value for value in values]
                    elif series.dtype.is_integer() and series.null_count():
                        values = [np.nan if value is None else float(value) for value in values]
                    elif series.dtype.is_float() and not (series.null_count() and series.is_nan().any()):
                        values = [np.nan if value is None else value for value in values]
                    columns.append(values)
                    distinct_missing = (namespace.__name__ != 'pandas' and series.dtype.is_float()
                                        and series.null_count() and series.is_nan().any())
                    numeric.append(dtype != 'object' and series.dtype.is_numeric() and not distinct_missing)
                if all(numeric):
                    if any(frame[name].dtype.is_float() for name in self.groups):
                        # pandas nullable numeric rows share a floating dtype
                        # even when pd.NA forces NumPy's array to object.
                        columns = [[float(value) if isinstance(value, (int, float, np.number)) else value
                                    for value in values] for values in columns]
                    rows = np.asarray(columns).T
                else:
                    rows = zip(*columns)
                labels = ['_'.join(map(str, row)) for row in rows]
                model_labels = [str(label) + '_' + group for label, group in zip(frame['_label_'].to_list(), labels)]
                frame = frame.drop(self.groups).with_columns(
                    nw.new_series('_groups_', labels, backend=namespace),
                    nw.new_series('_label_', model_labels, backend=namespace))
            if self.numerical and not _large_integer(frame, '_x_') and '_x_kind_' not in frame.columns:
                frame = frame.with_columns(nw.col('_x_').cast(nw.Float64))
            if '_x_kind_' in frame.columns:
                frame = frame.drop('_x_kind_')
            pieces.append(frame)
        return concat_frames(pieces).to_native()

    def _dalex_plot_x_is_numeric(self):
        # The legacy projection uses object storage for exact large integers
        # and distinct null/NaN keys; preserve its bar-plot selection.
        return self.numerical and not any(
            _large_integer(frame, '_x_') or
            ('_x_kind_' in frame.columns and 'null' in frame['_x_kind_'].to_list())
            for frame in (nw.from_native(data) for data in self.frames))

    def _dalex_pandas_view(self):
        import pandas as pd
        pieces = []
        for data in self.frames:
            frame = nw.from_native(data, eager_only=True)
            piece = AggregatedTable(data, None, self.groups, self.group_dtypes)._dalex_pandas_view()
            kinds = frame['_x_kind_'].to_list() if '_x_kind_' in frame.columns else []
            separate = 'null' in kinds
            if self.numerical and (_large_integer(frame, '_x_') or separate):
                values = np.empty(len(frame), dtype=object)
                values[:] = frame['_x_'].to_list()
                if separate:
                    values[:] = [None if kind == 'null' else np.nan if kind == 'nan' else value
                                 for value, kind in zip(values, kinds)]
                piece['_x_'] = values
            elif self.numerical:
                piece['_x_'] = piece['_x_'].astype(float)
            if '_x_kind_' in piece:
                piece = piece.drop(columns='_x_kind_')
            pieces.append(piece)
        result = pd.concat(pieces, ignore_index=True)
        if self.index is not None:
            result.index = self.index
        return result

    def _selected(self, partition):
        frames = [nw.from_native(data, eager_only=True) for data in self.frames]
        if partition is not None:
            variable, label = partition if isinstance(partition, tuple) else (partition, None)
            selected = []
            for frame in frames:
                mask = nw.col('_vname_') == variable
                if label is not None:
                    mask = mask & (nw.col('_label_') == label)
                frame = frame.filter(mask)
                if len(frame):
                    selected.append(frame)
            frames = selected
            if not frames:
                raise ValueError(f'Unknown aggregate result partition {partition!r}')
        return frames

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        frames = self._selected(partition)
        namespace = nw.get_native_namespace(frames[0]).__name__
        if backend == 'pandas':
            result = pandas_view.copy() if pandas_view is not None else self._dalex_pandas_view()
            if partition is not None:
                variable, label = partition if isinstance(partition, tuple) else (partition, None)
                result = result.loc[result['_vname_'] == variable]
                if label is not None:
                    result = result.loc[result['_label_'] == label]
            return result.copy()
        target = namespace if backend == 'native' else backend
        if pandas_view is not None:
            result = pandas_view.copy()
            if partition is not None:
                variable, label = partition if isinstance(partition, tuple) else (partition, None)
                result = result.loc[result['_vname_'] == variable]
                if label is not None:
                    result = result.loc[result['_label_'] == label]
            # Validate any combined scalar promotion before crossing the
            # publication boundary. A partition retains the exact grid type.
            concat_compatible([frame.to_native() for frame in frames])
            if target == 'pandas':
                return result
            if self.numerical and (any(_large_integer(frame, '_x_') for frame in frames)
                                   or any('_x_kind_' in frame.columns for frame in frames)):
                import pyarrow as pa
                import pandas as pd
                reference = frames[0].to_arrow().schema.field('_x_').type
                values = [None if value is None or value is pd.NA or value is pd.NaT else value for value in result['_x_']]
                column = pa.array(values, type=reference, from_pandas=False)
                arrow = nw.from_native(result.drop(columns='_x_'), eager_only=True).to_arrow()
                arrow = arrow.append_column('_x_', column).select(result.columns)
                return nw.from_arrow(arrow, backend=target).to_native()
            return export_result_frame(result, target)
        return export_result_frame(concat_compatible([frame.to_native() for frame in frames]), target)
