"""Typed native CP partitions with a backwards-compatible pandas projection."""
import narwhals.stable.v2 as nw
from ..._explanation import export_result_frame


def _legacy_column(column):
    """Historical pandas publication dtype, without deprecated floating flags."""
    import pandas as pd
    converted = column.convert_dtypes()
    if pd.api.types.is_float_dtype(converted.dtype) and not isinstance(column.dtype, pd.ArrowDtype):
        converted = column.infer_objects()
    return converted


def concat_compatible(frames):
    """Concatenate compatible typed tables without silently stringifying cells."""
    frames = [nw.from_native(frame, eager_only=True) for frame in frames]
    if len(frames) == 1:
        return frames[0].clone().to_native()
    for name in frames[0].columns:
        dtypes = [frame.schema[name] for frame in frames]
        if all(dtype == dtypes[0] for dtype in dtypes):
            continue
        if all(dtype.is_numeric() for dtype in dtypes):
            for frame, dtype in zip(frames, dtypes):
                if dtype.is_integer() and any(value is not None and abs(value) > 2**53
                                              for value in frame[name].to_list()):
                    raise ValueError('Combining profile partitions would round large integers; export a partition')
            frames = [frame.with_columns(nw.col(name).cast(nw.Float64)) for frame in frames]
        elif all(isinstance(dtype, (nw.String, nw.Categorical, nw.Enum)) for dtype in dtypes):
            frames = [frame.with_columns(nw.col(name).cast(nw.String)) for frame in frames]
        else:
            raise ValueError('These profile partitions have different scalar types; export with partition=variable')
    return nw.concat(frames, how='vertical').to_native()


def export_frame(data, backend):
    return export_result_frame(data, backend)


class ProfileTables:
    """Keep `_original_` typed separately for each explained variable.

    pandas permits a single object column mixing floats, text, dates and nested
    cells. Native exports require a partition where no portable common scalar
    type exists, instead of rounding or converting values to text silently.
    """

    def __init__(self, frames, indexes, variables):
        self.frames = list(frames)
        self.indexes = list(indexes)
        self.variables = list(variables)

    def _dalex_set_column(self, name, value):
        self.frames = [nw.from_native(frame, eager_only=True).with_columns(nw.lit(value).alias(name)).to_native()
                       for frame in self.frames]

    def _dalex_pandas_view(self):
        import pandas as pd
        from ..._dataframe import to_pandas
        tables = []
        for frame, index in zip(self.frames, self.indexes):
            table = to_pandas(frame)
            table.index = index
            tables.append(table)
        result = pd.concat(tables)
        for name in self.variables:
            column = _legacy_column(result[name])
            result[name] = column.array
        return result

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        namespace = nw.get_native_namespace(nw.from_native(self.frames[0])).__name__
        if pandas_view is not None or backend == 'pandas' or backend == 'native' and namespace == 'pandas':
            table = pandas_view.copy() if pandas_view is not None else self._dalex_pandas_view()
            if partition is not None:
                if partition not in self.variables:
                    raise ValueError(f'Unknown profile partition {partition!r}')
                table = table.loc[table['_vname_'] == partition].copy()
            if backend == 'pandas':
                return table
            target = nw.get_native_namespace(nw.from_native(self.frames[0])) if backend == 'native' else backend
            frame = nw.from_native(table)
            if partition is not None:
                reference = nw.from_native(self.frames[self.variables.index(partition)])
                # Legacy mixed `_original_` columns use object storage. Restore
                # the selected partition's temporal dtype before Arrow export
                # so object scalar inference cannot truncate nanoseconds.
                casts = [nw.col(name).cast(dtype) for name, dtype in reference.schema.items()
                         if name in frame.columns and isinstance(dtype, (nw.Datetime, nw.Duration))
                         and frame.schema[name] != dtype]
                frame = frame.with_columns(casts)
            return export_frame(frame.to_native(), target.__name__ if not isinstance(target, str) else target)
        selected = [frame for variable, frame in zip(self.variables, self.frames)
                    if partition is None or variable == partition]
        if not selected:
            raise ValueError(f'Unknown profile partition {partition!r}')
        return export_frame(concat_compatible(selected), backend)


class ObservationTable:
    """Native observation metadata with its original pandas index at publication."""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def _dalex_pandas_view(self):
        from ..._dataframe import to_pandas
        result = to_pandas(self.data)
        result.index = self.index
        return result

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        if partition is not None:
            raise ValueError('Observation tables have no variable partitions')
        if backend == 'pandas':
            return pandas_view.copy() if pandas_view is not None else self._dalex_pandas_view()
        if pandas_view is not None:
            target = nw.get_native_namespace(nw.from_native(self.data)) if backend == 'native' else backend
            return export_frame(pandas_view, target.__name__ if not isinstance(target, str) else target)
        return export_frame(self.data, backend)
