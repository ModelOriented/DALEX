"""Native table preparation shared by explanation plots.

An explicitly read or edited pandas compatibility view remains authoritative.
Plotting must not create that view as a side effect of inspecting native results.
"""
import narwhals.stable.v2 as nw
import numpy as np

from ._explanation import export_result_frame


def descending_order(values):
    """Legacy pandas quicksort order, with missing numeric values last."""
    values = np.asarray(values)
    missing = np.isnan(values)
    positions = np.flatnonzero(~missing)[::-1]
    order = positions[np.argsort(values[positions], kind='quicksort')][::-1]
    return np.concatenate([order, np.flatnonzero(missing)])


def result_frame(explanation, *, table='result'):
    cached = explanation.__dict__.get('_compatibility_' + table)
    if cached is not None:
        return nw.from_native(cached, eager_only=True)
    original = explanation.__dict__.get(table)
    if callable(getattr(type(original), '_dalex_plot_data', None)):
        return nw.from_native(original._dalex_plot_data(), eager_only=True)
    return nw.from_native(explanation.result_as(table=table), eager_only=True)


def concat_frames(frames, *, how='vertical'):
    """Combine plot tables, converting only when their backends differ."""
    frames = list(frames)
    # Prefer an existing native backend for mixed-backend plots, so a pandas
    # companion never forces the native explanation through pandas.
    backend = next((nw.get_native_namespace(frame).__name__ for frame in frames
                    if nw.get_native_namespace(frame).__name__ != 'pandas'), 'pandas')
    frames = [nw.from_native(export_result_frame(frame.to_native(), backend), eager_only=True)
              if nw.get_native_namespace(frame).__name__ != backend else frame for frame in frames]
    if backend != 'pandas':
        # Native concatenation is stricter than pandas for otherwise compatible
        # model outputs (for example int32 predictions beside float64). Promote
        # those columns deliberately, without rounding exact large integers.
        schemas = [frame.schema for frame in frames]
        numeric_types = {dtype.__name__.lower(): dtype for dtype in
                         [nw.Int8, nw.Int16, nw.Int32, nw.Int64, nw.UInt8, nw.UInt16,
                          nw.UInt32, nw.UInt64, nw.Float32, nw.Float64]}
        numeric_types['bool'] = nw.Boolean
        for name in dict.fromkeys(name for schema in schemas for name in schema):
            present = [(i, schema[name]) for i, schema in enumerate(schemas) if name in schema]
            if all(dtype == present[0][1] for _, dtype in present):
                if backend == 'pyarrow' and present[0][1] == nw.String:
                    physical = [frames[index].to_native().schema.field(name).type for index, _ in present]
                    if any(dtype != physical[0] for dtype in physical):
                        # pandas 3 exports large_string, while other producers
                        # use string/string_view. Narwhals calls all of them
                        # String; normalize physical storage only when needed.
                        for index, _ in present:
                            frames[index] = frames[index].with_columns(nw.col(name).cast(nw.String))
                continue
            dtypes = [dtype for index, dtype in present if not
                      (dtype == nw.Unknown and frames[index][name].null_count() == len(frames[index]))]
            if not dtypes:
                continue
            if all(dtype.is_numeric() or dtype == nw.Boolean for dtype in dtypes) and not any(
                isinstance(dtype, nw.Decimal) for dtype in dtypes
            ):
                common = np.result_type(*[np.dtype('bool' if dtype == nw.Boolean else str(dtype).lower()) for dtype in dtypes])
                target = numeric_types[common.name]
                if common.kind == 'f':
                    for index, dtype in present:
                        if dtype.is_integer():
                            series = frames[index][name]
                            extremes = (series.min(), series.max())
                            if any(value is not None and abs(value) > 2**53 for value in extremes):
                                raise ValueError('Combining plot tables would round large integers; plot these models separately')
            elif all(isinstance(dtype, (nw.String, nw.Categorical, nw.Enum)) for dtype in dtypes):
                target = nw.String
            elif all(dtype == dtypes[0] for dtype in dtypes):
                target = dtypes[0]  # A missing column/Null dtype beside typed values.
            else:
                continue  # Let the backend report genuinely incompatible schemas.
            for index, _ in present:
                frames[index] = frames[index].with_columns(nw.col(name).cast(target))
    return nw.concat(frames, how=how)
