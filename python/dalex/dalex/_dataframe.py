"""Narwhals-based dataframe interoperability for DALEX.

Operate on eager native dataframes and return them to the model in their original
backend.  The pandas-specific row-taking fast path intentionally preserves the
existing DALEX behavior for non-default pandas indexes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
import narwhals.stable.v2 as nw
from narwhals.dependencies import get_polars, get_pyarrow, is_polars_dataframe, is_pyarrow_table


def _legacy_polars_frame(data):
    # Polars 2 protects its dataframe with internal read/write locks. Earlier
    # export methods can hold a mutable Python borrow while releasing the GIL.
    # Avoid the per-column Arc clone cost on wide modern frames.
    return is_polars_dataframe(data) and get_polars().__version__.startswith(("0.", "1."))


def as_eager_dataframe(data):
    """Validate a Narwhals-supported *eager* dataframe without converting it."""
    try:
        frame = nw.from_native(data, eager_only=True)
    except TypeError as exc:
        raise TypeError(
            "'data' must be a pandas.DataFrame, numpy.ndarray, "
            "or Narwhals-supported eager dataframe"
        ) from exc
    if not isinstance(frame, nw.DataFrame):
        raise TypeError("'data' must be a two-dimensional eager dataframe")
    return frame


def as_numpy_vector(value, name):
    """Normalize native vectors or one-column eager DataFrame metadata."""
    arrow = get_pyarrow()
    if arrow is not None and isinstance(value, arrow.Array):
        # Narwhals represents Arrow Series as ChunkedArray, whereas pa.array()
        # returns Array. Wrapping one existing chunk does not copy its buffers.
        value = arrow.chunked_array([value])
    try:
        series = nw.from_native(value, series_only=True)
    except TypeError:
        frame = as_eager_dataframe(value)
        if len(frame.columns) != 1:
            raise ValueError(f"'{name}' must have one column") from None
        return to_numpy(frame.to_native()).reshape(-1)
    native = series.to_native()
    if hasattr(getattr(native, "type", None), "index_type"):
        return native.cast(native.type.value_type).to_numpy()
    return series.to_numpy()


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    if _legacy_polars_frame(data):
        # Export methods in older Polars versions borrow the frame mutably.
        # A cheap buffer-sharing clone isolates simultaneous explanation calls.
        data = data.clone()
    if is_pyarrow_table(data) and any(
        hasattr(column.type, "index_type") for column in data.columns
    ):
        # Arrow dictionary.to_numpy can expose a null index's physical value
        # (e.g. an unused Enum category). Decode before crossing the boundary.
        return np.column_stack([column_values(data, name) for name in data.column_names])
    if is_pyarrow_table(data):
        # Pandas schema hints can consume a physical feature as an index or
        # restore a stale extension dtype. A model matrix uses Arrow fields.
        data = data.replace_schema_metadata(None)
    return as_eager_dataframe(data).to_numpy()


def column_names(data):
    """Feature labels, including backends whose .columns contains column arrays."""
    if isinstance(data, pd.DataFrame) or is_polars_dataframe(data):
        return data.columns
    if is_pyarrow_table(data):
        return data.column_names
    return as_eager_dataframe(data).columns


def native_namespace(data):
    """Identify storage without validating pandas' legacy column-label layout."""
    if isinstance(data, pd.DataFrame):
        return pd
    return nw.get_native_namespace(as_eager_dataframe(data))


def iter_column_arrays(data, numeric_only=False):
    """Inspect features without materializing a full mixed/structured matrix."""
    if isinstance(data, pd.DataFrame):
        yield from (series.to_numpy() for _, series in data.items())
    else:
        frame = as_eager_dataframe(data)
        for name in frame.columns:
            column = frame.get_column(name)
            if numeric_only and not (column.dtype.is_numeric() or isinstance(
                column.dtype, (nw.Boolean, nw.Decimal, nw.Categorical, nw.Enum, nw.Object)
            )):
                # Datetime/nested/string features cannot equal a numeric target.
                # Avoid optional conversions unsupported by older backends.
                continue
            yield column_values(data, name) if is_pyarrow_table(data) else column.to_numpy()


def _row_positions(positions, size):
    """Validate positions before narrowing integers (uint64 must not wrap)."""
    index = np.asarray(positions)
    if index.ndim != 1 or (index.size and index.dtype.kind not in "iu"):
        raise TypeError("row positions must be a one-dimensional integer sequence")
    if np.any(index >= size) or (index.dtype.kind != "u" and np.any(index < -size)):
        raise IndexError("row position out of bounds")
    return index.astype(np.int64, copy=False)


def take_rows(data, positions):
    """Take positional rows (including repeats) while preserving native backend.

    Pandas .iloc is kept for its existing index-retention semantics. On other
    backends Narwhals gathers the requested positional order.
    """
    size = data.shape[0]
    index = _row_positions(positions, size)
    if isinstance(data, pd.DataFrame):
        return data.iloc[index, :]
    frame = as_eager_dataframe(data)
    if not index.size:
        return frame.head(0).to_native()
    index = np.where(index < 0, index + size, index)
    return frame[index, :].to_native()


def shuffle_columns(data, columns, positions):
    """Permute one or more columns with a shared row permutation.

    Narwhals preserves native backends; pandas assignment retains legacy index semantics.
    """
    if len(positions) != data.shape[0]:
        raise ValueError("permutation must have the same length as data")
    index = _row_positions(positions, data.shape[0])
    if isinstance(data, pd.DataFrame):
        if not data.columns.is_unique:
            # Duplicate pandas labels select every matching physical column.
            # Resolve positions once, then assign each array independently:
            # result[name] and other[name] would select a whole subframe.
            selected = pd.Series(np.arange(data.shape[1]), index=data.columns).loc[columns]
            selected = selected.to_numpy() if isinstance(selected, pd.Series) else np.asarray([selected])
            result = data.copy()
            for position in selected:
                result.isetitem(int(position), data.iloc[index, int(position)].array)
            return result
        multilevel = isinstance(data.columns, pd.MultiIndex)
        if multilevel:
            # A tuple can be one complete feature label or a partial level
            # selector. Resolve pandas' selection on positional metadata, so
            # prefixes still shuffle their features jointly and retain every
            # original column level/name and extension dtype.
            selected = pd.Series(np.arange(data.shape[1]), index=data.columns).loc[columns]
            selected = selected.to_numpy() if isinstance(selected, pd.Series) else np.asarray([selected])
            other = data.iloc[index, selected]
        else:
            scalar = pd.api.types.is_scalar(columns) or isinstance(columns, tuple) and columns in data.columns
            names = [columns] if scalar else list(columns)
            other = data.loc[:, names].iloc[index, :]
        result = data.copy()
        # Extension arrays assign positionally without losing categorical/nullable
        # dtypes on older pandas, or aligning shuffled row labels.
        for position, name in enumerate(other.columns):
            result[name] = (other.iloc[:, position] if multilevel else other[name]).array
        return result
    names = [columns] if np.isscalar(columns) else list(columns)
    frame = as_eager_dataframe(data)
    # Keep the Narwhals wrapper throughout the operation. Round-tripping the
    # selected native frame through take_rows used to validate/wrap it twice more.
    if not index.size:
        return frame.to_native()
    index = np.where(index < 0, index + len(frame), index)
    other = frame[index, names]
    result = frame.with_columns(other.get_column(col) for col in names).to_native()
    if is_pyarrow_table(data):
        # Narwhals replaces Arrow columns by name, which drops their Field
        # metadata. A permutation cannot change types, so reuse the complete
        # reference schema over the existing immutable buffers (no value copy).
        result = get_pyarrow().Table.from_arrays(result.columns, schema=data.schema)
    return result


def to_pandas(data):
    """Materialize an explanation result/plot boundary; never use for model input."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if is_pyarrow_table(data):
        # Arrow has no dataframe index: every physical column is a feature.
        return data.to_pandas(ignore_metadata=True)
    if _legacy_polars_frame(data):
        data = data.clone()
    return as_eager_dataframe(data).to_pandas()


def copy_for_edit(data):
    """Own pandas work buffers; native expression edits already return a frame."""
    return data.copy(deep=True) if isinstance(data, pd.DataFrame) else data


def column_values(data, name, *, preserve_nulls=False):
    if isinstance(data, pd.DataFrame):
        if preserve_nulls and isinstance(data[name].dtype, pd.api.extensions.ExtensionDtype):
            return data[name].to_numpy(dtype=object, na_value=None)
        return data[name].to_numpy()
    temporal = as_eager_dataframe(data).get_column(name)
    if isinstance(temporal.dtype, (nw.Datetime, nw.Duration)):
        # Older Polars builds can crash exporting timezone/nanosecond Series to
        # NumPy directly. Integer epoch buffers also avoid Python datetime's
        # microsecond limit and float promotion of nullable nanosecond counts.
        counts = temporal.cast(nw.Int64).to_list()
        absent = temporal.is_null().to_list()
        integers = np.array([np.iinfo(np.int64).min if missing else int(value)
                             for value, missing in zip(counts, absent)], dtype=np.int64)
        kind = 'datetime64' if isinstance(temporal.dtype, nw.Datetime) else 'timedelta64'
        values = integers.view(f'{kind}[{temporal.dtype.time_unit}]')
        if preserve_nulls and any(absent):
            result = np.empty(len(values), dtype=object)
            result[:] = [None if missing else value for value, missing in zip(values, absent)]
            return result
        return values
    if preserve_nulls:
        column = temporal
        if column.dtype.is_numeric() and column.null_count():
            # Float NumPy exports collapse null into NaN and can round nullable
            # integers. Masked feature edits must retain the untouched values.
            values = np.empty(len(column), dtype=object)
            values[:] = column.to_list()
            return values
    if is_pyarrow_table(data):
        column = data.column(name)
        if hasattr(column.type, "index_type"):
            return column.cast(column.type.value_type).to_numpy()
    return temporal.to_numpy()


def _arrow_cell(scalar):
    """Decode nested cells without truncating nanosecond children."""
    arrow = get_pyarrow()
    if not scalar.is_valid:
        return None
    dtype = scalar.type
    if arrow.types.is_timestamp(dtype):
        return pd.Timestamp(scalar.value, unit=dtype.unit, tz=dtype.tz)
    if arrow.types.is_duration(dtype):
        return pd.Timedelta(scalar.value, unit=dtype.unit)
    if arrow.types.is_struct(dtype):
        return {field.name: _arrow_cell(scalar[field.name]) for field in dtype}
    if arrow.types.is_list(dtype) or arrow.types.is_large_list(dtype) or arrow.types.is_fixed_size_list(dtype):
        return [_arrow_cell(value) for value in scalar.values]
    if arrow.types.is_dictionary(dtype):
        return _arrow_cell(scalar.value)
    return scalar.as_py()


def observation_values(data, metadata=None):
    """First-row feature cells, preserving native nulls before interventions.

    ``metadata`` is the already-created pandas display table. Its per-column
    scalars preserve temporal precision and integer values, but its NumPy-backed
    floating columns merge native nulls with NaN. Read the native validity mask
    before using these scalars as model inputs. Result labels keep their existing
    pandas representation.
    """
    if metadata is None:
        if isinstance(data, pd.DataFrame):
            return {name: data[name].iloc[0] for name in column_names(data)}
        values = {}
        frame = as_eager_dataframe(data)
        for name in frame.columns:
            column = frame.get_column(name)
            if column.dtype.is_nested():
                values[name] = _arrow_cell(column.to_arrow()[0])
            else:
                values[name] = column_values(data, name, preserve_nulls=True)[0]
        return values
    values = {name: metadata[name].iloc[0] for name in column_names(data)}
    if not isinstance(data, pd.DataFrame):
        frame = as_eager_dataframe(data)
        for name, absent in zip(frame.columns, frame.head(1).null_count().to_numpy()[0]):
            if absent:
                values[name] = None
    return values


def observation_display_values(data, values=None):
    """Legacy one-row scalar formatting, without making a pandas dataframe."""
    values = observation_values(data) if values is None else values
    if isinstance(data, pd.DataFrame):
        return values
    frame = as_eager_dataframe(data)
    display = dict(values)
    for name, value in values.items():
        dtype = frame.get_column(name).dtype
        if isinstance(dtype, nw.Datetime):
            timestamp = pd.Timestamp(value)
            if dtype.time_zone and not pd.isna(timestamp):
                timestamp = timestamp.tz_localize('UTC').tz_convert(dtype.time_zone)
            display[name] = timestamp
        elif isinstance(dtype, nw.Duration):
            display[name] = pd.Timedelta(value)
        elif value is None and (dtype.is_numeric() and not isinstance(dtype, nw.Decimal)
                                or isinstance(dtype, (nw.Categorical, nw.Enum))):
            display[name] = np.nan
    return display


def _temporal_counts(values, dtype):
    """Epoch counts avoid Python datetime's microsecond precision limit."""
    constructor = pd.Timestamp if isinstance(dtype, nw.Datetime) else pd.Timedelta
    kind = "datetime64" if isinstance(dtype, nw.Datetime) else "timedelta64"
    method = "to_datetime64" if isinstance(dtype, nw.Datetime) else "to_timedelta64"
    counts = []
    for value in values:
        if pd.isna(value):
            counts.append(None)
        else:
            native = getattr(constructor(value), method)()
            counts.append(int(native.astype(f"{kind}[{dtype.time_unit}]").view("i8")))
    return counts


def _updated_pandas_metadata(metadata, promotions):
    """Keep pandas' optional dtype hints consistent with promoted Arrow fields."""
    if not metadata or b'pandas' not in metadata or not promotions:
        return metadata
    try:
        hints = json.loads(metadata[b'pandas'])
    except (ValueError, TypeError):
        return metadata
    if not isinstance(hints, dict) or not isinstance(hints.get('columns'), list):
        return metadata
    changed = False
    for column in hints['columns']:
        if isinstance(column, dict) and column.get('field_name') in promotions:
            dtype = promotions[column['field_name']]
            column.update(pandas_type=dtype, numpy_type=dtype, metadata=None)
            changed = True
    return {**metadata, b'pandas': json.dumps(hints).encode()} if changed else metadata


def _restore_native_fields(result, reference, names, schema):
    """Keep metadata not represented by Narwhals' portable dtype classes."""
    if is_pyarrow_table(reference):
        promotions = {}
        for name in names:
            dtype = schema[name]
            field = reference.schema.field(name)
            column = result.column(name)
            if not dtype.is_numeric() or isinstance(dtype, nw.Decimal):
                column = column.cast(field.type)
            else:
                # Numeric grids may legitimately promote integer features to
                # floats; retain field metadata without truncating split points.
                if field.type != column.type:
                    try:
                        promotions[name] = np.dtype(column.type.to_pandas_dtype()).name
                    except (TypeError, NotImplementedError):
                        promotions[name] = 'object'
                field = field.with_type(column.type)
            result = result.set_column(result.schema.get_field_index(name), field, column)
        result = result.replace_schema_metadata(_updated_pandas_metadata(reference.schema.metadata, promotions))
    elif is_polars_dataframe(reference):
        for name in names:
            if isinstance(schema[name], nw.Decimal):
                # Decimal casting was not implemented by Narwhals 2.0. Use the
                # native reference dtype rather than infer a smaller precision.
                result = result.with_columns(result.get_column(name).cast(reference.schema[name]))
    return result


def _uint64_integer_values(dtype, values):
    # Inference from Python integers defaults to signed int64 in Arrow. Keep
    # the explicit reference type when a nullable UInt64 replacement is exact;
    # fractional interventions still follow the normal floating promotion.
    return isinstance(dtype, nw.UInt64) and all(
        value is None or (isinstance(value, (int, np.integer)) and 0 <= value <= 2**64 - 1)
        for value in values
    )


def set_columns(data, values, *, scalar=False, copy=True):
    """Replace columns positionally, copying pandas sources by default.

    ``scalar=True`` broadcasts observation cells, including list/struct cells.
    Otherwise sequences represent a full replacement column.
    ``copy=False`` can update an already-owned pandas work frame in place;
    native expression backends still return a new frame.
    """
    if isinstance(data, pd.DataFrame):
        result = data.copy() if copy else data
        for name, value in values.items():
            if isinstance(value, pd.Series):
                value = value.array  # replacements are positional, never index-aligned
            if scalar and isinstance(value, (list, tuple, dict, np.ndarray)):
                result[name] = pd.Series([value] * len(data), index=data.index, dtype=data[name].dtype)
            elif value is None or (isinstance(result[name].dtype, pd.CategoricalDtype) and not pd.api.types.is_scalar(value)):
                column = result[name].copy()
                column.iloc[:] = value
                result[name] = column.array
            else:
                try:
                    result.loc[:, name] = value
                except TypeError:
                    # pandas 3 rejects implicit integer -> float widening in
                    # .loc; counterfactual observations may be fractional.
                    result[name] = value
        return result
    frame = as_eager_dataframe(data)
    backend = nw.get_native_namespace(frame)
    # Only inspect columns being replaced. Collecting the entire portable schema
    # for each Shapley/break-down step creates quadratic work on wide frames.
    schema = {name: frame.get_column(name).dtype for name in values}
    expressions = []
    for name, value in values.items():
        dtype = schema[name]
        if scalar or pd.api.types.is_scalar(value) or isinstance(value, dict):
            if value is pd.NA or value is pd.NaT:
                value = None
            if (not dtype.is_numeric() or isinstance(dtype, nw.Decimal)) and pd.api.types.is_scalar(value) and pd.isna(value):
                # pandas represents missing categorical/date cells as NaN/NaT.
                # Native enum/dictionary casts require an actual null scalar.
                value = None
            if dtype.is_nested() and value is not None:
                expressions.append(nw.new_series(name, [value] * len(frame), dtype=dtype, backend=backend).cast(dtype))
                continue
            if isinstance(dtype, (nw.Datetime, nw.Duration)):
                value = _temporal_counts([value], dtype)[0]
            expression = nw.lit(value)
            if (value is None or not dtype.is_numeric()) and not isinstance(dtype, (nw.Decimal, nw.Unknown)):
                expression = expression.cast(dtype)
            expressions.append(expression.alias(name))
        else:
            if len(value) != len(frame):
                raise ValueError("replacement must have the same length as data")
            if isinstance(dtype, (nw.Datetime, nw.Duration)):
                series = nw.new_series(name, _temporal_counts(value, dtype), dtype=nw.Int64, backend=backend).cast(dtype)
            else:
                if isinstance(dtype, (nw.Categorical, nw.Enum, nw.String, nw.Boolean)):
                    value = [None if pd.isna(cell) else cell for cell in value]
                elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
                    # Polars preserves a NumPy object array as Object, which
                    # cannot be cast back to Boolean/List/Struct. Infer from
                    # Python values while keeping genuine floating NaNs.
                    value = [None if cell is pd.NA or cell is pd.NaT else cell for cell in value]
                series = (nw.new_series(name, value, dtype=dtype, backend=backend)
                          if _uint64_integer_values(dtype, value)
                          else nw.from_dict({name: value}, backend=backend).get_column(name))
                if (not dtype.is_numeric() or series.null_count() == len(series)) and not isinstance(dtype, (nw.Decimal, nw.Unknown)):
                    series = series.cast(dtype)
            expressions.append(series)
    return _restore_native_fields(frame.with_columns(expressions).to_native(), data, values, schema)


def native_observation(observation, reference, single=False):
    """Validate observations for a non-pandas explainer without collecting lazy data."""
    frame = as_eager_dataframe(reference)
    if isinstance(observation, (list, np.ndarray, pd.Series)):
        values = np.asarray(observation, dtype=object) if isinstance(observation, (list, pd.Series)) else np.asarray(observation)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(frame.columns):
            raise ValueError("Wrong new_observation dimension")
        observation = frame_from_values(values, reference)
    obs = as_eager_dataframe(observation)
    if nw.get_native_namespace(obs) is not nw.get_native_namespace(frame):
        raise TypeError("new_observation must use the same dataframe backend as explainer.data")
    if obs.columns != frame.columns:
        raise ValueError("new_observation columns must match explainer.data in order")
    if single and len(obs) != 1:
        raise ValueError("Wrong new_observation dimension")
    return observation


class NativePredictFunction:
    """Picklable prediction adapter: normalize series outputs, preserve input backend."""

    def __init__(self, function):
        self.function = function

    def __call__(self, model, data):
        if _legacy_polars_frame(data):
            data = data.clone()
        prediction = self.function(model, data)
        if isinstance(prediction, np.ndarray):
            return prediction
        return as_numpy_vector(prediction, "prediction")



class NativeResidualFunction:
    """Preserve the three-argument callback contract and normalize native vectors."""

    def __init__(self, function):
        self.function = function

    def __call__(self, model, data, y):
        if _legacy_polars_frame(data):
            data = data.clone()
        result = self.function(model, data, y)
        return result if isinstance(result, np.ndarray) else as_numpy_vector(result, "residuals")


def select_columns(data, columns):
    names = list(columns)
    if isinstance(data, pd.DataFrame):
        return data.loc[:, names]
    return as_eager_dataframe(data).select(names).to_native()


def numeric_columns(data):
    if isinstance(data, pd.DataFrame):
        return list(data.select_dtypes(include=np.number).columns)
    return [name for name, dtype in as_eager_dataframe(data).schema.items() if dtype.is_numeric()]


def categorical_levels(series):
    """Declared Enum levels also work with old Narwhals on Polars 2."""
    if isinstance(series.dtype, nw.Enum):
        return nw.new_series(series.name, list(series.dtype.categories), dtype=nw.String,
                             backend=nw.get_native_namespace(series))
    try:
        return series.cat.get_categories()
    except (AttributeError, NotImplementedError):
        # Polars 2 Categorical has no fixed category list; older Narwhals still
        # calls its removed get_categories method. Enum remains fixed above.
        return series.drop_nulls().unique()


def frame_from_values(values, reference, columns=None):
    """Explicit NumPy-library boundary: restore native feature batches for callbacks."""
    names = list(column_names(reference)) if columns is None else list(columns)
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError("Feature matrix must have the same ordered columns as the reference")
    if isinstance(reference, pd.DataFrame):
        selected_reference = reference if columns is None else reference.loc[:, names]
        if selected_reference.shape[1] != len(names):
            raise ValueError('Explicit columns select duplicate labels ambiguously; omit columns to retain every physical feature')
        # A mixed NumPy matrix has object dtype for every column. Reconstruct
        # each column separately so numeric features remain numeric, and restore
        # categorical metadata expected by dataframe-aware model pipelines.
        # Construct by physical position so duplicate labels do not collapse
        # dictionary keys, and retain original Index/MultiIndex metadata.
        result = pd.DataFrame({i: array[:, i].tolist() for i in range(len(names))},
                              index=pd.RangeIndex(len(array)))
        result.columns = selected_reference.columns.copy()
        for i in range(len(names)):
            dtype = selected_reference.iloc[:, i].dtype
            if isinstance(dtype, pd.CategoricalDtype):
                result.isetitem(i, pd.Categorical(result.iloc[:, i], dtype=dtype))
        return result
    frame = as_eager_dataframe(reference)
    backend = nw.get_native_namespace(frame)
    schema = {name: frame.get_column(name).dtype for name in names}
    columns_data = {}
    casts = []
    for i, name in enumerate(names):
        dtype = schema[name]
        values = [None if value is pd.NA or value is pd.NaT else value for value in array[:, i].tolist()]
        if isinstance(dtype, (nw.Datetime, nw.Duration)):
            columns_data[name] = nw.new_series(name, _temporal_counts(values, dtype), dtype=nw.Int64, backend=backend).cast(dtype).to_native()
        else:
            if isinstance(dtype, (nw.Categorical, nw.Enum, nw.String, nw.Boolean)):
                values = [None if pd.isna(value) else value for value in values]
            columns_data[name] = (nw.new_series(name, values, dtype=dtype, backend=backend).to_native()
                                  if _uint64_integer_values(dtype, values) else values)
            if (not dtype.is_numeric() or all(value is None for value in values)) and not isinstance(dtype, (nw.Decimal, nw.Unknown)):
                casts.append(nw.col(name).cast(dtype))
    native = nw.from_dict(columns_data, backend=backend)
    result = native.with_columns(casts).to_native() if casts else native.to_native()
    return _restore_native_fields(result, reference, names, schema)
