"""Narwhals profile reductions. Legacy display formatting lives at the boundary.

The input is a typed, long profile table. No pandas conversion is necessary for
means, Gaussian weighting, within-observation differences or accumulated effects.
The caller retains row-index and label-format metadata for its legacy public API.
"""
import numpy as np
import narwhals.stable.v2 as nw
from narwhals.dependencies import get_pandas
from ..._explanation import IndexedTable
from ..._dataframe import column_values
from ...predict_explanations._ceteris_paribus._table import concat_compatible, export_frame


def _non_nan(expression):
    # Expr.fill_nan was introduced after Narwhals 2.0.
    return nw.when(expression.is_nan()).then(None).otherwise(expression)


def _missing_kind(frame, name):
    column = nw.col(name)
    kind = nw.when(column.is_null()).then(nw.lit('null')).otherwise(nw.lit('value'))
    if frame.schema[name].is_float():
        nan = nw.when(column.is_nan()).then(nw.lit('nan')).otherwise(kind)
        # Native float buffers distinguish null from NaN. Ordinary pandas
        # float storage has one missing state and historically uses NaN.
        kind = nan if nw.get_native_namespace(frame).__name__ == 'pandas' else nw.when(column.is_null()).then(nw.lit('null')).otherwise(nan)
    return kind


def _large_integer(frame, name):
    series = frame[name]
    return series.dtype.is_integer() and any(value is not None and abs(value) > 2**53
                                             for value in (series.min(), series.max()))


def _integer_weights(frame, span):
    """Subtract integer grid keys exactly before entering floating statistics."""
    from decimal import Decimal
    grid, original = frame['_x_'].to_list(), frame['_original_'].to_list()
    valid = [int(value) if isinstance(value, (int, np.integer)) else value
             for value in grid if not _missing(value)]
    width = (max(valid) - min(valid)) if valid else 1.
    if not width:
        width = 1.
    distances = []
    for value, split in zip(original, grid):
        if _missing(value) or _missing(split):
            distances.append(np.nan)
        elif isinstance(value, (int, np.integer)) and isinstance(split, (int, np.integer)):
            # Older pandas returns NumPy integer scalars here. Their unsigned
            # subtraction wraps, and a signed max-min range can overflow too.
            distances.append(float(int(value) - int(split)))
        elif isinstance(value, (int, np.integer)) and isinstance(split, (float, np.floating)):
            distances.append(float(Decimal(int(value)) - Decimal.from_float(float(split))))
        elif isinstance(split, (int, np.integer)) and isinstance(value, (float, np.floating)):
            distances.append(float(Decimal.from_float(float(value)) - Decimal(int(split))))
        else:
            distances.append(float(value - split))
    distances = np.asarray(distances)
    values = np.exp(-1 * ((distances / float(width)) / span)**2 / 2) / np.sqrt(2*np.pi) / span
    return nw.new_series('__weight__', values, backend=nw.get_native_namespace(frame))


def _ordered_aggregate(frame, keys, expressions):
    # Native group-by ordering is not a portable contract. Record first-observed
    # order explicitly, as pandas groupby(sort=False) does for result rows.
    ordinal = '__dalex_profile_position__'
    while ordinal in frame.columns:
        ordinal += '_'
    return (frame.with_row_index(ordinal).group_by(keys, drop_null_keys=False)
            .agg(*expressions, nw.col(ordinal).min()).sort(ordinal).drop(ordinal))


def _window(frame, keys, operation):
    expression = getattr(nw.col('_yhat_'), operation)().over(keys)
    namespace = nw.get_native_namespace(frame).__name__
    missing_groups = operation == 'cum_sum' and any(frame[key].null_count() for key in keys)
    if namespace != 'pyarrow' and not (namespace == 'pandas' and missing_groups):
        return frame.with_columns(expression)
    # Arrow supports elementary aggregate windows, but not diff/cumsum windows
    # (including at the declared Narwhals minimum). Use numeric vectors for this
    # operation only; the table and reductions retain the input backend. Older
    # Narwhals pandas windows also drop missing groups; the same fallback keeps
    # the legacy dropna=False cumulative-sum contract for those memberships.
    columns = frame.select(keys).to_dict(as_series=False)
    missing = object()
    nan_key = missing if namespace == 'pandas' else object()
    positions = {}
    for i, cells in enumerate(zip(*(columns[key] for key in keys))):
        pandas = get_pandas()
        key = tuple(missing if value is None or pandas is not None and (value is pandas.NA or value is pandas.NaT)
                    else nan_key if isinstance(value, (float, np.floating)) and np.isnan(value) else value for value in cells)
        positions.setdefault(key, []).append(i)
    values = frame.get_column('_yhat_').to_numpy()
    output = np.empty_like(values)
    for key, rows in positions.items():
        chunk = values[rows]
        if operation == 'diff':
            output[rows] = 0. if missing in key or nan_key in key else np.r_[0., np.diff(chunk)]
        else:
            absent = np.isnan(chunk)
            accumulated = np.cumsum(np.where(absent, 0., chunk))
            accumulated[absent] = np.nan
            output[rows] = accumulated
    return frame.with_columns(nw.new_series('_yhat_', output, backend=nw.get_native_namespace(frame)))


def _numeric_weights(frame, kind, span):
    """Detached float vectors implement the Gaussian numerical boundary.

    Typed keys remain in the dataframe. Large integer distances still use exact
    subtraction; no integer feature or group key is narrowed to enter statistics.
    """
    large = _large_integer(frame, '_x_') or _large_integer(frame, '_original_')
    if large:
        # Arrow/Polars exports can be read-only views. Missing-bin normalization
        # below edits only this owned numerical work buffer.
        weights = np.array(_integer_weights(frame, span).to_numpy(), dtype=float, copy=True)
    else:
        grid = np.asarray(column_values(frame.to_native(), '_x_', preserve_nulls=True), dtype=float)
        original = np.asarray(column_values(frame.to_native(), '_original_', preserve_nulls=True), dtype=float)
        known = grid[~np.isnan(grid)] if np.isnan(grid).any() else grid
        width = known.max() - known.min() if len(known) else 1.
        if width == 0:
            width = 1.
        with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
            weights = np.exp(-1 * (((original - grid) / width) / span)**2 / 2) / np.sqrt(2 * np.pi) / span
    if kind == 'conditional' and '_x_kind_' in frame.columns:
        original_kind = np.asarray(frame['_original_kind_'].to_list(), dtype=object)
        grid_kind = np.asarray(frame['_x_kind_'].to_list(), dtype=object)
        original_missing, grid_missing = set(original_kind) - {'value'}, set(grid_kind) - {'value'}
        if len(original_missing) == len(grid_missing) == 1:
            original_kind = np.where(original_kind != 'value', next(iter(grid_missing)), original_kind)
        weights[np.isnan(weights)] = 0.
        weights = np.where(grid_kind != 'value', (grid_kind == original_kind).astype(float), weights)
    elif kind == 'conditional' and not large and np.isnan(grid).any():
        weights[np.isnan(weights)] = 0.
        weights[np.isnan(grid) & np.isnan(original)] = 1.
    return weights


def _profile_groups(frame):
    """A CP partition normally already has one variable and model label.

    Avoid copying its entire table through a group-by for that common case.
    Edited legacy tables can contain several labels; they retain the general
    split path, including its missing-key behavior and original row order.
    """
    names = ['_vname_', '_label_']
    columns = [frame[name] for name in names]
    values = [column.unique().to_list() for column in columns]
    if all(len(value) == 1 for value in values):
        if any(column.null_count() == len(frame) for column in columns):
            return
        yield tuple(value[0] for value in values), frame
        return
    yield from frame.group_by(names, drop_null_keys=True)


def reduce_profiles(data, kind, groups=(), *, span=.25, partition_keys=()):
    """Reduce one variable/model partition; retain typed group memberships.

    Returns a native result with ``_x_``, group columns and ``_yhat_``. Missing
    predictions follow legacy skip-NaN reductions, while missing grid keys remain
    keys. Native float storage retains separate null and NaN memberships.
    """
    if kind not in ('partial', 'conditional', 'accumulated'):
        raise ValueError('Unknown profile type')
    frame = nw.from_native(data, eager_only=True)
    if not len(frame):
        raise ValueError('Cannot aggregate an empty profile partition')
    groups = list(groups)
    keys = [*partition_keys, '_x_', *(['_x_kind_'] if '_x_kind_' in frame.columns else []), *groups]
    numeric = frame.schema['_x_'].is_numeric() or frame.schema['_x_'] == nw.Boolean
    if kind == 'partial':
        if not frame.schema['_yhat_'].is_float() or frame['_yhat_'].is_nan().any():
            frame = frame.with_columns(_non_nan(nw.col('_yhat_')).cast(frame.schema['_yhat_']).alias('_yhat_'))
        return _ordered_aggregate(frame, keys, [nw.col('_yhat_').mean()]).to_native()
    if numeric:
        if kind == 'accumulated' and (frame['_x_'].null_count() or frame.schema['_x_'].is_float() and frame['_x_'].is_nan().any()):
            raise ValueError('Accumulated effects cannot order a missing value in a numeric grid')
        weights = _numeric_weights(frame, kind, span)
    else:
        weight_frame = frame.select((nw.col('_original_') == nw.col('_x_')).cast(nw.Float64).alias('__weight__'))
        weights = np.asarray(column_values(weight_frame.to_native(), '__weight__', preserve_nulls=True), dtype=float)
    weights = np.asarray(weights, dtype=float).copy()
    weights[np.isnan(weights)] = 0.
    if kind == 'accumulated':
        frame = _window(frame, ['_ids_'], 'diff')
    predictions = np.asarray(column_values(frame.to_native(), '_yhat_', preserve_nulls=True), dtype=float)
    with np.errstate(invalid='ignore'):
        products = predictions * weights
    # NaN products contributed zero to the historical skip-missing sums. Keeping
    # this numerical normalization outside expressions avoids repeated full-table
    # clones while leaving every grouping key and its dtype in native storage.
    products[np.isnan(products)] = 0.
    backend = nw.get_native_namespace(frame)
    frame = frame.with_columns(nw.new_series('__weight__', weights, backend=backend),
                                nw.new_series('__product__', products, backend=backend))
    result = _ordered_aggregate(frame, keys, [nw.col('__product__').sum().alias('__numerator__'),
                                             nw.col('__weight__').sum().alias('__denominator__')])
    divisor = nw.when(nw.col('__denominator__') != 0).then(nw.col('__denominator__')).otherwise(None)
    result = result.with_columns((nw.col('__numerator__') / divisor).fill_null(0.).alias('_yhat_'))
    result = result.drop('__numerator__', '__denominator__')
    if kind == 'accumulated':
        if groups:
            result = _window(result, groups, 'cum_sum')
        else:
            result = result.with_columns(nw.col('_yhat_').cum_sum())
    return result.to_native()


def _missing(value):
    pandas = get_pandas()
    return value is None or (pandas is not None and (value is pandas.NA or value is pandas.NaT)) or (
        isinstance(value, (float, np.floating)) and np.isnan(value))


def _numeric_column(frame, name):
    dtype = frame.schema[name]
    if isinstance(dtype, nw.Decimal):
        return False  # Legacy Decimal object columns are categorical.
    if dtype.is_numeric() or isinstance(dtype, nw.Boolean):
        return True
    if isinstance(dtype, (nw.Object, nw.Unknown)):
        values = [value for value in frame[name].to_list() if not _missing(value)]
        return bool(values) and np.asarray(values).dtype.kind in 'biufc'
    return False


def _time_label(count, dtype):
    if count is None:
        return 'NaT'
    count = int(count) * {'s': 10**9, 'ms': 10**6, 'us': 1000, 'ns': 1}[dtype.time_unit]
    pandas = get_pandas()
    if pandas is not None:
        if isinstance(dtype, nw.Duration):
            return str(pandas.Timedelta(count, unit='ns'))
        value = pandas.Timestamp(count, unit='ns', tz='UTC') if dtype.time_zone else pandas.Timestamp(count, unit='ns')
        return str(value.tz_convert(dtype.time_zone) if dtype.time_zone else value)
    # Native-only installations still need precise scalar display labels.
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    if isinstance(dtype, nw.Duration):
        days, rest = divmod(count, 86400 * 10**9)
        seconds, fraction = divmod(rest, 10**9)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        text = f'{days} days ' + ('+' if days < 0 and rest else '') + f'{hours:02}:{minutes:02}:{seconds:02}'
        return text + (f'.{fraction:09}' if fraction else '')
    seconds, fraction = divmod(count, 10**9)
    value = datetime.fromtimestamp(seconds, timezone.utc)
    if dtype.time_zone:
        value = value.astimezone(ZoneInfo(dtype.time_zone))
    text = value.strftime('%Y-%m-%d %H:%M:%S')
    if fraction:
        text += f'.{fraction:09}' if fraction % 1000 else f'.{fraction // 1000:06}'
    offset = value.strftime('%z') if dtype.time_zone else ''
    return text + (offset[:3] + ':' + offset[3:] if offset else '')


def _keys_labels(frame, variable, integer_labels=False):
    column = frame.get_column(variable)
    dtype = column.dtype
    if isinstance(dtype, (nw.Datetime, nw.Duration)):
        absent = column.is_null().to_numpy()
        counts = column.cast(nw.Int64).to_list()
        values = [None if missing else int(value) for value, missing in zip(counts, absent)]
        return values, [_time_label(value, dtype) for value in values], 'NaT'
    values = column.to_list()
    missing_label = ('nan' if isinstance(dtype, (nw.Categorical, nw.Enum)) else
                     '<NA>' if isinstance(dtype, (nw.String, nw.Boolean)) or integer_labels else
                     'nan' if dtype.is_numeric() else 'None')
    labels = [missing_label if _missing(value) else str(int(value)) if integer_labels else str(value) for value in values]
    return values, labels, missing_label


def _integer_display(frames, variable):
    # CP's legacy convert_dtypes turns wholly integral floating columns into
    # nullable integers. Inspect all partitions, including unmodified cells.
    for frame in frames:
        if variable not in frame.columns:
            continue
        values = frame[variable].to_list()
        for value in values:
            if not _missing(value) and (not isinstance(value, (int, float, np.integer, np.floating))
                                        or isinstance(value, (bool, np.bool_)) or value != np.floor(value)):
                return False
    return True


class AggregatedTable(IndexedTable):
    """Format legacy group labels only at the public pandas boundary."""

    def __init__(self, data, index, groups, group_dtypes):
        super().__init__(data, index)
        self.groups = list(groups)
        self.group_dtypes = group_dtypes

    def _dalex_pandas_view(self):
        result = super()._dalex_pandas_view()
        frame = nw.from_native(self.data, eager_only=True)
        for group in self.groups:
            series = frame[group]
            if (nw.get_native_namespace(frame).__name__ != 'pandas' and series.dtype.is_float()
                    and series.null_count() and series.is_nan().any()):
                values = np.empty(len(series), dtype=object)
                values[:] = series.to_list()
                result[group] = values
                continue
            dtype = self.group_dtypes.get(group)
            if dtype == 'object':
                result[group] = result[group].astype(object).where(result[group].notna(), np.nan)
            elif dtype is not None:
                result[group] = result[group].astype(dtype)
        if self.groups:
            result['_groups_'] = result.loc[:, self.groups].apply(lambda row: '_'.join(map(str, row)), axis=1)
            result = result.drop(columns=self.groups)
            result['_label_'] = result.loc[:, ['_label_', '_groups_']].apply(lambda row: '_'.join(row), axis=1)
        return result


def aggregate_long_profiles(data, mean_prediction, kind, groups, center, span):
    """Adapt an already prepared long table to the same native reductions.

    This supports legacy private helpers without maintaining a second pandas
    groupby/apply implementation. Index and group-label formatting are deferred
    to the existing compatibility table.
    """
    frame = nw.from_native(data, eager_only=True)
    groups = list(groups)
    group_dtypes = {}
    for group in groups:
        dtype = frame.schema[group]
        if isinstance(dtype, (nw.Categorical, nw.Enum)):
            group_dtypes[group] = 'object'
            frame = frame.with_columns(nw.col(group).cast(nw.String))
        elif nw.get_native_namespace(frame).__name__ == 'pandas':
            group_dtypes[group] = frame.to_native()[group].dtype
    columns = ['_vname_', '_label_', '_x_', *groups, '_yhat_', '_ids_']
    if kind == 'partial':
        result = nw.from_native(reduce_profiles(frame.to_native(), kind, groups, span=span,
            partition_keys=['_vname_', '_label_'])).with_columns(nw.lit(0).alias('_ids_'))
        index = None
    else:
        partitions = []
        for key, split in _profile_groups(frame):
            key = tuple(value.as_py() if hasattr(value, 'as_py') else value for value in key)
            partitions.append((key, split))
        partitions.sort(key=lambda item: tuple((isinstance(value, str), value) for value in item[0]))
        results, index = [], []
        for (variable, label), split in partitions:
            reduced = nw.from_native(reduce_profiles(split.to_native(), kind, groups, span=span))
            reduced = reduced.with_columns(nw.lit(variable).alias('_vname_'),
                nw.lit(label).alias('_label_'), nw.lit(0).alias('_ids_'))
            results.append(reduced.select(columns))
            index.extend(range(len(reduced)))
        result = nw.concat(results)
    if kind == 'accumulated' and center:
        average = np.nanmean(result['_yhat_'].to_numpy())
        result = result.with_columns((nw.col('_yhat_') - average + mean_prediction).alias('_yhat_'))
    return AggregatedTable(result.select(columns).to_native(), index, groups, group_dtypes)


def aggregate_ceteris(profiles, variables, variable_type, kind, groups, span, center):
    """Compute aggregated explanations directly from typed CP partitions."""
    entries = [(variable, nw.from_native(data)) for cp in profiles for variable, data in cp._native_profile_frames()]
    all_variables = list(dict.fromkeys(variable for variable, _ in entries))
    if variables is not None:
        selected = set(variables).intersection(all_variables)
        if not selected:
            raise ValueError('variables do not overlap with profiles')
    else:
        selected = set(all_variables)
    numerical = {variable for variable, frame in entries if _numeric_column(frame, variable)}
    selected &= numerical if variable_type == 'numerical' else selected if variables is not None else set(all_variables) - numerical
    if not selected:
        raise ValueError('There are no numerical variables' if variable_type == 'numerical' else 'There are no non-numerical variables')
    missing_variables = {variable for cp in profiles if getattr(cp, '_native_features', False)
                         for variable, values in cp.variable_splits.items()
                         if variable in selected and any(_missing(value) for value in values)}
    predictions = [nw.from_native(cp._native_observation_frame())['_yhat_'].to_numpy() for cp in profiles]
    mean_prediction = np.nanmean(np.concatenate(predictions))
    namespaces = [nw.get_native_namespace(frame).__name__ for _, frame in entries]
    target = next((namespace for namespace in namespaces if namespace != 'pandas'), namespaces[0])
    groups = list(groups)
    group_dtypes = {}
    for _, frame in entries:
        for group in groups:
            if group not in frame.columns:
                raise KeyError(group)
            dtype = frame.schema[group]
            if isinstance(dtype, (nw.Categorical, nw.Enum)):
                group_dtypes[group] = 'object'
            elif nw.get_native_namespace(frame).__name__ == 'pandas':
                group_dtypes.setdefault(group, frame.to_native()[group].dtype)
            elif isinstance(dtype, (nw.String, nw.Boolean, nw.Object, nw.Unknown)):
                group_dtypes.setdefault(group, 'object')
    # Category labels are global to a variable, even across model labels/CPs.
    category_maps = {}
    if variable_type == 'categorical':
        frames = [frame for _, frame in entries]
        for variable in selected:
            integer_labels = _integer_display(frames, variable)
            levels, missing_labels, nonmissing, missing_kinds = {}, [], set(), set()
            for name, frame in entries:
                if name != variable:
                    continue
                values, labels, missing_label = _keys_labels(frame, variable, integer_labels)
                for value, label in zip(values, labels):
                    if _missing(value):
                        missing_labels.append(missing_label)
                    else:
                        levels[value] = label
                        nonmissing.add(label)
                for value in frame['_original_'].to_list():
                    if _missing(value):
                        missing_kinds.add('nan' if isinstance(value, (float, np.floating)) and np.isnan(value) else 'null')
            missing_label = missing_labels[0] if missing_labels else '<missing>'
            while missing_label in nonmissing:
                missing_label += ' (missing)'
            distinct = {}
            if variable in missing_variables and len(missing_kinds) > 1:
                for missing_kind in missing_kinds:
                    label = '<' + missing_kind + '>'
                    while label in nonmissing:
                        label += ' (missing)'
                    distinct[missing_kind] = label
            category_maps[variable] = (levels, missing_label, integer_labels, distinct)
    partitions = {}
    for variable, frame in entries:
        if variable not in selected:
            continue
        if variable_type == 'numerical':
            dtype = frame.schema[variable]
            if dtype.is_numeric() or dtype == nw.Boolean:
                frame = frame.with_columns(frame.get_column(variable).alias('_x_'))
            else:
                from ..._dataframe import column_values
                values = np.asarray(column_values(frame.to_native(), variable, preserve_nulls=True), dtype=float)
                frame = frame.with_columns(nw.new_series('_x_', values, backend=nw.get_native_namespace(frame)))
            if variable in missing_variables:
                frame = frame.with_columns(_missing_kind(frame, '_x_').alias('_x_kind_'),
                                           _missing_kind(frame, '_original_').alias('_original_kind_'))
        else:
            levels, missing_label, integer_labels, distinct = category_maps[variable]
            values, labels, _ = _keys_labels(frame, variable, integer_labels)
            originals, original_labels, _ = _keys_labels(frame, '_original_', False)
            def missing_display(value):
                key = 'nan' if isinstance(value, (float, np.floating)) and np.isnan(value) else 'null'
                return distinct.get(key, missing_label)
            labels = [missing_display(value) if _missing(value) else label for value, label in zip(values, labels)]
            original_labels = [missing_display(value) if _missing(value) else levels.get(value, label)
                               for value, label in zip(originals, original_labels)]
            namespace = nw.get_native_namespace(frame)
            frame = frame.with_columns(nw.new_series('_x_', labels, backend=namespace),
                                       nw.new_series('_original_', original_labels, backend=namespace))
        for group in groups:
            dtype = frame.schema[group]
            if isinstance(dtype, (nw.Categorical, nw.Enum)):
                frame = frame.with_columns(nw.col(group).cast(nw.String))
            elif isinstance(dtype, (nw.Float32, nw.Float64)) and nw.get_native_namespace(frame).__name__ == 'pandas':
                frame = frame.with_columns(_non_nan(nw.col(group)).cast(dtype).alias(group))
        keep = list(dict.fromkeys(['_vname_', '_label_', '_x_', '_yhat_', '_ids_', '_original_', *groups]))
        keep += [name for name in ('_x_kind_', '_original_kind_') if name in frame.columns]
        frame = frame.select(keep)
        for key, split in _profile_groups(frame):
            key = tuple(value.as_py() if hasattr(value, 'as_py') else value for value in key)
            native = split.to_native()
            partitions.setdefault(key, []).append(
                native if nw.get_native_namespace(split).__name__ == target else export_frame(native, target))
    ordered = list(partitions)
    if kind != 'partial':
        ordered.sort(key=lambda key: tuple((isinstance(value, str), value) for value in key))
    results, indexes = [], []
    for variable, label in ordered:
        parts = partitions[(variable, label)]
        # Reductions below create new result buffers and never edit their input.
        # Public exports still clone independently at their documented boundary.
        joined = parts[0] if len(parts) == 1 else concat_compatible(parts)
        reduced = nw.from_native(reduce_profiles(joined, kind, groups, span=span))
        reduced = reduced.with_columns(nw.lit(variable).alias('_vname_'), nw.lit(label).alias('_label_'),
                                        nw.lit(0).cast(nw.Int64).alias('_ids_'))
        columns = ['_vname_', '_label_', '_x_', *groups, '_yhat_', '_ids_']
        if missing_variables and variable_type == 'numerical':
            if '_x_kind_' not in reduced.columns:
                reduced = reduced.with_columns(nw.lit('value').alias('_x_kind_'))
            columns.append('_x_kind_')
        results.append(reduced.select(columns).to_native())
        indexes.extend(range(len(reduced)))
    if kind == 'accumulated' and center:
        predictions = np.concatenate([nw.from_native(data)['_yhat_'].to_numpy() for data in results])
        average = np.nanmean(predictions)
        results = [nw.from_native(data).with_columns((nw.col('_yhat_') - average + mean_prediction).alias('_yhat_')).to_native()
                   for data in results]
    index = None if kind == 'partial' else np.asarray(indexes)
    from ._table import AggregatedPartitions
    return AggregatedPartitions(results, index, groups, group_dtypes, variable_type == 'numerical'), mean_prediction
