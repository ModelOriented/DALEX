"""Native CP plot tables; scalar captions retain the pandas display contract."""
import numpy as np
import narwhals.stable.v2 as nw
from narwhals.dependencies import get_pandas

from ..._plotting import concat_frames
from ._table import ProfileTables, ObservationTable, _legacy_column
from .plot import tooltip_text


def _index_positions(index, requested):
    if hasattr(index, 'get_indexer'):
        positions = index.get_indexer(requested)
        if np.any(positions < 0):
            raise KeyError('Profile observation indexes do not match new_observation')
        return positions
    lookup = {value: position for position, value in enumerate(index)}
    try:
        return np.asarray([lookup[value] for value in requested], dtype=np.int64)
    except KeyError:
        raise KeyError('Profile observation indexes do not match new_observation') from None


def profile_frame(objects):
    """Align original predictions before combining different models/variables."""
    frames, originals, converted = [], [], set()
    for obj in objects:
        current = []
        stored = obj.__dict__.get('result')
        cached = obj.__dict__.get('_compatibility_result')
        observation = obj._native_observation_frame()
        observed = nw.from_native(observation, eager_only=True)
        metadata = obj.__dict__.get('new_observation')
        cached_observation = obj.__dict__.get('_compatibility_new_observation')
        index = (cached_observation.index if cached_observation is not None else
                 metadata.index if isinstance(metadata, ObservationTable) else
                 observation.index if hasattr(observation, 'index') else observed['_ids_'].to_list())
        entries = obj._native_profile_frames()
        if isinstance(stored, ProfileTables) and cached is None:
            indexes = stored.indexes
            converted.update(stored.variables)
        else:
            indexes = [data.index if hasattr(data, 'index') else nw.from_native(data)['_ids_'].to_list()
                       for _, data in entries]
        for (_, data), requested in zip(entries, indexes):
            frame = nw.from_native(data, eager_only=True)
            originals.extend(_point_values(frame, '_original_'))
            positions = _index_positions(index, requested)
            baseline = observed['_yhat_'][positions].alias('_original_yhat_')
            if nw.get_native_namespace(observed).__name__ != nw.get_native_namespace(frame).__name__:
                baseline = nw.new_series('_original_yhat_', baseline.to_list(), dtype=baseline.dtype,
                                         backend=nw.get_native_namespace(frame))
            current.append(frame.drop('_original_').with_columns(baseline))
        combined = concat_frames(current, how='diagonal')
        if isinstance(stored, ProfileTables) and cached is None and nw.get_native_namespace(combined).__name__ == 'pandas':
            # Existing pandas inputs retain their publication dtypes before
            # combining models. This one-dimensional display normalization is
            # separate from dataframe manipulation and native feature batches.
            for name in stored.variables:
                column = _legacy_column(combined.to_native()[name])
                combined = combined.with_columns(nw.new_series(name, column.array, backend=get_pandas()))
        frames.append(combined)
    absent = {name: np.concatenate([np.full(len(frame), name not in frame.columns, dtype=bool) for frame in frames])
              for name in dict.fromkeys(name for frame in frames for name in frame.columns)
              if any(name not in frame.columns for frame in frames)}
    result = concat_frames(frames, how='diagonal')
    return result, originals, converted, absent


def _integral(frame, name):
    series = frame[name]
    if series.dtype.is_integer():
        return True
    if not series.dtype.is_float():
        return False
    if nw.get_native_namespace(frame).__name__ == 'pandas' and isinstance(
        frame.to_native()[name].dtype, get_pandas().api.extensions.ExtensionDtype
    ):
        return False
    values = np.asarray(series.to_numpy(), dtype=float)
    finite = values[~np.isnan(values)]
    return (not len(finite) or bool(np.all(np.isfinite(finite)) and np.all(finite == np.floor(finite))
                                  and np.all(finite >= -(2**63)) and np.all(finite < 2**63)))


def _display_column(frame, name, converted, integral):
    """Format scalars only; never construct a pandas DataFrame or Series."""
    pd = get_pandas()
    series = frame[name]
    namespace = nw.get_native_namespace(frame).__name__
    dtype = series.dtype
    if namespace != 'pandas' and (isinstance(dtype, (nw.Datetime, nw.Duration)) or
                                 dtype == nw.Date and namespace == 'polars'):
        counts = series.cast(nw.Int64).to_list()
        if isinstance(dtype, nw.Duration):
            return [pd.NaT if value is None else pd.Timedelta(value, unit=dtype.time_unit) for value in counts]
        return [pd.NaT if value is None else pd.Timestamp(
            value, unit='D' if dtype == nw.Date else dtype.time_unit,
            tz=None if dtype == nw.Date else dtype.time_zone) for value in counts]
    values = series.to_list()
    if converted and integral:
        return [pd.NA if value is None or isinstance(value, (float, np.floating)) and np.isnan(value)
                else int(value) for value in values]
    if converted and (dtype == nw.String or dtype == nw.Boolean):
        return [pd.NA if value is None else value for value in values]
    if namespace != 'pandas':
        if isinstance(dtype, (nw.Categorical, nw.Enum)) or dtype.is_float():
            return [np.nan if value is None else value for value in values]
        if dtype.is_integer() and series.null_count():
            return [np.nan if value is None else float(value) for value in values]
    return values


def _point_values(frame, name):
    dtype = frame.schema[name]
    return (_display_column(frame, name, False, False) if isinstance(dtype, (nw.Datetime, nw.Duration))
            else frame[name].to_list())


def plot_frame(frame, originals, converted, absent, variables, categorical):
    """Build typed axes, exact point membership and scalar hover captions."""
    selected = frame['_vname_'].is_in(list(variables)).to_numpy()
    originals = [value for value, keep in zip(originals, selected) if keep]
    integral = {name: _integral(frame, name) for name in converted if name in frame.columns}
    frame = frame.filter(nw.col('_vname_').is_in(list(variables)))
    display = {name: _display_column(frame, name, name in converted, integral.get(name, False))
               for name in frame.columns}
    if nw.get_native_namespace(frame).__name__ != 'pandas':
        for name, missing in absent.items():
            dtype = frame.schema[name]
            if name not in converted and (dtype in (nw.String, nw.Boolean, nw.Object, nw.Unknown) or
                                          isinstance(dtype, nw.Decimal) or dtype == nw.Date):
                display[name] = [np.nan if missing else value for value, missing in zip(display[name], missing[selected])]
    captions = [tooltip_text(dict(zip(display, values))) for values in zip(*display.values())]
    pieces = []
    ordinal = '__dalex_plot_position__'
    while ordinal in frame.columns:
        ordinal += '_'
    for variable in dict.fromkeys(frame['_vname_'].to_list()):
        positions = np.flatnonzero(frame['_vname_'].to_numpy() == variable)
        part = frame[positions, :]
        namespace = nw.get_native_namespace(part)
        if categorical:
            axis = [str(display[variable][position]) for position in positions]
            points = [isinstance(originals[position], str) and originals[position] == value
                      for position, value in zip(positions, axis)]
            x = nw.new_series('_x_', axis, backend=namespace)
        else:
            x = part[variable].alias('_x_')
            points = []
            for position, value in zip(positions, _point_values(part, variable)):
                original = originals[position]
                points.append(False if original is None or value is None or original is get_pandas().NA or value is get_pandas().NA
                              else bool(original == value))
        pieces.append(part.with_columns(x,
            nw.new_series('_text_', [captions[position] for position in positions], backend=namespace),
            nw.new_series(ordinal, positions, backend=namespace),
            nw.new_series('_is_observation_', points, dtype=nw.Boolean, backend=namespace)))
    return concat_frames(pieces).sort(ordinal).drop(ordinal)


def categorical_color(frame, name):
    """Plotly treats legacy object colors as discrete, including numeric ids."""
    dtype = frame.schema[name]
    if dtype == nw.String or isinstance(dtype, (nw.Categorical, nw.Enum)):
        return frame
    values = _display_column(frame, name, False, False)
    pd = get_pandas()
    labels = [None if value is None or value is pd.NA or value is pd.NaT or
              isinstance(value, (float, np.floating)) and np.isnan(value) else str(value) for value in values]
    return frame.with_columns(nw.new_series(name, labels, backend=nw.get_native_namespace(frame)))
