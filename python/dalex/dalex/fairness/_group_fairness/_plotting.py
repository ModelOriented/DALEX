"""Native fairness plot tables and explicit numerical display boundaries."""
import numpy as np
import narwhals.stable.v2 as nw
from narwhals.dependencies import get_pandas
from pandas import DataFrame as PandasFrame, Series as PandasSeries

from ..._plotting import concat_frames
from ..._explainer import helper
from .._basics.exceptions import ParameterCheckError


def table_frame(obj, name='result'):
    original = obj.__dict__.get(name)
    value = obj.__dict__.get('_compatibility_' + name, original)
    pd = get_pandas()
    if isinstance(value, (PandasFrame, PandasSeries)):
        backend = getattr(obj, '_result_backend', 'pandas')
        if isinstance(value, PandasSeries):
            data = {'metric': value.index.to_numpy(), 'score': scores(value.to_numpy())}
        else:
            data = {'subgroup': value.index.to_numpy(),
                    **{column: scores(value[column].to_numpy()) for column in value.columns}}
        return nw.from_dict(data, backend=backend)
    return nw.from_native(value.data if hasattr(value, '_dalex_native_view') else value, eager_only=True)


def scores(values):
    pd = get_pandas()
    return np.asarray([np.nan if value is None or value is pd.NA else float(value) for value in values], dtype=float)


def long_table(obj, name='result', *, row_major=True):
    frame = table_frame(obj, name)
    return long_frame(frame, row_major=row_major).with_columns(nw.lit(obj.label).alias('label'))


def long_frame(frame, *, row_major=True):
    columns = [column for column in frame.columns if column != 'subgroup']
    result = frame.with_row_index('__row__').unpivot(
        on=columns, index=['subgroup', '__row__'], variable_name='metric', value_name='score')
    ordinal = {column: position for position, column in enumerate(columns)}
    result = result.with_columns(nw.new_series('__metric__', [ordinal[name] for name in result['metric'].to_list()],
                                               dtype=nw.Int64, backend=nw.get_native_namespace(result)))
    return (result.sort(['__row__', '__metric__'] if row_major else ['__metric__', '__row__'])
            .select('subgroup', 'metric', 'score')
            .with_columns(nw.col('score').cast(nw.Float64)))


def parity_data(obj, others, metrics, verbose):
    if metrics is None or isinstance(metrics, str) or not len(metrics):
        raise ParameterCheckError('metrics must be a nonempty sequence of metric names')
    objects = [obj, *(others or [])]
    frames = [table_frame(value, 'parity_loss').with_columns(nw.col('score').cast(nw.Float64),
                                                           nw.lit(value.label).alias('label')) for value in objects]
    result = concat_frames(frames).filter(nw.col('metric').is_in(list(metrics)))
    if not len(result):
        raise ParameterCheckError('Requested metrics do not overlap parity-loss metrics')
    invalid = result.filter(nw.col('score').is_null() | nw.col('score').is_nan())
    if len(invalid):
        helper.verbose_cat(f"Found NaNs in following models: {set(invalid['label'].to_list())}", verbose)
    return result


def finite_bounds(values, default=(0., 1.)):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if len(finite) else default


def unique(frame, name):
    return frame[name].unique(maintain_order=True).to_list()


def first(frame, name):
    if not len(frame):
        raise ParameterCheckError('A required model/subgroup/metric combination is missing')
    value = frame[name][0]
    return np.nan if value is None else value


def rename_metrics(frame, mapping):
    labels = [mapping.get(name, name) for name in frame['metric'].to_list()]
    return frame.with_columns(nw.new_series('metric', labels, backend=nw.get_native_namespace(frame)))


def score_matrix(data, labels=None):
    """Heatmap's numerical boundary retains each model/metric association."""
    labels = unique(data, 'label') if labels is None else labels
    metrics = unique(data, 'metric')
    matrix = np.full((len(labels), len(metrics)), np.nan)
    rows, columns = {value: i for i, value in enumerate(labels)}, {value: i for i, value in enumerate(metrics)}
    seen = set()
    for row in data.iter_rows(named=True):
        key = (row['label'], row['metric'])
        if key in seen:
            raise ParameterCheckError('Parity-loss metrics must be unique within each model')
        seen.add(key)
        matrix[rows[key[0]], columns[key[1]]] = np.nan if row['score'] is None else row['score']
    return matrix, labels, metrics
