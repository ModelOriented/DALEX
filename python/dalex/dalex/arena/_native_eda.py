"""Native EDA reductions; JSON receives only explicit display values."""
import numpy as np
import narwhals.stable.v2 as nw
from .._dataframe import column_values, categorical_levels


def numeric(series):
    return series.dtype.is_numeric() or series.dtype == nw.Boolean


def present(series):
    mask = ~series.is_null()
    if series.dtype.is_float():
        mask = mask & ~series.is_nan()
    return series.filter(mask)


def display_categories(series):
    if isinstance(series.dtype, (nw.Datetime, nw.Duration)):
        import pandas as pd
        dtype = series.dtype
        counts = series.cast(nw.Int64).to_list()
        if isinstance(dtype, nw.Datetime):
            values = [None if count is None else pd.Timestamp(count, unit=dtype.time_unit, tz=dtype.time_zone).isoformat()
                      for count in counts]
        else:
            values = [None if count is None else pd.Timedelta(count, unit=dtype.time_unit).isoformat() for count in counts]
        return nw.new_series(series.name, values, dtype=nw.String, backend=nw.get_native_namespace(series))
    if isinstance(series.dtype, nw.Date):
        return series.cast(nw.String)
    if isinstance(series.dtype, (nw.Categorical, nw.Enum)):
        name = series.name
        frame = series.alias('__original').to_frame()
        return frame.select(nw.when(nw.col('__original').is_null()).then(None)
                            .otherwise(nw.col('__original').cast(nw.String)).alias(name))[name]
    return series


def counts(series):
    column = display_categories(present(series)).alias('__value')
    if not len(column) and not isinstance(series.dtype, (nw.Categorical, nw.Enum)):
        return {'names': [], 'count': [], 'density': []}
    frame = column.to_frame().with_row_index('__order')
    counted = frame.group_by('__value', drop_null_keys=True).agg(nw.len().alias('count'), nw.col('__order').min())
    if isinstance(series.dtype, (nw.Categorical, nw.Enum)):
        levels = categorical_levels(series).cast(nw.String).alias('__value').to_frame().with_row_index('__category_order')
        counted = levels.join(counted, on='__value', how='left').with_columns(nw.col('count').fill_null(0).cast(nw.Int64))
        order = '__category_order'
    else:
        order = '__order'
    counted = counted.sort(['count', order], descending=[True, False])
    values = counted['count'].to_numpy()
    return {'names': counted['__value'].to_list(), 'count': values.tolist(),
            'density': (values / values.sum()).tolist() if values.sum() else [0.] * len(values)}


def distribution(data, name, bins):
    frame = nw.from_native(data, eager_only=True)
    series = frame[name]
    if not numeric(series):
        return 'DistributionCounts', counts(series)
    values = np.asarray(column_values(data, name, preserve_nulls=True), dtype=float)
    values = values[np.isfinite(values)]
    output = {}
    for count in bins:
        breaks = np.histogram_bin_edges(values, bins=count)
        frequencies = np.histogram(values, bins=breaks)[0]
        output[count] = {'breaks': breaks.tolist(), 'mids': ((breaks[1:] + breaks[:-1])/2).tolist(),
                         'density': (frequencies/frequencies.sum()).tolist() if frequencies.sum() else [0.] * len(frequencies),
                         'counts': frequencies.tolist()}
    return 'DistributionHistogram', output


def boxplots(data, number, category):
    frame = nw.from_native(data, eager_only=True)
    values = np.asarray(column_values(data, number, preserve_nulls=True), dtype=float)
    raw = column_values(data, number, preserve_nulls=True)
    grouping = display_categories(frame[category])
    levels = present(grouping).unique(maintain_order=True).to_list()
    result = []
    for level in levels:
        selected = np.asarray((grouping == level).fill_null(False).to_numpy(), dtype=bool) & np.isfinite(values)
        sample = values[selected]
        if len(sample):
            q1, median, q3 = np.quantile(sample, [.25, .5, .75])
            low = max(q1 - 1.5*(q3-q1), sample.min())
            high = min(q3 + 1.5*(q3-q1), sample.max())
            mean = sample.mean()
            outliers = raw[selected & ((values < low) | (values > high))].tolist()
        else:
            q1 = median = q3 = low = high = mean = np.nan
            outliers = []
        result.append({'q1': q1, 'q3': q3, 'mean': mean, 'median': median,
                       'lf': low, 'uf': high, 'outliers': outliers})
    return result, levels


def against(data, name, points):
    frame = nw.from_native(data, eager_only=True)
    first = frame[name]
    output = {}
    for other in frame.columns:
        if name == other:
            continue
        second = frame[other]
        if numeric(first) and numeric(second):
            ids = np.random.choice(len(frame), size=min(points, len(frame)), replace=False)
            output[other] = {'type': 'scatter', 'first': first.to_frame()[ids, :][name].to_list(),
                             'secondary': second.to_frame()[ids, :][other].to_list()}
        elif numeric(first):
            boxes, levels = boxplots(data, name, other)
            output[other] = {'type': 'boxplots', 'first': boxes, 'secondary': levels, 'numerical': 'first'}
        elif numeric(second):
            boxes, levels = boxplots(data, other, name)
            output[other] = {'type': 'boxplots', 'first': levels, 'secondary': boxes, 'numerical': 'secondary'}
        else:
            a = display_categories(first).alias('__first')
            b = display_categories(second).alias('__second')
            pairs = a.to_frame().with_columns(b).group_by('__first', '__second', drop_null_keys=True).agg(nw.len().alias('count'))
            left = sorted(pairs['__first'].unique().to_list())
            right = sorted(pairs['__second'].unique().to_list())
            left_index, right_index = {value: i for i, value in enumerate(left)}, {value: i for i, value in enumerate(right)}
            frequencies = np.zeros((len(left), len(right)), dtype=int)
            for av, bv, count in pairs.iter_rows():
                frequencies[left_index[av], right_index[bv]] = count
            output[other] = {'type': 'table', 'counts': frequencies.tolist(), 'first': left, 'secondary': right}
    return output
