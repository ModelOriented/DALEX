"""Native background results with lazy historical resource dictionary values."""
import numpy as np
import narwhals.stable.v2 as nw
from .._dataframe import to_pandas, take_rows
from .._explanation import export_result_frame
from ..predict_explanations._ceteris_paribus._table import concat_compatible


class ResourceData(dict):
    def __init__(self):
        super().__init__()
        self._legacy = {}

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self._legacy.pop(key, None)

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if callable(getattr(type(value), '_dalex_pandas_view', None)):
            if key not in self._legacy:
                self._legacy[key] = value._dalex_pandas_view()
            return self._legacy[key]
        return value

    def native(self, key):
        return self._legacy.get(key, dict.get(self, key))

    def __iter__(self):
        # Also makes dict(resource.data) use projected public values.
        return super().__iter__()

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return self.copy().items()

    def values(self):
        return self.copy().values()

    def copy(self):
        return {key: self[key] for key in self}

    def update(self, values=(), **kwargs):
        for key, value in dict(values, **kwargs).items():
            self[key] = value

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._legacy.pop(key, None)

    def clear(self):
        dict.clear(self)
        self._legacy.clear()

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *default):
        if key not in self:
            return dict.pop(self, key, *default)
        value = self[key]
        del self[key]
        return value

    def __reduce__(self):
        # Keep native storage and any authoritative pandas edits separate when
        # snapshots are pickled, without triggering pandas publication.
        return (_restore_resource_data, ({key: value for key, value in dict.items(self)}, self._legacy))


def _restore_resource_data(values, legacy):
    result = ResourceData()
    for key, value in values.items():
        result[key] = value
    result._legacy = legacy
    return result


def clean_contributions(frame):
    """Match pandas reductions' omission of NaNs on every backend."""
    return frame.with_columns(nw.when(nw.col('contribution').is_nan()).then(None)
                              .otherwise(nw.col('contribution')).cast(nw.Float64).alias('contribution'))


class ShapRecords:
    def __init__(self, partitions, count):
        self.partitions = partitions
        self.count = count

    @classmethod
    def from_shap(cls, shap, row=None):
        original = shap.__dict__['result']
        cached = shap.__dict__.get('_compatibility_result')
        if cached is None and not callable(getattr(type(original), '_dalex_pandas_view', None)):
            cached = to_pandas(original)
        if cached is not None:
            cached = cached.loc[cached.B != 0].copy()
            if row is not None:
                cached['row'] = row
            return cls.from_legacy(cached)
        raw = nw.from_native(original.data, eager_only=True)
        observation = nw.from_native(original.observation, eager_only=True)
        names = raw['variable_name'].unique(maintain_order=True).to_list()
        variables = raw['variable_name'].to_numpy()
        batches = raw['B'].to_numpy()
        parts = {}
        for name in names:
            positions = np.flatnonzero((variables == name) & (batches != 0))
            table = raw.filter((nw.col('variable_name') == name) & (nw.col('B') != 0))
            value = nw.from_native(take_rows(observation.select(name).to_native(), np.zeros(len(table), dtype=np.int64)), eager_only=True)[name]
            table = table.with_columns(value.alias('variable_value'))
            table = table.select('variable', 'contribution', 'variable_name', 'variable_value', 'sign', 'label', 'B')
            backend = nw.get_native_namespace(table)
            table = table.with_columns(nw.new_series('__order', positions, backend=backend),
                                       nw.new_series('__index', positions % len(names), backend=backend))
            if row is not None:
                table = table.with_columns(nw.lit(row).cast(nw.Int64).alias('row'))
            parts[name] = table.to_native()
        return cls(parts, int(np.count_nonzero(batches)))

    @classmethod
    def from_legacy(cls, data):
        frame = nw.from_native(data, eager_only=True).with_row_index('__order')
        backend = nw.get_native_namespace(frame)
        frame = frame.with_columns(nw.new_series('__index', data.index.to_numpy(), backend=backend))
        names = frame['variable_name'].unique(maintain_order=True).to_list()
        return cls({name: frame.filter(nw.col('variable_name') == name).to_native() for name in names}, len(frame))

    def append(self, other):
        parts = dict(self.partitions)
        for name, part in other.partitions.items():
            frame = nw.from_native(part, eager_only=True).with_columns(nw.col('__order') + self.count)
            if name in parts:
                old = nw.from_native(parts[name], eager_only=True)
                backend = nw.get_native_namespace(frame).__name__
                if nw.get_native_namespace(old).__name__ != backend:
                    old = nw.from_native(export_result_frame(parts[name], backend), eager_only=True)
                frame = nw.concat([old, frame])
            parts[name] = frame.to_native()
        return ShapRecords(parts, self.count + other.count)

    def numeric(self):
        return nw.concat([nw.from_native(part, eager_only=True).drop('variable_value')
                          for part in self.partitions.values()])

    def _dalex_pandas_view(self):
        import pandas as pd
        result = pd.concat([to_pandas(part) for part in self.partitions.values()]).sort_values('__order', kind='stable')
        result.index = result.pop('__index').to_numpy()
        result.index.name = None
        return result.drop(columns='__order')

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        if pandas_view is not None:
            if backend == 'native':
                backend = nw.get_native_namespace(nw.from_native(next(iter(self.partitions.values())), eager_only=True)).__name__
            if partition is not None:
                frame = nw.from_native(pandas_view, eager_only=True).filter(nw.col('variable_name') == partition)
                reference = nw.from_native(self.partitions[partition], eager_only=True)['variable_value'].dtype
                pandas_view = frame.with_columns(nw.col('variable_value').cast(reference)).to_pandas()
            return ShapRecords.from_legacy(pandas_view)._dalex_native_view(backend, partition=partition)
        if backend == 'pandas':
            result = self._dalex_pandas_view()
            return result if partition is None else result[result.variable_name == partition].copy()
        if partition is not None:
            if partition not in self.partitions:
                raise ValueError(f'Unknown resource result partition {partition!r}')
            parts = [self.partitions[partition]]
        else:
            parts = list(self.partitions.values())
        frames = [nw.from_native(part, eager_only=True).drop('__order', '__index').to_native() for part in parts]
        return export_result_frame(concat_compatible(frames), backend)


def records(value):
    return value if isinstance(value, ShapRecords) else ShapRecords.from_legacy(value)


class ShapStats:
    def __init__(self, data, sources):
        self.data = data
        self.sources = sources

    def display_values(self):
        from ._native_eda import display_categories
        return {name: display_categories(nw.from_native(part, eager_only=True)['variable_value'])[0]
                for name, part in self.sources.items()}

    def _dalex_pandas_view(self):
        result = to_pandas(self.data)
        values = {name: to_pandas(part)['variable_value'].iloc[0] for name, part in self.sources.items()}
        result['variable_value'] = [values[name] for name in result.variable_name]
        return result.set_index(['variable_name', 'variable_value'])

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        if pandas_view is not None:
            if backend == 'pandas':
                return pandas_view.copy()
            frame = nw.from_native(pandas_view.reset_index(), eager_only=True)
            if partition is not None:
                frame = frame.filter(nw.col('variable_name') == partition)
                reference = nw.from_native(self.sources[partition], eager_only=True)['variable_value'].dtype
                frame = frame.with_columns(nw.col('variable_value').cast(reference))
            target = nw.get_native_namespace(nw.from_native(self.data, eager_only=True)).__name__ if backend == 'native' else backend
            return export_result_frame(frame.to_native(), target)
        if backend == 'pandas':
            return self._dalex_pandas_view()
        frame = nw.from_native(self.data, eager_only=True)
        names = frame['variable_name'].to_list() if partition is None else [partition]
        parts = []
        for name in names:
            value = nw.from_native(self.sources[name], eager_only=True)['variable_value'].head(1)
            parts.append(frame.filter(nw.col('variable_name') == name).with_columns(value).to_native())
        return export_result_frame(concat_compatible(parts), backend)


def stats(value):
    collection = records(value)
    rows, sources = [], {}
    for name, part in sorted(collection.partitions.items()):
        frame = nw.from_native(part, eager_only=True)
        cell = frame['variable_value'].head(1)
        if cell.is_null()[0] or cell.dtype.is_float() and cell.is_nan()[0]:
            continue  # historical groupby excluded missing display memberships
        sample = frame['contribution'].to_numpy()
        sample = sample[~np.isnan(sample)]
        if len(sample):
            mean, low, high = sample.mean(), sample.min(), sample.max()
            q1, q3 = np.quantile(sample, [.25, .75])
        else:
            mean = low = high = q1 = q3 = np.nan
        rows.append((name, mean, high, low, q1, q3, abs(mean)))
        sources[name] = part
    columns = ['variable_name', 'mean', 'max', 'min', 'q1', 'q3', 'abs']
    backend = nw.get_native_namespace(nw.from_native(next(iter(collection.partitions.values())), eager_only=True))
    data = {name: [row[i] for row in rows] if i == 0 else np.asarray([row[i] for row in rows], dtype=float)
            for i, name in enumerate(columns)}
    return ShapStats(nw.from_dict(data, backend=backend).to_native(), sources)
