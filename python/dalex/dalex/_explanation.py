from abc import ABC, abstractmethod
import narwhals.stable.v2 as nw
from narwhals.dependencies import is_pandas_dataframe


def export_result_frame(data, backend):
    """Export typed values without a Python-scalar datetime/nullable round-trip."""
    if is_pandas_dataframe(data) and not data.columns.is_unique:
        if backend in ('native', 'pandas'):
            return data.copy(deep=True)
        raise ValueError('Duplicate pandas column labels cannot be exported to this backend; rename the result columns first')
    frame = nw.from_native(data, eager_only=True)
    namespace = nw.get_native_namespace(frame)
    if backend == 'native' or namespace.__name__ == backend:
        return frame.clone().to_native()
    if backend == 'pandas':
        from ._dataframe import to_pandas
        return to_pandas(data)
    if not all(isinstance(name, str) for name in frame.columns):
        raise ValueError('This backend requires string column names; retain the pandas result for non-string labels')
    # Scalar dictionaries can truncate nanosecond timestamps and erase nullable
    # extension types. Arrow carries typed buffers across this explicit boundary.
    arrow = frame.to_arrow().select(frame.columns).replace_schema_metadata(None)
    return nw.from_arrow(arrow, backend=backend).to_native()


class CompatibilityTable:
    """Materialize legacy pandas tables only when their public attribute is read.

    Store under the historical attribute name so older pickles remain readable.
    A materialized pandas view is authoritative: callers have always been able
    to mutate ``explanation.result`` in place.
    """

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = instance.__dict__.get(self.name)
        if value is None or is_pandas_dataframe(value):
            return value
        key = '_compatibility_' + self.name
        if key not in instance.__dict__:
            if callable(getattr(type(value), '_dalex_pandas_view', None)):
                instance.__dict__[key] = value._dalex_pandas_view()
                return instance.__dict__[key]
            try:
                frame = nw.from_native(value, eager_only=True)
            except TypeError:
                return value  # SHAP arrays and other non-tabular results
            if not isinstance(frame, nw.DataFrame):
                return value
            instance.__dict__[key] = frame.to_pandas()
        return instance.__dict__[key]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value
        instance.__dict__.pop('_compatibility_' + self.name, None)

class ResultTables:
    result = CompatibilityTable()

    def result_as(self, backend='native', *, table='result', partition=None):
        """Export an independent result table without changing the legacy API.

        ``native`` retains the table's computation backend. Explicit backends
        are ``pandas``, ``polars`` and ``pyarrow``. pandas indexes remain indexes
        on pandas exports; they are not silently added as feature columns on
        other backends. In-place edits to the public pandas table are respected.
        Non-tabular results can only be exported with ``backend='native'``.
        """
        if backend not in ('native', 'pandas', 'polars', 'pyarrow'):
            raise ValueError("backend must be 'native', 'pandas', 'polars' or 'pyarrow'")
        if not isinstance(getattr(type(self), table, None), CompatibilityTable):
            raise ValueError(f'No exportable table {table!r}')
        original = self.__dict__.get(table)
        value = self.__dict__.get('_compatibility_' + table, original)
        if value is None:
            return None
        if callable(getattr(type(original), '_dalex_native_view', None)):
            cached = self.__dict__.get('_compatibility_' + table)
            return original._dalex_native_view(backend, partition=partition, pandas_view=cached)
        if partition is not None:
            raise ValueError('This result table has no partitions')
        if is_pandas_dataframe(value) and not value.columns.is_unique:
            if backend == 'native' and not is_pandas_dataframe(original):
                backend = nw.get_native_namespace(nw.from_native(original, eager_only=True)).__name__
            return export_result_frame(value, backend)
        try:
            frame = nw.from_native(value, eager_only=True)
        except TypeError:
            if backend != 'native':
                raise TypeError('This explanation result is not a dataframe') from None
            from copy import deepcopy
            return deepcopy(value)
        if not isinstance(frame, nw.DataFrame):
            raise TypeError('This explanation result is not a dataframe')
        namespace = nw.get_native_namespace(frame)
        if backend == 'native':
            namespace = nw.get_native_namespace(nw.from_native(original, eager_only=True))
        elif namespace.__name__ != backend:
            namespace = backend
        return export_result_frame(value, namespace.__name__ if not isinstance(namespace, str) else namespace)

class Explanation(ResultTables, ABC):
    @abstractmethod
    def fit(self):
        pass
    @abstractmethod
    def plot(self):
        pass


def set_result_column(explanation, name, value):
    """Apply a legacy result override without publishing a pandas view."""
    original = explanation.__dict__.get('result')
    if '_compatibility_result' in explanation.__dict__ or is_pandas_dataframe(original):
        explanation.result[name] = value
    elif callable(getattr(type(original), '_dalex_set_column', None)):
        original._dalex_set_column(name, value)
    else:
        try:
            frame = nw.from_native(original, eager_only=True)
        except TypeError:
            explanation.result[name] = value  # Retain non-tabular legacy behavior.
        else:
            explanation.result = frame.with_columns(nw.lit(value).alias(name)).to_native()


class EmptyTable:
    """Unfitted explanations retain a lazily created empty pandas result."""

    def _dalex_pandas_view(self):
        import pandas as pd
        return pd.DataFrame()

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        if partition is not None:
            raise ValueError('This result table has no partitions')
        if pandas_view is not None:
            frame = nw.from_native(pandas_view)
            if backend in ('native', 'pandas'):
                return frame.clone().to_native()
            return nw.from_dict(frame.to_dict(as_series=False), backend=backend).to_native()
        return nw.from_dict({}, backend='pandas' if backend == 'native' else backend).to_native()


class IndexedTable:
    """Native result values with separate legacy pandas-index metadata."""

    def __init__(self, data, index=None, index_column=None):
        self.data = data
        self.index = index
        self.index_column = index_column

    def _dalex_set_column(self, name, value):
        frame = nw.from_native(self.data, eager_only=True)
        if getattr(self, 'index_column', None) == name:
            # Historical label overrides add a column while keeping the
            # original performance-table index, rather than renaming it.
            if self.index is None:
                self.index = frame[name].to_numpy().copy()
            self.index_column = None
        self.data = frame.with_columns(nw.lit(value).alias(name)).to_native()

    def _dalex_pandas_view(self):
        from ._dataframe import to_pandas
        result = to_pandas(self.data)
        index_column = getattr(self, 'index_column', None)
        if index_column is not None:
            result = result.set_index(index_column)
            result.index.name = None
        if self.index is not None:
            result.index = self.index
        return result

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        if partition is not None:
            raise ValueError('This result table has no partitions')
        if backend == 'pandas':
            return pandas_view.copy() if pandas_view is not None else self._dalex_pandas_view()
        original = nw.from_native(self.data, eager_only=True)
        namespace = nw.get_native_namespace(original)
        source = nw.from_native(pandas_view, eager_only=True) if pandas_view is not None else original
        index_column = getattr(self, 'index_column', None)
        if pandas_view is not None and index_column is not None:
            source = source.with_columns(nw.new_series(index_column, pandas_view.index.to_numpy(),
                                                       backend=nw.get_native_namespace(source)))
        if backend == 'native':
            backend = namespace.__name__
        return export_result_frame(source.to_native(), backend)


class PandasFeatureTable:
    """Publish legacy duplicate-label features beside native diagnostic vectors.

    Narwhals requires unique physical column names. Original pandas features
    are a compatibility snapshot, not a second statistical implementation.
    """

    def __init__(self, features, values, scalar_columns=None):
        self.features = features
        self.values = values
        self.scalar_columns = scalar_columns or {}

    def _dalex_set_column(self, name, value):
        frame = nw.from_native(self.values, eager_only=True)
        self.values = frame.with_columns(nw.lit(value).alias(name)).to_native()
        self.scalar_columns[name] = value

    def _dalex_pandas_view(self):
        from pandas import MultiIndex
        frame = nw.from_native(self.values, eager_only=True)
        assignments = {name: frame[name].to_numpy().copy() for name in frame.columns}
        assignments.update(self.scalar_columns)
        # This explicit publication retains pandas' duplicate/partial-MultiIndex
        # assignment, column order, original row index and extension dtypes.
        result = self.features.copy(deep=True)
        for name, value in assignments.items():
            key = name
            if isinstance(result.columns, MultiIndex) and name not in result.columns:
                # Preserve the old padded-label layout without pandas 3's
                # deprecated partial key for adding a new MultiIndex column.
                key = (name,) + ('',) * (result.columns.nlevels - 1)
            result[key] = value
        return result

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        if partition is not None:
            raise ValueError('This result table has no partitions')
        result = pandas_view if pandas_view is not None else self._dalex_pandas_view()
        return export_result_frame(result, backend)
