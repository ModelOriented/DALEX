"""Native fairness tables with compatible subgroup indexes and parity Series."""
from ..._explanation import IndexedTable, export_result_frame
from ..._dataframe import to_pandas


class FairnessTable(IndexedTable):
    def __init__(self, data, *, index_name=None, empty_objects=False):
        super().__init__(data, index_column='subgroup')
        self.index_name = index_name
        self.empty_objects = empty_objects

    def _dalex_pandas_view(self):
        result = super()._dalex_pandas_view()
        result.index.name = self.index_name
        if self.empty_objects and result.empty:
            result = result.astype(object)
            result.index = result.index.astype(object)
        return result


class ParityTable(IndexedTable):
    def _dalex_pandas_view(self):
        result = to_pandas(self.data).set_index('metric')['score']
        result.index.name = None
        result.name = None
        return result

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        if partition is not None:
            raise ValueError('This result table has no partitions')
        if backend == 'pandas':
            return pandas_view.copy() if pandas_view is not None else self._dalex_pandas_view()
        import narwhals.stable.v2 as nw
        original = nw.from_native(self.data, eager_only=True)
        if backend == 'native':
            backend = nw.get_native_namespace(original).__name__
        source = self.data if pandas_view is None else pandas_view.rename_axis('metric').reset_index(name='score')
        return export_result_frame(source, backend)
