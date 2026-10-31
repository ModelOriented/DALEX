"""Typed Aspect results with heterogeneous observation metadata kept separate."""
import narwhals.stable.v2 as nw
from .._explanation import IndexedTable, export_result_frame
from .._dataframe import to_pandas


class AspectTable(IndexedTable):
    def __init__(self, data, names, pairs, observation=None, *, value_lists=False,
                 columns=None, single=False, index=None):
        super().__init__(data, index=index)
        self.names = names
        self.pairs = pairs
        self.observation = observation
        self.value_lists = value_lists if isinstance(value_lists, list) else [value_lists] * len(names)
        self.columns = columns
        self.single = single

    def _columns(self):
        if self.columns is not None:
            return self.columns
        if self.observation is None:
            return ['aspect_name', 'variable_names', 'dropout_loss', 'dropout_loss_change',
                    'min_depend', 'vars_min_depend', 'label']
        return ['aspect_name', 'variable_names', 'variable_values', 'importance',
                'min_depend', 'vars_min_depend', 'label']

    def _dalex_pandas_view(self):
        result = to_pandas(self.data).reset_index(drop=True)
        result['variable_names'] = [names[0] for names in self.names] if self.single else self.names
        result['vars_min_depend'] = self.pairs
        if self.observation is not None:
            observation = to_pandas(self.observation)
            values = [observation[list(names)].values[0] for names in self.names]
            result['variable_values'] = ([value[0] for value in values] if self.single else
                                         [value.tolist() if as_list else value
                                          for value, as_list in zip(values, self.value_lists)])
        if self.index is not None:
            result.index = self.index
        if self.observation is not None and 'aspect_name' not in self._columns() and 'min_depend' in self._columns():
            # The historical triplot populated an object column cell by cell.
            result['min_depend'] = result['min_depend'].astype(object)
        return result[self._columns()]

    def subset(self, indices, *, columns=None, single=False, keep_index=False):
        frame = nw.from_native(self.data, eager_only=True)
        index = None
        if keep_index:
            previous = self.index if self.index is not None else list(range(len(frame)))
            index = [previous[i] for i in indices]
        return AspectTable(frame[indices, :].to_native(), [self.names[i] for i in indices],
                           [self.pairs[i] for i in indices], self.observation,
                           value_lists=[self.value_lists[i] for i in indices], columns=columns, single=single, index=index)

    def with_height(self, height):
        frame = nw.from_native(self.data, eager_only=True).with_columns(nw.lit(height).alias('h'))
        return AspectTable(frame.to_native(), self.names, self.pairs, self.observation,
                           value_lists=self.value_lists, columns=self._columns() + ['h'], index=self.index)

    @classmethod
    def concatenate(cls, tables, *, columns=None, keep_index=False):
        frames = [nw.from_native(table.data, eager_only=True) for table in tables]
        index = None
        if keep_index:
            index = [i for table, frame in zip(tables, frames)
                     for i in (table.index if table.index is not None else range(len(frame)))]
        return cls(nw.concat(frames).to_native(), [names for table in tables for names in table.names],
                   [pair for table in tables for pair in table.pairs], tables[0].observation,
                   value_lists=[flag for table in tables for flag in table.value_lists], columns=columns, index=index)

    def _value_struct(self, groups, cached=None):
        """Native structs retain heterogeneous group values without coercion."""
        import pyarrow as pa
        observation = nw.from_native(self.observation, eager_only=True).to_arrow()
        wanted = {name for group in groups for name in group}
        names = [name for name in observation.column_names if name in wanted]
        arrays = []
        for name in names:
            if cached is None:
                ids = pa.array([0 if name in group else None for group in groups], type=pa.int64())
                array = observation[name].combine_chunks().take(ids)
            else:
                values = [values[list(group).index(name)] if name in group else None
                          for group, values in zip(groups, cached)]
                array = pa.array(values, type=observation.schema.field(name).type, from_pandas=True)
            arrays.append(array)
        return pa.StructArray.from_arrays(arrays, names=names)

    def _dalex_native_view(self, backend, *, partition=None, pandas_view=None):
        original = nw.from_native(self.data, eager_only=True)
        namespace = nw.get_native_namespace(original)
        if pandas_view is not None or backend == 'pandas' or (backend == 'native' and namespace.__name__ == 'pandas'):
            result = pandas_view.copy() if pandas_view is not None else self._dalex_pandas_view()
            if partition is not None:
                result = result[result.aspect_name == partition].copy()
                if result.empty:
                    raise ValueError(f'Unknown Aspect result partition {partition!r}')
            if backend == 'pandas' or (backend == 'native' and namespace.__name__ == 'pandas'):
                return result
            if self.observation is not None:
                groups = [[name] for name in result.variable_names] if self.single else result.variable_names.tolist()
                values = [[value] for value in result.variable_values] if self.single else result.variable_values.tolist()
                arrow = nw.from_native(result.drop(columns='variable_values'), eager_only=True).to_arrow()
                arrow = arrow.append_column('variable_values', self._value_struct(groups, values))
                return nw.from_arrow(arrow.select(self._columns()), backend=namespace if backend == 'native' else backend).to_native()
            if backend == 'native':
                return export_result_frame(result, namespace.__name__)
            return export_result_frame(result, backend)
        indices = list(range(len(original)))
        if partition is not None:
            indices = [i for i, name in enumerate(original['aspect_name'].to_list()) if name == partition]
            if not indices:
                raise ValueError(f'Unknown Aspect result partition {partition!r}')
        frame = original[indices, :]
        names = [None if self.names[i] is None else list(self.names[i]) for i in indices]
        pairs = [None if self.pairs[i] is None else list(self.pairs[i]) for i in indices]
        frame = frame.with_columns(nw.new_series('variable_names', [group[0] for group in names] if self.single else names, backend=namespace),
                                   nw.new_series('vars_min_depend', pairs, backend=namespace))
        if self.observation is not None:
            arrow = frame.to_arrow().append_column('variable_values', self._value_struct(names))
            return nw.from_arrow(arrow.select(self._columns()), backend=namespace if backend == 'native' else backend).to_native()
        return export_result_frame(frame.select(self._columns()).to_native(), namespace.__name__ if backend == 'native' else backend)
