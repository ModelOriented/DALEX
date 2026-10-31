from .._dataframe import column_names
from .._explanation import CompatibilityTable, IndexedTable
from abc import ABC, abstractmethod
import numpy as np
import math
import pandas as pd
import narwhals.stable.v2 as nw
from .._dataframe import column_values, frame_from_values, observation_values, observation_display_values, select_columns, take_rows

class Param(ABC):
    @abstractmethod
    def get_type(self):
        pass
    @abstractmethod
    def get_label(self):
        pass
    @abstractmethod
    def get_attributes(self):
        pass
    @staticmethod
    @abstractmethod
    def list_attributes(arena):
        pass
    @staticmethod
    def get_param_class(param_type):
        return {'model': ModelParam, 'variable': VariableParam, 'observation': ObservationParam, 'dataset': DatasetParam}.get(param_type)


class ModelParam(Param):
    def __init__(self, explainer):
        self.explainer = explainer
        indexes = []
        for name in column_names(explainer.data):
            equal = False
            if explainer.y is not None:
                try:
                    equal = np.array_equal(column_values(explainer.data, name), explainer.y)
                except (TypeError, ValueError, NotImplementedError):
                    # Structured feature columns cannot equal a scalar target.
                    pass
            indexes.append(not equal)
        names = column_names(explainer.data)
        self.variables = (pd.Index(names)[indexes] if isinstance(explainer.data, pd.DataFrame)
                          else np.asarray(names, dtype=object)[indexes])
    def get_label(self):
        return self.explainer.label
    def get_type(self):
        return 'model'
    def get_attributes(self):
        return {}
    @staticmethod
    def list_attributes(arena):
        return []

class DatasetParam(Param):
    def __init__(self, dataset, label, target):
        self.dataset = dataset
        self.label = label
        self.target = target
        self.variables = (dataset.columns.astype(str).drop(target) if isinstance(dataset, pd.DataFrame)
                          else [str(name) for name in column_names(dataset) if str(name) != target])
    def get_label(self):
        return self.label
    def get_type(self):
        return 'dataset'
    def get_attributes(self):
        return {}
    @staticmethod
    def list_attributes(arena):
        return []

class VariableParam(Param):
    def __init__(self, variable):
        self.variable = variable
        self.clear_attributes()
    def get_label(self):
        return self.variable
    def get_type(self):
        return 'variable'
    def clear_attributes(self):
        self.type = None
        self.min = None
        self.max = None
        self.levels = None
        self._many_levels = False
    def update_attributes(self, column):
        column = nw.from_native(column, series_only=True)
        dtype = column.dtype
        temporal = isinstance(dtype, (nw.Date, nw.Datetime, nw.Duration))
        if dtype == nw.Boolean:
            col_type = 'logical'
        elif dtype.is_numeric():
            col_type = 'numeric'
        elif temporal or isinstance(dtype, (nw.String, nw.Categorical, nw.Enum, nw.Object)):
            col_type = 'categorical'
        elif dtype == nw.Null:
            col_type = 'categorical'
        else:
            raise Exception('Column type is not supported')
        if self.type is None:
            self.type = col_type
        elif self.type != col_type:
            raise Exception('The same name was used for columns with different type')
        present = column.drop_nulls()
        if dtype.is_float():
            present = present.filter(~present.is_nan())
        values = present.cast(nw.String).unique().to_list() if temporal else present.unique().to_list()
        if self.type == 'numeric':
            if len(present):
                self.min = min(present.min(), math.inf if self.min is None else self.min)
                self.max = max(present.max(), -math.inf if self.max is None else self.max)
            if not self._many_levels:
                levels = np.unique((self.levels or []) + values).tolist()
                self._many_levels = len(levels) >= 20
                self.levels = None if self._many_levels else levels
        else:
            self.levels = np.unique((self.levels or []) + values).tolist()

    def get_attributes(self):
        return {
            'type': self.type,
            'min': self.min,
            'max': self.max,
            'levels': self.levels
        }
    @staticmethod
    def list_attributes(arena):
        return ['type', 'min', 'max', 'levels']

class ObservationParam(Param):
    dataset = CompatibilityTable()
    def __init__(self, dataset, index, native_dataset=None, position=None):
        self.dataset = dataset
        self.index = index
        self.native_dataset = native_dataset
        self.position = position
    def get_label(self):
        return self.index
    def get_type(self):
        return 'observation'
    def get_row(self):
        return self.dataset.loc[self.index]
    def get_row_for_model(self, model_param):
        try:
            reference = model_param.explainer.data
            names = list(column_names(reference))
            if self.native_dataset is not None:
                native = select_columns(take_rows(self.native_dataset, [self.position]), names)
                if type(native) is type(reference):
                    return native
                values = observation_values(native)
            else:
                row = self.dataset.loc[[self.index], names]
                if isinstance(reference, pd.DataFrame):
                    return row
                values = observation_values(row, row)
            # A whole-row NumPy export promotes mixed integer/float rows and
            # loses large UInt64 cells. Fill an explicit object matrix by column.
            matrix = np.empty((1, len(names)), dtype=object)
            for i, name in enumerate(names):
                matrix[0, i] = values[name]
            return frame_from_values(matrix, reference)
        except:
            return None
    def get_attributes(self):
        if self.native_dataset is not None:
            native = take_rows(self.native_dataset, [self.position])
            values = observation_values(native)
            display = observation_display_values(native, values)
            return {name: None if value is None else display[name] for name, value in values.items()}
        return {name: self.dataset.at[self.index, name] for name in self.dataset.columns}
    @staticmethod
    def list_attributes(arena):
        names = set()
        for observation in arena.get_params('observation'):
            dataset = observation.native_dataset if observation.native_dataset is not None else observation.dataset
            names.update(column_names(dataset))
        return sorted(names)


class ObservationDataset(IndexedTable):
    def _dalex_pandas_view(self):
        if not hasattr(self, '_pandas_cache'):
            self._pandas_cache = super()._dalex_pandas_view()
        return self._pandas_cache
