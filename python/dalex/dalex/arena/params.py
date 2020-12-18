from abc import ABC, abstractmethod
import numpy as np
import math
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_object_dtype

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
        indexes = np.apply_along_axis(lambda x, y: not (x == y).all(), 0, explainer.data, explainer.y)
        self.variables = explainer.data.columns[indexes]
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
        self.variables = dataset.columns.astype(str).drop(target)
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
    def update_attributes(self, column):
        if is_bool_dtype(column):
            col_type = 'logical'
        elif is_numeric_dtype(column):
            col_type = 'numeric'
        elif is_object_dtype(column):
            col_type = 'categorical'
        else:
            raise Exception('Column type is not supported')
        if self.type is None:
            self.type = col_type
        elif self.type != col_type:
            raise Exception('The same name was used for columns with different type')
        if self.type == 'numeric':
            self.min = min(column.min(), math.inf if self.min is None else self.min)
            self.max = max(column.max(), -math.inf if self.max is None else self.max)
            if len(column.unique()) < 20:
                self.levels = np.unique((self.levels or []) + column.unique().tolist()).tolist()
            if not self.levels is None and len(self.levels) >= 20:
                self.levels = None
        else:
            self.levels = np.unique((self.levels or []) + column.unique().tolist()).tolist()
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
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index
    def get_label(self):
        return self.index
    def get_type(self):
        return 'observation'
    def get_row(self):
        return self.dataset.loc[self.index]
    def get_row_for_model(self, model_param):
        try:
            return self.dataset.loc[[self.index]][model_param.explainer.data.columns]
        except:
            return None
    def get_attributes(self):
        return dict(self.get_row())
    @staticmethod
    def list_attributes(arena):
        datasets = []
        for obs in arena.get_params('observation'):
            if len([1 for dataset in datasets if dataset is obs.dataset]) == 0:
                datasets.append(obs.dataset)
        columns = list(map(lambda x: x.columns.tolist(), datasets))
        # flatten
        columns = [col for columns_list in columns for col in columns_list]
        columns = np.unique(columns).tolist()
        return columns
