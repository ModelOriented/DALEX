from abc import ABC, abstractmethod
import numpy as np

class Param(ABC):
    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def get_label(self):
        pass

class ModelParam(Param):
    def __init__(self, explainer):
        self.explainer = explainer
        indexes = np.apply_along_axis(lambda x, y: not (x == y).all(), 0, explainer.data, explainer.y)
        self.variables = explainer.data.columns[indexes]
    def get_label(self):
        return self.explainer.label
    def get_type(self):
        return 'model'

class DatasetParam(Param):
    def __init__(self, dataset, label, target):
        self.dataset = dataset
        self.label = label
        self.target = target
        self.variables = dataset.columns.astype(str).drop(target)
    def get_label(self):
        return self.explainer.label
    def get_type(self):
        return 'dataset'

class VariableParam(Param):
    def __init__(self, variable):
        self.variable = variable
    def get_label(self):
        return self.variable
    def get_type(self):
        return 'variable'

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
