import threading
import time
import numpy as np
from datetime import datetime
from dalex import Explainer
from pandas.core.frame import DataFrame
from .server import start_server
from ._plot_container import PlotContainer
from .params import ModelParam, DatasetParam, VariableParam, ObservationParam
from .plots._break_down_container import BreakDownContainer
from .plots._shapley_values_container import ShapleyValuesContainer
from .plots._feature_importance_container import FeatureImportanceContainer
from .plots._partial_dependence_container import PartialDependenceContainer
from .plots._accumulated_dependence_container import AccumulatedDependenceContainer
from .plots._ceteris_paribus_container import CeterisParibusContainer


class Arena:
    def __init__(self, precalculate=False):
        self.models = []
        self.observations = []
        self.datasets = []
        self.server_thread = None
        self.precalculate = bool(precalculate)
        self.timestamp = datetime.timestamp(datetime.now())
        self.cache = []
        self.mutex = threading.Lock()
        self.plots = [
            ShapleyValuesContainer,
            FeatureImportanceContainer,
            PartialDependenceContainer,
            AccumulatedDependenceContainer,
            CeterisParibusContainer,
            BreakDownContainer
        ]
        self.options = {}
        for plot in self.plots:
            options = {}
            for o in plot.options.keys():
                options[o] = plot.options.get(o).get('default')
            self.options[plot.info.get('plotType')] = options

    def run_server(self,
                   host='127.0.0.1',
                   port=8181,
                   open_browser=True,
                   append_data=False,
                   arena_url='https://arena.drwhy.ai/',
                   disable_logs=True):
        if self.server_thread:
            raise Exception('Server is already running')
        self.server_thread = threading.Thread(target=start_server, args=(self, host, port, disable_logs))
        self.server_thread.start()

    def stop_server(self):
        if not self.server_thread:
            raise Exception('Server is not running')
        self._stop_server()
        self.server_thread.join()
        self.server_thread = None

    def update_timestamp(self):
        now = datetime.now()
        self.timestamp = datetime.timestamp(now)

    def clear_cache(self, plot_type=None):
        self.cache = list(filter(lambda p: p.plot_type != plot_type, self.cache))
        self.update_timestamp()

    def find_in_cache(self, plot_type, params):
        _filter = lambda p: p.plot_type == plot_type and params == p.params
        with self.mutex:
            result = next(filter(_filter, self.cache), None)
        return result
    
    def put_to_cache(self, plot_container):
        if not isinstance(plot_container, PlotContainer):
            raise Exception('Invalid plot container')
        with self.mutex:
            self.cache.append(plot_container)

    def fill_cache(self, fixed_params={}):
        if not isinstance(fixed_params, dict):
            raise Exception('Params argument must be a dict')
        for plot_class in self.plots:
            required_params = plot_class.info.get('requiredParams')
            if [k for k in fixed_params.keys() if not k in required_params]:
                continue
            available_params = self.get_available_params()
            iteration_pools = map(lambda p: available_params.get(p) if fixed_params.get(p) is None else [fixed_params.get(p)], required_params)
            combinations = [[]]
            for pool in iteration_pools:
                combinations = [x + [y] for x in combinations for y in pool]
            for params_values in combinations:
                params = dict(zip(required_params, params_values))
                self.get_plot(plot_type=plot_class.info.get('plotType'), params_values=params)

    def push_model(self, explainer, precalculate=None):
        if not isinstance(explainer, Explainer):
            raise Exception('Invalid Explainer argument')
        if explainer.label in self.list_params('model'):
            raise Exception('Explainer with the same label was already added')
        precalculate = self.precalculate if precalculate is None else bool(precalculate)
        param = ModelParam(explainer)
        with self.mutex:
            self.update_timestamp()
            self.models.append(param)
        if precalculate:
            self.fill_cache({'model': param})

    def push_observations(self, observations, precalculate=None):
        if not isinstance(observations, DataFrame):
            raise Exception('Observations argument is not a pandas DataFrame')
        if len(observations.index.names) != 1:
            raise Exception('Observations argument need to have only one index')
        if not observations.index.is_unique:
            raise Exception('Observations argument need to have unique indexes')
        precalculate = self.precalculate if precalculate is None else bool(precalculate)
        old_observations = self.list_params('observation')
        observations = observations.set_index(observations.index.astype(str))
        params_objects = []
        for x in observations.index:
            if x in old_observations:
                raise Exception('Indexes of observations need to be unique across all observations')
            params_objects.append(ObservationParam(dataset=observations, index=x))
        with self.mutex:
            self.update_timestamp()
            self.observations.extend(params_objects)
        if precalculate:
            for obs in params_objects:
                self.fill_cache({'observation': obs})

    def push_dataset(self, dataset, target, label, precalculate=None):
        if not isinstance(dataset, DataFrame):
            raise Exception('Dataset argument is not a pandas DataFrame')
        if len(dataset.columns.names) != 1:
            raise Exception('Dataset argument need to have only one level column names')
        precalculate = self.precalculate if precalculate is None else bool(precalculate)
        target = str(target)
        if target not in dataset.columns:
            raise Exception('Target is not a column from dataset')
        if (not isinstance(label, str)) or (len(label) == 0):
            raise Exception('Label need to be at least one letter')
        if label in self.list_params('dataset'):
            raise Exception('Labels need to be unique')
        param = DatasetParam(dataset=dataset, label=label, target=target)
        with self.mutex:
            self.update_timestamp()
            self.datasets.append(param)
        if precalculate:
            self.fill_cache({'dataset': param})

    def get_params(self, param_type):
        if param_type == 'observation':
            with self.mutex:
                result = self.observations
        elif param_type == 'variable':
            with self.mutex:
                result_datasets = [col for dataset in self.datasets for col in dataset.variables]
                result_explainers = [col for model in self.models for col in model.variables]
                result_str = np.unique(result_datasets + result_explainers).tolist()
                result = [VariableParam(x) for x in result_str]
        elif param_type == 'model':
            with self.mutex:
                result = self.models
        elif param_type == 'dataset':
            with self.mutex:
                result = self.datasets
        else:
            raise Exception('Invalid param type')
        return result

    def list_params(self, param_type):
        return [x.get_label() for x in self.get_params(param_type)]

    def get_available_params(self):
        result = {}
        for param_type in ['model', 'observation', 'variable', 'dataset']:
            result[param_type] = self.get_params(param_type)
        return result

    def list_available_params(self):
        result = {}
        for param_type in ['model', 'observation', 'variable', 'dataset']:
            result[param_type] = self.list_params(param_type)
        return result
    
    def find_param_value(self, param_type, param_label):
        if param_label is None or not isinstance(param_label, str):
            return None
        return next((x for x in self.get_params(param_type) if x.get_label() == param_label), None)

    def print_options(self, plot_type=None):
        plot = next((x for x in self.plots if x.info.get('plotType') == plot_type), None)
        if plot_type is None or plot is None:
            for plot in self.plots:
                self.print_options(plot.info.get('plotType'))
            return
        print('\n\033[1m' + plot.info.get('plotType') + '\033[0m')
        print('---------------------------------')
        for o in plot.options.keys():
            option = plot.options.get(o)
            value = self.options.get(plot_type).get(o)
            print(o + ': ' + str(value) + '   #' + option.get('desc'))

    def get_option(self, plot_type, option):
        options = self.options.get(plot_type)
        if options is None:
            raise Exception('Invalid plot_type')
        if not option in options.keys():
            return
        with self.mutex:
            result = self.options.get(plot_type).get(option)
        return result

    def set_option(self, plot_type, option, value):
        if plot_type is None:
            for plot in self.plots:
                self.set_option(plot.info.get('plotType'), option, value)
            return
        options = self.options.get(plot_type)
        if options is None:
            raise Exception('Invalid plot_type')
        if not option in options.keys():
            return
        with self.mutex:
            self.options.get(plot_type)[option] = value
            self.clear_cache(plot_type)
        if self.precalculate:
            self.fill_cache()

    def get_plot(self, plot_type, params_values):
        plot_class = next((c for c in self.plots if c.info.get('plotType') == plot_type), None)
        if plot_class is None:
            raise Exception('Not supported plot type')
        plot_type = plot_class.info.get('plotType')
        required_params_values = {}
        required_params_labels = {}
        for p in plot_class.info.get('requiredParams'):
            if params_values.get(p) is None:
                raise Exception('Required param is missing')
            required_params_values[p] = params_values.get(p)
            required_params_labels[p] = params_values.get(p).get_label()
        result = self.find_in_cache(plot_type, required_params_labels)
        if result is None:
            result = plot_class(self).fit(required_params_values)
            self.put_to_cache(result)
        return result

