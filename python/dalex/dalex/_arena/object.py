import threading
import time
import numpy as np
from datetime import datetime
from dalex import Explainer
from pandas.core.frame import DataFrame
from .server import start_server
from .helper import get_variables
from .plots._break_down_container import BreakDownContainer
from .plots._shapley_values_container import ShapleyValuesContainer
from .plots._feature_importance_container import FeatureImportanceContainer
from .plots._partial_dependence_container import PartialDependenceContainer
from .plots._accumulated_dependence_container import AccumulatedDependenceContainer
from .plots._ceteris_paribus_container import CeterisParibusContainer


class Arena:
    def __init__(self, live=True):
        self.models = []
        self.observations = []
        self.datasets = []
        self.live = live
        self.server_thread = None
        self.timestamp = datetime.timestamp(datetime.now())
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
                   arena_url='https://arena.drwhy.ai/'):
        if self.server_thread:
            raise Exception('Server is already running')
        self.server_thread = threading.Thread(target=start_server, args=(self, host, port))
        self.server_thread.start()

    def update_timestamp(self):
        now = datetime.now()
        self.timestamp = datetime.timestamp(now)

    def stop_server(self):
        if not self.server_thread:
            raise Exception('Server is not running')
        self._stop_server()
        self.server_thread.join()
        self.server_thread = None

    def push_model(self, explainer):
        if not isinstance(explainer, Explainer):
            raise Exception('Invalid Explainer argument')
        if explainer.label in self.list_models():
            raise Exception('Explainer with the same label was already added')
        with self.mutex:
            self.update_timestamp()
            self.models.append(explainer)

    def push_observations(self, observations):
        if not isinstance(observations, DataFrame):
            raise Exception('Observations argument is not a pandas DataFrame')
        if len(observations.index.names) != 1:
            raise Exception('Observations argument need to have only one index')
        if not observations.index.is_unique:
            raise Exception('Observations argument need to have unique indexes')
        old_observations = self.list_observations()
        observations = observations.set_index(observations.index.astype(str))
        for x in observations.index:
            if x in old_observations:
                raise Exception('Indexes of observations need to be unique across all observations')
        with self.mutex:
            self.update_timestamp()
            self.observations.append(observations)

    def push_dataset(self, dataset, target, label):
        if not isinstance(dataset, DataFrame):
            raise Exception('Dataset argument is not a pandas DataFrame')
        if len(dataset.columns.names) != 1:
            raise Exception('Dataset argument need to have only one level column names')
        target = str(target)
        if target not in dataset.columns:
            raise Exception('Target is not a column from dataset')
        if (not isinstance(label, str)) or (len(label) == 0):
            raise Exception('Label need to be at least one letter')
        if label in self.list_datasets():
            raise Exception('Labels need to be unique')
        with self.mutex:
            self.update_timestamp()
            self.datasets.append({
                'dataset': dataset,
                'label': label,
                'target': target,
                'variables': dataset.columns.astype(str).drop(target)
            })

    def get_models(self):
        with self.mutex:
            result = self.models
        return result

    def get_observations(self):
        with self.mutex:
            result = self.observations
        return result

    def get_datasets(self):
        with self.mutex:
            result = self.datasets
        return result

    def list_models(self):
        with self.mutex:
            result = [x.label for x in self.models]
        return result

    def list_observations(self):
        with self.mutex:
            result = [row for batch in self.observations for row in batch.index]
        return result

    def list_datasets(self):
        with self.mutex:
            result = [x.get('label') for x in self.datasets]
        return result

    def list_variables(self):
        with self.mutex:
            result_datasets = [col for dataset in self.datasets for col in dataset.get('variables')]
            result_explainers = [col for explainer in self.models for col in get_variables(explainer)]
            result = np.unique(result_datasets + result_explainers).tolist()
        return result

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
            self.update_timestamp()
            self.options.get(plot_type)[option] = value
