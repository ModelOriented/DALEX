import threading
import time
import numpy as np
from datetime import datetime
from dalex import Explainer
from pandas.core.frame import DataFrame
from .server import start_server
from ._plot_container import PlotContainer
from .params import ModelParam, DatasetParam, VariableParam, ObservationParam, Param
from .plots import *
from .._global_checks import global_check_import

class Arena:
    """ Creates Arena object

    This class should be used to create Python connector for Arena dashboard. Initialized
    object can work both in static and live mode. Use `push_*` methods to add your
    models, observations and datasets.

    Parameters
    ----------
    precalculate : bool
        Enables precalculating plots after using each `push_*` method.
    enable_attributes : bool
        Enables attributes of observations and variables. Attributes are required to
        display details of observation in Arena, but it also increases generated
        file size.
    enable_custom_params : bool
        Enables modififying observations in dashboard. It requires attributes and works
        only in live version.

    Attributes
    --------
    models : list of ModelParam objects
        List of pushed models encapsulated in ModelParam class
    observations : list of ObservationParam objects
        List of pushed observations encapsulated in ObservationParam class
    datasets : list of DatasetParam objects
        List of pushed datasets encapsulated in DatasetParam class
    variables_cache : list of VariableParam objects
        Cached list of VariableParam objects generated using pushed models and datasets
    server_thread : threading.Thread
        Thread of running server or None otherwise
    precalculate : bool
        if plots should be precalculated
    enable_attributes : bool
        if attributes are enabled
    enable_custom_params : bool
        if modifying observations is enabled
    timestamp : float
        timestamp of last modification
    cache : list of PlotContainer objects
        List of already calculated plots
    mutex : _thread.lock
        Mutex for params and cache
    plots : list of classes extending PlotContainer
        List of enabled plots
    options : dict
        Options for plots

    """
    def __init__(self, precalculate=False, enable_attributes=True, enable_custom_params=True):
        self.models = []
        self.observations = []
        self.datasets = []
        self.variables_cache = []
        self.server_thread = None
        self.precalculate = bool(precalculate)
        self.enable_attributes = bool(enable_attributes)
        self.enable_custom_params = bool(enable_custom_params)
        self.timestamp = datetime.timestamp(datetime.now())
        self.cache = []
        self.mutex = threading.Lock()
        self.plots = [
            ShapleyValuesContainer,
            FeatureImportanceContainer,
            PartialDependenceContainer,
            AccumulatedDependenceContainer,
            CeterisParibusContainer,
            BreakDownContainer,
            MetricsContainer
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
                   append_data=False,
                   arena_url='https://arena.drwhy.ai/',
                   disable_logs=True):
        """Starts server for live mode of Arena

        Parameters
        -----------
        host : str
            ip or hostname for the server
        port : int
            port number for the server
        append_data : bool
            if generated link should append data to already existing Arena window.
        arena_url : str
            URl of Arena dhasboard
        disable_logs : str
            if logs should be muted

        Returns
        -----------
        Link to Arena
        """
        if self.server_thread:
            raise Exception('Server is already running')
        global_check_import('flask')
        global_check_import('flask_cors')
        global_check_import('requests')
        self.server_thread = threading.Thread(target=start_server, args=(self, host, port, disable_logs))
        self.server_thread.start()
        if append_data:
            print(arena_url + '?append=http://' + host + ':' + str(port) + '/')
        else:
            print(arena_url + '?data=http://' + host + ':' + str(port) + '/')

    def stop_server(self):
        """Stops running server"""
        if not self.server_thread:
            raise Exception('Server is not running')
        self._stop_server()
        self.server_thread.join()
        self.server_thread = None

    def update_timestamp(self):
        """Updates timestamp

        Notes
        -------
        This function must be called from mutex context
        """
        now = datetime.now()
        self.timestamp = datetime.timestamp(now)

    def clear_cache(self, plot_type=None):
        """Clears cache
        
        Parameters
        -----------
        plot_type : str or None
            If None all cache is cleared. Otherwise only plots with
            provided plot_type are removed.

        Notes
        -------
        This function must be called from mutex context
        """
        self.cache = list(filter(lambda p: p.plot_type != plot_type, self.cache))
        self.update_timestamp()

    def find_in_cache(self, plot_type, params):
        """Function searches for cached plot

        Parameters
        -----------
        plot_type : str
            Value of plot_type field, that requested plot must have
        params : dict
            Keys of this dict are params types (model, observation, variable, dataset)
            and values are corresponding params labels. Requested plot must have equal
            params field.

        Returns
        --------
        PlotContainer or None
        """

        _filter = lambda p: p.plot_type == plot_type and params == p.params
        with self.mutex:
            result = next(filter(_filter, self.cache), None)
        return result
    
    def put_to_cache(self, plot_container):
        """Puts new plot to cache

        Parameters
        -----------
        plot_container : PlotContainer
        """
        if not isinstance(plot_container, PlotContainer):
            raise Exception('Invalid plot container')
        with self.mutex:
            self.cache.append(plot_container)

    def fill_cache(self, fixed_params={}):
        """Generates all available plots and cache them

        This function tries to generate all plots that are not cached already and
        put them to cache. Range of generated plots can be narrow using `fixed_params`

        Parameters
        -----------
        fixed_params : dict
            This dict specifies which plots should be generated. Only those with
            all keys from `fixed_params` present and having the same value will be
            calculated.
        """
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
        """Adds model to Arena

        This method encapsulate explainer in ModelParam object and
        save appends models fields. When precalculation is enabled
        triggers filling cache.

        Parameters
        -----------
        explainer : dalex.Explainer
            Explainer created using dalex package
        precalculate : bool or None
            Overrides constructor `precalculate` parameter when it is not None.
            If true, then only plots using this model will be precalculated.
        """
        if not isinstance(explainer, Explainer):
            raise Exception('Invalid Explainer argument')
        if explainer.label in self.list_params('model'):
            raise Exception('Explainer with the same label was already added')
        precalculate = self.precalculate if precalculate is None else bool(precalculate)
        param = ModelParam(explainer)
        with self.mutex:
            self.update_timestamp()
            self.models.append(param)
            self.variables_cache = []
        if precalculate:
            self.fill_cache({'model': param})

    def push_observations(self, observations, precalculate=None):
        """Adds observations to Arena

        Pushed observations will be used to local explainations. Function
        creates ObservationParam object for each row of pushed dataset. Label
        for each observation is taken from row name. When precalculation
        is enabled triggers filling cache.

        Parameters
        -----------
        observations : pandas.DataFrame
            Data frame of observations to be explained using instance level plots.
            Label for each observation is taken from row name.
        precalculate : bool or None
            Overrides constructor `precalculate` parameter when it is not None.
            If true, then only plots using thease observations will be precalculated.
        """
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
        """Adds dataset to Arena

        Pushed dataset will visualised using exploratory data analysis plots.
        Function creates DatasetParam object with specified label and target name.
        When precalculation is enabled triggers filling cache.

        Parameters
        -----------
        dataset : pandas.DataFrame
            Data frame to be visualised using EDA plots. This
            dataset should contain target variable.
        target : str
            Name of target column
        label : str
            Label for this dataset
        precalculate : bool or None
            Overrides constructor `precalculate` parameter when it is not None.
            If true, then only plots using this model will be precalculated.
        """
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
            self.variables_cache = []
        if precalculate:
            self.fill_cache({'dataset': param})

    def get_params(self, param_type):
        """Returns list of available params

        Parameters
        -----------
        param_type : str
            One of ['model', 'variable', 'observation', 'dataset']. Params of this type
            will be returned

        Returns
        --------
        List of Param objects
        """
        if param_type == 'observation':
            with self.mutex:
                result = self.observations
        elif param_type == 'variable':
            with self.mutex:
                if not self.variables_cache:
                    result_datasets = [col for dataset in self.datasets for col in dataset.variables]
                    result_explainers = [col for model in self.models for col in model.variables]
                    result_str = np.unique(result_datasets + result_explainers).tolist()
                    self.variables_cache = [VariableParam(x) for x in result_str]
                    if self.enable_attributes:
                        for var in self.variables_cache:
                            try:
                                for dataset in self.datasets:
                                    if var.variable in dataset.variables:
                                        var.update_attributes(dataset.dataset[var.variable])
                                for model in self.models:
                                    if var.variable in model.variables:
                                        var.update_attributes(model.explainer.data[var.variable])
                            except:
                                var.clear_attributes()
                result = self.variables_cache
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
        """Returns list of available params's labels

        Parameters
        -----------
        param_type : str
            One of ['model', 'variable', 'observation', 'dataset']. Labels of params
            of this type will be returned

        Returns
        --------
        List of str
        """
        return [x.get_label() for x in self.get_params(param_type)]

    def get_available_params(self):
        """Returns dict containing available params of all types

        This method collect result of `get_params` method for each param type into
        a dict. Keys are param types and values are lists of Param objects.

        Returns
        --------
        dict
        """
        result = {}
        for param_type in ['model', 'observation', 'variable', 'dataset']:
            result[param_type] = self.get_params(param_type)
        return result

    def list_available_params(self):
        """Returns dict containing labels of available params of all types

        This methods collect result of `list_params` for each param type into
        a dict. Keys are param types and values are list of labels.

        Returns
        --------
        dict
        """
        result = {}
        for param_type in ['model', 'observation', 'variable', 'dataset']:
            result[param_type] = self.list_params(param_type)
        return result
    
    def find_param_value(self, param_type, param_label):
        """Searches for Param object with specified label

        Parameters
        -----------
        param_type : str
            One of ['model', 'variable', 'observation', 'dataset'].
        param_label : str
            Label of searched param

        Returns
        --------
        Param or None
        """
        if param_label is None or not isinstance(param_label, str):
            return None
        return next((x for x in self.get_params(param_type) if x.get_label() == param_label), None)

    def print_options(self, plot_type=None):
        """Prints available options for plots

        Parameters
        -----------
        plot_type : str or None
            When not None, then only options for plots with this plot_type will
            be printed.
        """

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
        """Returns value of specified option

        Parameters
        -----------
        plot_type : str
           Type of plot, the option is corresponding to.
        option : str
            Name of the option

        Returns
        --------
        None or value of option
        """
        options = self.options.get(plot_type)
        if options is None:
            raise Exception('Invalid plot_type')
        if not option in options.keys():
            return
        with self.mutex:
            result = self.options.get(plot_type).get(option)
        return result

    def set_option(self, plot_type, option, value):
        """Sets value for the plot option

        Parameters
        -----------
        plot_type : str
            When None, then value will be set for each plot with
            option of name from `option` argument. Otherwise only
            for plots with specified type.
        option : str
            Name of the option
        value : *
            Value to be set
        """
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

    def get_plot(self, plot_type, params_values, cache=True):
        """Returns plot for specified type and params

        Function serches for plot in cache, when not present creates
        requested plot and put it to cache.

        Parameters
        -----------
        plot_type : str
            Type of plot to be generated
        params_values : dict
            Dict for param types as keys and Param objects as values
        cache : bool
            If serach for plot in cache and put calculated plot into cache.

        Returns
        --------
        PlotContainer
        """
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
        result = self.find_in_cache(plot_type, required_params_labels) if cache else None
        if result is None:
            result = plot_class(self).fit(required_params_values)
            if cache:
                self.put_to_cache(result)
        return result

    def get_params_attributes(self, param_type=None):
        """Returns attributes for all params

        When `param_type` is not None, then function returns list of dicts. Each dict represents
        one of available attribute for specified param type. Field `name` is attribute name
        and field `values` is mapped list of available params to list of value of attribute.
        When `param_type` is None, then function returns dict with keys for each param type and
        values are lists described above.

        Parameters
        -----------
        param_type : str or None
            One of ['model', 'variable', 'observation', 'dataset'] or None. Specifies
            attributes of which params should be returned.

        Returns
        --------
        dict or list
        """

        if not self.enable_attributes:
            return {}
        if param_type is None:
            obj = {}
            for p in ['model', 'observation', 'variable', 'dataset']:
                obj[p] = self.get_params_attributes(p)
            return obj
        attrs = Param.get_param_class(param_type).list_attributes(self)
        array = []
        for attr in attrs:
            array.append({
                'name': attr,
                'values': [param.get_attributes().get(attr) for param in self.get_params(param_type)]
            })
        return array

    def get_param_attributes(self, param_type, param_label):
        """Returns attributes for one param

        Function searches for param with specified type and label and
        returns it's attributes.

        Parameters
        -----------
        param_type : str
            One of ['model', 'variable', 'observation', 'dataset'].
        param_label : str
            Label of param

        Returns
        --------
        dict
        """

        if not self.enable_attributes:
            return {}
        param_value = self.find_param_value(param_type=param_type, param_label=param_label)
        if param_value:
            return param_value.get_attributes()
        else:
            return {}
