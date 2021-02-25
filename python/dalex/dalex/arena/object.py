import threading
import time
import numpy as np
import webbrowser
from datetime import datetime
from dalex import Explainer
from pandas.core.frame import DataFrame
from .server import start_server
from ._plot_container import PlotContainer
from .params import ModelParam, DatasetParam, VariableParam, ObservationParam, Param
from .._global_checks import global_check_import
from .static import get_json, upload_arena, generate_token
from ._resource_manager import ResourceManager
from ._plots_manager import PlotsManager

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
    mutex : _thread.lock
        Mutex for params, plots and resources cache. Common to Arena, PlotsManager and ResourcesManager class.
    options : dict
        Options for plots
    resource_manager: ResourceManager
        Object responsible for managing resources
    plots_manager: PlotsManager
        Object responsible for managing plots

    Notes
    --------
    For tutorial look at https://arena.drwhy.ai/docs/guide/first-datasource

    """
    def __init__(self, precalculate=False, enable_attributes=True, enable_custom_params=True):
        self.mutex = threading.Lock()
        self.models = []
        self.observations = []
        self.datasets = []
        self.variables_cache = []
        self.resource_manager = ResourceManager(self)
        self.plots_manager = PlotsManager(self)
        self.server_thread = None
        self.precalculate = bool(precalculate)
        self.enable_attributes = bool(enable_attributes)
        self.enable_custom_params = bool(enable_custom_params)
        self.timestamp = datetime.timestamp(datetime.now())
        self.options = {}
        for x in (self.plots_manager.plots + self.resource_manager.resources):
            options = self.options.get(x.options_category) or {}
            for o in x.options.keys():
                options[o] = {'value': x.options.get(o).get('default'), 'desc': x.options.get(o).get('desc')}
            self.options[x.options_category] = options

    def get_supported_plots(self):
        """Returns plots classes that can produce at least one valid chart for this Arena.

        Returns
        -----------
        List of classes extending PlotContainer
        """
        return self.plots_manager.get_supported_plots()

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

        Notes
        --------
        Read more about data sources https://arena.drwhy.ai/docs/guide/basic-concepts

        Returns
        -----------
        Link to Arena
        """
        if self.server_thread:
            raise Exception('Server is already running. To stop ip use arena.stop_server().')
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
            self.plots_manager.fill_cache({'model': param})

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
                self.plots_manager.fill_cache({'observation': obs})

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
            self.plots_manager.fill_cache({'dataset': param})

    def get_params(self, param_type):
        """Returns list of available params

        Parameters
        -----------
        param_type : str
            One of ['model', 'variable', 'observation', 'dataset']. Params of this type
            will be returned

        Notes
        --------
        Information about params https://arena.drwhy.ai/docs/guide/params

        Returns
        --------
        List of Param objects
        """
        if param_type == 'observation':
            with self.mutex:
                return self.observations
        elif param_type == 'variable':
            with self.mutex:
                if not self.variables_cache:
                    # Extract column names from every dataset in self.dataset list and flatten it
                    result_datasets = [col for dataset in self.datasets for col in dataset.variables]
                    # Extract column names from every model in self.models list and flatten it
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
                return self.variables_cache
        elif param_type == 'model':
            with self.mutex:
                return self.models
        elif param_type == 'dataset':
            with self.mutex:
                return self.datasets
        else:
            raise Exception('Invalid param type')

    def list_params(self, param_type):
        """Returns list of available params's labels

        Parameters
        -----------
        param_type : str
            One of ['model', 'variable', 'observation', 'dataset']. Labels of params
            of this type will be returned

        Notes
        --------
        Information about params https://arena.drwhy.ai/docs/guide/params

        Returns
        --------
        List of str
        """
        return [x.get_label() for x in self.get_params(param_type)]

    def get_available_params(self):
        """Returns dict containing available params of all types

        This method collect result of `get_params` method for each param type into
        a dict. Keys are param types and values are lists of Param objects.

        Notes
        --------
        Information about params https://arena.drwhy.ai/docs/guide/params

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

        Notes
        --------
        Information about params https://arena.drwhy.ai/docs/guide/params

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

        Notes
        --------
        Information about params https://arena.drwhy.ai/docs/guide/params

        Returns
        --------
        Param or None
        """
        if param_label is None or not isinstance(param_label, str):
            return None
        return next((x for x in self.get_params(param_type) if x.get_label() == param_label), None)

    def print_options(self, options_category=None):
        """Prints available options for plots

        Parameters
        -----------
        options_category : str or None
            When not None, then only options for plots or resources with this
            category will be printed.

        Notes
        --------
        List of plots with described options for each one https://arena.drwhy.ai/docs/guide/observation-level
        """

        options = self.options.get(options_category)
        if options is None:
            for category in self.options.keys():
                self.print_options(category)
            return
        if len(options.keys()) == 0:
            return
        print('\n\033[1m' + options_category + '\033[0m')
        print('---------------------------------')
        for option_name in options.keys():
            value = options.get(option_name).get('value')
            print(option_name + ': ' + str(value) + '   #' + options.get(option_name).get('desc'))

    def get_option(self, options_category, option):
        """Returns value of specified option

        Parameters
        -----------
        options_category : str
           Category of option. In most cases category is coresponds to one plot_type.
           Categories are underlined in the output of arena.print_options()
        option : str
            Name of the option

        Notes
        --------
        List of plots with described options for each one https://arena.drwhy.ai/docs/guide/observation-level

        Returns
        --------
        None or value of option
        """
        options = self.options.get(options_category)
        if options is None:
            raise Exception('Invalid options category')
        if option not in options.keys():
            return
        with self.mutex:
            return self.options.get(options_category).get(option).get('value')

    def set_option(self, options_category, option, value):
        """Sets value for the plot option

        Parameters
        -----------
        options_category : str or None
            When None, then value will be set for each plot and resource
            having option with name equal to `option` argument. Otherwise only
            for plots and resources with specified options_category.
            In most cases category is coresponds to one plot_type.
            Categories are underlined in the output of arena.print_options()
        option : str
            Name of the option
        value : *
            Value to be set

        Notes
        --------
        List of plots with described options for each one https://arena.drwhy.ai/docs/guide/observation-level
        """
        if options_category is None:
            for category in self.options.keys():
                self.set_option(category, option, value)
            return
        options = self.options.get(options_category)
        if options is None:
            raise Exception('Invalid options category')
        if option not in options.keys():
            return
        with self.mutex:
            self.options[options_category][option]['value'] = value
            for plot in [x for x in self.plots_manager.plots if x.options_category == options_category]:
                self.plots_manager.clear_cache(plot.info.get('plotType'))
            for resource in [x for x in self.resource_manager.resources if x.options_category == options_category]:
                self.resource_manager.clear_cache(resource.resource_type)
        if self.precalculate:
            self.plots_manager.fill_cache()

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

        Notes
        --------
        Attribused are used for dynamicly modifying observations https://arena.drwhy.ai/docs/guide/modifying-observations

        Returns
        --------
        dict or list
        """

        if param_type is None:
            obj = {}
            for p in ['model', 'observation', 'variable', 'dataset']:
                obj[p] = self.get_params_attributes(p)
            return obj
        if not self.enable_attributes:
            return []
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

        Notes
        --------
        Attribused are used for dynamicly modifying observations https://arena.drwhy.ai/docs/guide/modifying-observations

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

    def save(self, filename="datasource.json"):
        """Generate all plots and saves them to JSON file

        Function generates only not cached plots.

        Parameters
        -----------
        filename : str
            Path or filename to output file

        Notes
        --------
        Read more about data sources https://arena.drwhy.ai/docs/guide/basic-concepts

        Returns
        --------
        None
        """
        with open(filename, 'w') as file:
            file.write(get_json(self))

    def upload(self, token=None, arena_url='https://arena.drwhy.ai/', open_browser=True):
        """Generate all plots and uploads them to GitHub Gist

        Function generates only not cached plots. If token is not provided
        then function uses OAuth to open GitHub authorization page.

        Parameters
        -----------
        token : str or None
            GitHub personal access token. If token is None, then OAuth is used.
        arena_url : str
            Address of Arena dashboard instance
        open_browser : bool
            Whether to open Arena after upload.

        Notes
        --------
        Read more about data sources https://arena.drwhy.ai/docs/guide/basic-concepts

        Returns
        --------
        Link to the Arena
        """
        global_check_import('requests')
        if token is None:
            global_check_import('flask')
            global_check_import('flask_cors')
            token = generate_token()
        data_url = upload_arena(self, token)
        url = arena_url + '?data=' + data_url
        if open_browser:
            webbrowser.open(url)
        return url
