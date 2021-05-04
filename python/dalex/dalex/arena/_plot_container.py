from ._option_base import OptionBase
from .params import ModelParam, DatasetParam, VariableParam, ObservationParam


class PlotContainer(OptionBase):
    """
    Class representing a chart.
    
    Parameters
    ----------
    arena : dalex.Arena
        Instance of Arena.
    cache : bool
        If this object is allowed to use cache when requesting resources

    Attributes
    --------
    arena : Arena
        Instance of dalex.Arena
    name : str
        Display name of chart
    plot_type : str
        Identifier of chart type
    plot_component : str
        Identifier of Arena's component that should render this chart
    plot_category : str
        Name of category of chart
    params : dict
        Dictionary with required param types as keys and param labels as values.
        This attribute is set when calling fit.
    data : dict
        Results of computations are placed there
    progress : float
        If progress is supprted, then value should be between [0,1]. For other situations -1 value must be set.
        Progress of plot container is based of progress of used resources at the moment of calling fit method.
        This value will not be updated.
    use_cache : bool
        If this object is allowed to use cache when requesting resources
    """
    def __init__(self, arena, cache=True):
        super().__init__(arena)
        info = self.__class__.info
        self.name = info.get('name')
        self.plot_type = info.get('plotType')
        self.plot_component = info.get('plotType')
        self.plot_category = info.get('plotCategory')
        self.params = {}
        self.data = {}
        self.is_done = None
        self.progress = -1
        # If plot class is allowed to use cache when requesting resources
        self.use_cache = cache
    def fit(self, params):
        """Function computes plots data for given params

        Parameters
        -----------
        params : dict
            Keys of this dict are params types (model, observation, variable, dataset)
            and values are corresponding params values (class Param).

        Returns
        --------
        PlotContainer object
        """
        required_params = {}
        for p in self.__class__.info.get('requiredParams'):
            self.check_param(p, params.get(p))
        for p in self.__class__.info.get('requiredParams'):
            required_params[p] = params.get(p)
            self.params[p] = params.get(p).get_label()
        self._fit(**required_params)
        return self
    def serialize(self):
        """Saves important attributes of PlotContainer into a dict.
        Returned dict is meant to be directly put into Arena data file.

        Returns
        --------
        dict
        """
        return {
            'name': self.name,
            'plotType': self.plot_type,
            'plotComponent': self.plot_component,
            'plotCategory': self.plot_category,
            'params': self.params,
            'data': self.data,
            'progress': self.progress,
            'isDone': True if self.is_done is None else self.is_done
        }
    def set_message(self, msg, msg_type='info'):
        """Changes plot component to message and sets data with provided message

        Parameters
        -----------
        msg : str
            Text of message
        msg_type : str
            Type of message. One of ['info', 'error']
        """
        if msg_type != 'info' and msg_type != 'error':
            raise Exception('Invalid message type')
        self.plot_component = 'Message'
        self.data = {'message': msg, 'type': msg_type}

    def check_param(self, param_type, value):
        """Function validates param values as param of given type

        Parameters
        -----------
        param_type : str
            One of ['model', 'variable', 'observation', 'dataset'].
        value : Object
            Function checks if this object have correct class.

        Returns
        --------
        None if value is correct. Else raises exception
        """
        correct_class = {'model': ModelParam, 'variable': VariableParam, 'observation': ObservationParam, 'dataset': DatasetParam}.get(param_type)
        if not isinstance(value, correct_class):
            raise Exception('Invalid param ' + str(param_type))
    @staticmethod
    def test_arena(arena):
        """Tests if plot can be created for at least one combination of params

        This method searches for params, that can produce valid chart. Displaying
        error messages are not counted as valid. One example of usage are charts for
        classification models. Such charts should override this method and check if
        there is at least one classification model in arena.

        Parameters
        -----------
        arena : dalex.Arena
            Object of class dalex.Arena

        Returns
        -----------
        bool
        """
        return True
