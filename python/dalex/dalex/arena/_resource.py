import threading
from .params import Param
from ._option_base import OptionBase


class Resource(OptionBase):
    """
    Class representing a resource that need to be calculated and will be used
    by one or more charts (PlotContainer).
    Computations will run in seperate thread and can support partial results.
    
    Parameters
    ----------
    arena : dalex.Arena
        Instance of Arena.

    Attributes
    --------
    arena : Arena
        Instance of dalex.Arena
    params : dict
        Dictionary with required param types as keys and param labels as values.
        This attribute is set when calling fit.
    mutex : _thread.lock
        Mutex used for data and progress attributes. See __init__ code for details.
    update_events : threading.Event
        Object used to block execution until update of results.
    cancel_signal : bool
        This variable is set to signal resource it should cancel computations.
    exception: Exception
        When exception occurs during computations, then that exception is saved to this variable
    is_done : bool
        Flag set by thread when computations are done.
    progress : float
        If progress is supprted, then value should be between [0,1]. For other situations -1 value must be set.
    data : dict
        Results of computations are placed there
    thread : threading.Thread
        Thread used for computations
    """
    def __init__(self, arena):
        super().__init__(arena)
        self.params = {}
        self.mutex = threading.Lock()
        self.update_event = threading.Event()
        # Protected by mutex:
        self.cancel_signal = False
        self.exception = None
        self.is_done = False  # to prevent deadlock it is important that thread cannot acquire mutex after setting this to true
        self.progress = -1
        self.data = {}
        self.thread = None
    def fit(self, params):
        """Function starts computations for given params

        Parameters
        -----------
        params : dict
            Keys of this dict are params types (model, observation, variable, dataset)
            and values are corresponding params values (class Param).
        Returns
        --------
        Resource object
        """
        required_params = {}
        for p in self.__class__.required_params:
            self.check_param(p, params.get(p))
        for p in self.__class__.required_params:
            required_params[p] = params.get(p)
            self.params[p] = params.get(p).get_label()
        with self.mutex:
            self.thread = threading.Thread(target=self._fit, kwargs=required_params)
            self.thread.start()
        return self
    def cancel(self):
        """
        Function stops resource computations
        """
        with self.mutex:
            self.cancel_signal = True
            self.exception = Exception('Task aborted')
    def _init_thread(self, kwargs):
        try:
            self._fit(**kwargs)
        except Exception as e:
            with self.mutex:
                self.exception = e
    def check_param(self, param_type, value):
        if not isinstance(value, Param.get_param_class(param_type)):
            raise Exception('Invalid param ' + str(param_type))
    def get_result(self):
        """Function return results with current progress.
        If computations failed, then this function will raise exception,

        --------
        Tuple (results, progress, is_done)
        """
        with self.mutex:
            if self.exception is not None:
                raise self.exception
            if self.is_done and self.thread is not None:
                self.thread.join()
                self.thread = None
            return (self.data, self.progress, self.is_done)
    def wait_for_update(self):
        """
        Function blocks until results are updated
        """
        self.update_event.wait()
    def _emit_update(self):
        self.update_event.set()
        self.update_event.clear()