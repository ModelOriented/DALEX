import threading
from .params import Param
from ._option_base import OptionBase


class Resource(OptionBase):
    """
    """
    def __init__(self, arena):
        super().__init__(arena)
        self.params = {}
        self.mutex = threading.Lock()
        self.update_event = threading.Event()
        # Protected by mutex:
        self.exception = None
        self.is_done = False  # to prevent deadlock it is important that thread cannot use mutex after setting this to true
        self.progress = -1
        self.data = {}
        self.thread = None
    def fit(self, params):
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
        with self.mutex:
            if self.exception is not None:
                raise self.exception
            if self.is_done and self.thread is not None:
                self.thread.join()
                self.thread = None
            return (self.data, self.progress, self.is_done)
    def wait_for_update(self):
        self.update_event.wait()
    def _emit_update(self):
        self.update_event.set()
        self.update_event.clear()
