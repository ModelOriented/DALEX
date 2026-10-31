import abc
from . import checks
from ..._explanation import ResultTables, CompatibilityTable


class _AbsObject(metaclass=abc.ABCMeta):

    def fairness_check(self):
        pass

    def plot(self):
        pass


class _FairnessObject(ResultTables, _AbsObject):

    metric_scores = CompatibilityTable()
    parity_loss = CompatibilityTable()

    def __init__(self, y, y_hat, protected, privileged, verbose, backend=None):
        import narwhals.stable.v2 as nw
        if backend is None:
            try:
                source = nw.from_native(protected, allow_series=True, eager_only=True)
                backend = nw.get_native_namespace(source)
            except TypeError:
                backend = 'pandas'
        self._result_backend = backend if isinstance(backend, str) else backend.__name__
        from ..._dataframe import as_numpy_vector
        import numpy as np
        y = np.asarray(y) if isinstance(y, (np.ndarray, list, tuple)) else as_numpy_vector(y, 'y')
        y_hat = np.asarray(y_hat) if isinstance(y_hat, (np.ndarray, list, tuple)) else as_numpy_vector(y_hat, 'y_hat')
        y, y_hat, protected, privileged = checks.check_parameters(y, y_hat, protected, privileged, verbose)
        self.privileged = privileged
        self.protected = protected
        self.y_hat = y_hat
        self.y = y

    def fairness_check(self):
        pass

    def plot(self):
        pass
