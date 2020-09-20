import abc
from .exceptions import ParameterCheckError
from .checks import check_parameters


class _AbsObject(metaclass=abc.ABCMeta):

    def fairness_check(self):
        pass


class _FairnessObject(_AbsObject):

    def __init__(self, y, y_hat, protected, privileged, verbose):
        y, y_hat, protected, privileged, cutoff = check_parameters(y, y_hat, protected, privileged, verbose)

        self.cutoff = cutoff
        self.privileged = privileged
        self.protected = protected
        self.y_hat = y_hat
        self.y = y

    def fairness_check(self):
        pass
