import numpy as np
import pandas as pd
from ..._explainer.helper import verbose_cat


def check_parameters(y, y_hat, protected, privileged, cutoff, verbose):
    if not len(y) == len(y_hat) == len(protected):
        raise ParameterCheckError("protected and explainer attributes y, y_hat must have the same lengths")

    if list(np.unique(y)) != [0, 1]:
        raise ParameterCheckError("explainer must predict binary target")

    if not 0 <= y_hat <= 1:
        raise ParameterCheckError("y_hat must have probabilistic output between 0 and 1")

    if isinstance(protected, list):
        try:
            verbose_cat("protected list will be converted to nd.array")
            protected = np.array(protected)
        except:
            ParameterCheckError("failed to convert list to nd.array")
    elif isinstance(protected, pd.Series):

        try:
            verbose_cat("protected Series will be converted to nd.array")
            protected = np.array(protected)
        except:
            ParameterCheckError("failed to convert list to nd.array")

    if protected.dtype.type is not np.string_:
        verbose_cat("protected array is not string type, converting to string ", verbose)
        try:
            protected = protected.astype(str)
        except:
            ParameterCheckError('Could not convert protected to String type')

    if not isinstance(privileged, str):
        raise ParameterCheckError('privileged parameter must be a String')

    if privileged not in protected:
        raise ParameterCheckError("privileged parameter must be in protected vector")

    if isinstance(cutoff, float):
        if not 0 < cutoff < 1:
            raise ParameterCheckError("if cutoff is a float it should be in [0, 1] range")
        verbose_cat("Converting cutoff to dict")

        subgroups = np.unique(protected)
        value = cutoff
        cutoff = {}
        for subgroup in subgroups:
            cutoff[subgroup] = value

    return y, y_hat, protected, privileged, cutoff


def check_epsilon(epsilon):
    if not isinstance(epsilon, float):
        raise ParameterCheckError("epsilon must be float")

    if not 0 < epsilon < 1:
        raise ParameterCheckError("epsilon must be in (0,1) range")

    return epsilon


class ParameterCheckError(Exception):

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):

        if self.message:
            return f'Parameter Check Error, {self.message}'
        else:
            return 'Parameter Check Error has been raised'
