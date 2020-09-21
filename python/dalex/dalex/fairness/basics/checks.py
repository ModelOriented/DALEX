from .exceptions import *
from ..._explainer.helper import verbose_cat
import numpy as np
import pandas as pd


def check_parameters(y, y_hat, protected, privileged, verbose):
    # @TODO crashes when pd.series is provided instead of np.array in protected
    if not len(y) == len(y_hat) == len(protected):
        raise ParameterCheckError("protected and explainer attributes y, y_hat must have the same lengths")

    if not isinstance(verbose, bool):
        raise ParameterCheckError("verbose must be boolean, either False or True")

    if list(np.unique(y)) != [0, 1]:
        raise ParameterCheckError("explainer must predict binary target")

    if not 0 <= y_hat.all() <= 1:
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
            protected = protected.to_numpy()
        except:
            ParameterCheckError("failed to convert list to nd.array")

    else:
        ParameterCheckError("Please convert protected to flat np.ndarray")

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

    return y, y_hat, protected, privileged
