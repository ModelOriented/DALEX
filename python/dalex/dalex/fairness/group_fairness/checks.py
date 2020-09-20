import numpy as np
from ..._explainer.helper import verbose_cat
from ..basics.checks import check_parameters
from ..basics.exceptions import ParameterCheckError


def check_classifier_parameters(y, y_hat, protected, privileged, cutoff, verbose):
    y, y_hat, protected, privileged = check_parameters(y, y_hat, protected, privileged, verbose)

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

