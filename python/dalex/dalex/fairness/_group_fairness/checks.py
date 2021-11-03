import numpy as np

from .._basics.exceptions import ParameterCheckError
from ..._explainer import helper


def check_cutoff(protected, cutoff, verbose):
    if isinstance(cutoff, dict):
        for value in cutoff.values():
            if not isinstance(value, float):
                raise ParameterCheckError("cutoff values must be floats")
            else:
                for value in cutoff.values():
                    if not 0 < value < 1:
                        raise ParameterCheckError("cutoff values should be in [0, 1] range")

        # changing to sets
        cutoff_set = set(cutoff.keys())
        protected_set = set(protected)

        # all cutoff keys are in protected
        if cutoff_set | protected_set != protected_set:
            raise ParameterCheckError("cutoff dict contains keys not present in protected")

        # if some keys are in protected but not in cutoff, add them to cutoff with default value
        if protected_set - cutoff_set != set():
            helper.verbose_cat("Adding default (0.5) cutoffs for subgroups not provided in cutoff keys", verbose)
            for subgroup in protected_set - cutoff_set:
                cutoff[subgroup] = 0.5

    elif isinstance(cutoff, float):
        if not 0 < cutoff < 1:
            raise ParameterCheckError("if cutoff is a float it should be in [0, 1] range")
        helper.verbose_cat("Converting cutoff to dict", verbose)
        subgroups = np.unique(protected)
        value = cutoff
        cutoff = {}
        for subgroup in subgroups:
            cutoff[subgroup] = value

    else:
        raise ParameterCheckError(f"cutoff must be a float or a dict, not {type(cutoff)}")

    return cutoff


def check_classification_parameters(y, y_hat, protected, privileged, verbose):
    if list(np.unique(y)) != [0, 1]:
        raise ParameterCheckError("model must predict binary target")

    if not 0 <= y_hat.all() <= 1:
        raise ParameterCheckError("y_hat must have probabilistic output between 0 and 1")

    return


def check_epsilon(epsilon, name = 'epsilon'):
    if not isinstance(epsilon, float):
        raise ParameterCheckError(name + " must be float")

    if not 0 < epsilon < 1:
        raise ParameterCheckError(name + " must be in (0,1) range")

    return epsilon

