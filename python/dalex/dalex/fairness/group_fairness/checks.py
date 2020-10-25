import numpy as np
from ..._explainer.helper import verbose_cat
from ..basics.checks import check_parameters
from ..basics.exceptions import ParameterCheckError


def check_cutoff(protected, cutoff, verbose):
    if isinstance(cutoff, dict):
        for value in cutoff.values():
            if not isinstance(value, float):
                raise ParameterCheckError("Cutoff values must be floats")
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
            verbose_cat("Adding default(0.5) cutoffs for subgroups not provided in cutoff keys", verbose)
            for subgroup in protected_set - cutoff_set:
                cutoff[subgroup] = 0.5



    elif isinstance(cutoff, float):
        if not 0 < cutoff < 1:
            raise ParameterCheckError("if cutoff is a float it should be in [0, 1] range")
        verbose_cat("Converting cutoff to dict", verbose)
        subgroups = np.unique(protected)
        value = cutoff
        cutoff = {}
        for subgroup in subgroups:
            cutoff[subgroup] = value

    else:
        raise ParameterCheckError(f"cuttof must be a float or a dict, not {type(cutoff)}")

    return cutoff


def check_epsilon(epsilon):
    if not isinstance(epsilon, float):
        raise ParameterCheckError("epsilon must be float")

    if not 0 < epsilon < 1:
        raise ParameterCheckError("epsilon must be in (0,1) range")

    return epsilon

