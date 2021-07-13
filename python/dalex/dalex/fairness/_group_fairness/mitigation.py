import copy

import numpy as np

import dalex.datasets
from . import checks
from .._basics import checks as basic_checks
from .._basics.exceptions import ParameterCheckError
from ..._explainer import helper


def reweight(protected, y, verbose = True):

    y = basic_checks.check_y(y, verbose)
    protected = basic_checks.check_protected(protected, verbose)

    if not len(y) == len(protected):
        raise ParameterCheckError("protected and target (y) must have the same length")

    weights = np.repeat(None, len(y))

    for subgroup in np.unique(protected):
        for c in np.unique(y):

            Xs = np.sum(protected == subgroup)
            Xc = np.sum(y == c)
            Xsc = np.sum((protected == subgroup) & (c == y))
            Wsc = (Xs * Xc) / (len(y) * Xsc)

            weights[(protected == subgroup) & (y == c)] = Wsc

    return weights

def roc_pivot(explainer, protected, privileged, cutoff = 0.5, theta = 0.05, verbose = True):

    if not isinstance(explainer, dalex.Explainer):
        raise ParameterCheckError("explainer must be of type 'Explainer")

    # same checking as in epsilon
    theta = checks.check_epsilon(theta, 'theta')
    cutoff = checks.check_epsilon(cutoff, 'cutoff')

    protected = basic_checks.check_protected(protected, verbose)
    privileged = basic_checks.check_privileged(privileged, protected, verbose)

    exp = copy.deepcopy(explainer)
    probs = exp.y_hat

    if not len(probs) == len(protected):
        raise ParameterCheckError("protected and target (y) must have the same length")

    is_close = np.abs(probs - cutoff) < theta
    is_privileged = privileged == protected
    is_favorable = probs > cutoff

    probs[is_close & is_privileged & is_favorable] = cutoff - (probs[is_close & is_privileged & is_favorable] - cutoff)
    probs[is_close & np.logical_not(is_privileged) & np.logical_not(is_favorable)] = cutoff + (cutoff - probs[is_close & np.logical_not(is_privileged) & np.logical_not(is_favorable)])

    probs[probs < 0] = 0
    probs[probs > 1] = 1

    exp.y_hat = probs

    return exp

def resample(protected, y, type = 'uniform', probs = None,  verbose = True):

    if type == 'preferential' and probs is None:
        raise ParameterCheckError("when using type 'preferential' probabilities (probs) must be provided")

    if type not in set(['uniform', 'preferential']):
        raise ParameterCheckError("type must be either 'uniform' or 'preferential'")


    protected = basic_checks.check_protected(protected, verbose)
    y = basic_checks.check_y(y, verbose)

    if type == 'preferential':
        try:
            probs = np.asarray(probs)
            helper.verbose_cat("converted 'probs' to numpy array", verbose=verbose)
        except Exception:
            raise ParameterCheckError("try converting 'probs' to 1D numpy array")

        if probs.ndim != 1 or len(probs) != len(y):
            raise ParameterCheckError("probs parameter must 1D numpy array with the same length as y")


    weights = reweight(protected, y, verbose=False)

    expected_size =  dict.fromkeys(np.unique(protected))
    for key in expected_size.keys():
        expected_size[key] = dict.fromkeys(np.unique(y))

    for subgroup in expected_size.keys():
        for value in np.unique(y):
            case_weights = weights[(subgroup == protected) & (value == y)]
            case_size = len(case_weights)
            weight = case_weights[0]
            expected_size[subgroup][value] = round(case_size * weight)

    indices = []

    for subgroup in expected_size.keys():
        for value in np.unique(y):
            current_case = np.arange(len(y))[(protected == subgroup) & (y == value)]
            expected = expected_size[subgroup][value]
            actual = np.sum((protected == subgroup) & (y == value))
            if expected == actual:
                 indices += list(current_case)

            elif expected < actual:
                if type == 'uniform':
                    indices += list(np.random.choice(current_case, expected, replace=False))
                else:
                    sorted_current_case = current_case[np.argsort(probs[current_case])]
                    if value == 0:
                        indices += list(sorted_current_case[:expected])
                    if value == 1:
                        indices += list(sorted_current_case[-expected:])
            else:
                if type == 'uniform':
                    u_ind = list(np.repeat(current_case, expected // actual))
                    u_ind += list(np.random.choice(current_case, expected % actual))

                    indices += u_ind

                else:
                    sorted_current_case = current_case[np.argsort(probs[current_case])]
                    p_ind = list(np.repeat(current_case, expected // actual))

                    if expected % actual != 0:
                        if value == 0:
                            p_ind += list(sorted_current_case[-(expected % actual):])
                        if value == 1:
                            p_ind += list(sorted_current_case[:(expected % actual)])

                    indices += p_ind

    return indices
