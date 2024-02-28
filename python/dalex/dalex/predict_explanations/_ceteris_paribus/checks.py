import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from . import utils


def check_variables(variables, explainer, variable_splits):
    if isinstance(variables, str):
        variables = [variables]
    variables_ = deepcopy(variables)
    if variable_splits is not None:
        variables_ = variable_splits.keys()
        warnings.warn("Parameter `variable_splits` overrides `variables`. Variables taken from `variables_splits.keys()`.")
    elif variables_ is not None and isinstance(variables_, (list, np.ndarray, pd.Series)):
        if not set(variables_).issubset(explainer.data.columns):
            raise ValueError('Invalid variable names')
    elif variables_ is not None and not isinstance(variables_, (list, np.ndarray, pd.Series)):
        raise TypeError("variables must be None or str or list or np.ndarray or pd.Series")
    else:
        variables_ = explainer.data.columns

    return list(variables_)


def check_new_observation(new_observation, explainer):
    new_observation_ = deepcopy(new_observation)
    if isinstance(new_observation_, pd.Series):
        new_observation_ = new_observation_.to_frame().T
        new_observation_.columns = explainer.data.columns
    elif isinstance(new_observation_, np.ndarray):
        if new_observation_.ndim == 1:
            # make 1D array 2D
            new_observation_ = new_observation_.reshape((1, -1))
        elif new_observation_.ndim > 2:
            raise ValueError("Wrong new_observation dimension")

        new_observation_ = pd.DataFrame(new_observation_)
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, list):
        new_observation_ = pd.DataFrame(new_observation_).T
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, pd.DataFrame):
        new_observation_.columns = explainer.data.columns
    else:
        raise TypeError("new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame")

    if new_observation_.index.unique().shape[0] != new_observation_.shape[0]:
        raise ValueError("new_observation index is not unique")

    if pd.api.types.is_bool_dtype(new_observation_.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation_


def check_data(data, variables):
    if data is None:
        raise ValueError("Data must be provided by explainer")

    return data


def check_y(y):
    if y is not None and isinstance(y, pd.Series):
        y = np.array(y)
    elif y is not None and not isinstance(y, np.ndarray):
        raise TypeError("y must be numpy.ndarray or pandas.Series")

    return y


def check_variable_splits(variable_splits,
                          variables,
                          grid_points,
                          data,
                          variable_splits_type,
                          variable_splits_with_obs,
                          new_observation):
    """
    Validate variable splits
    """
    if variable_splits is None:
        variable_splits_ = utils.calculate_variable_split(
            data,
            variables,
            grid_points,
            variable_splits_type,
            variable_splits_with_obs,
            new_observation
        )
    else:
        if not isinstance(variable_splits, dict):
            raise TypeError("variable_splits has to be a dict")

        if not set(variable_splits.keys()) == set(variables):
            raise ValueError("variable_splits variables do not match with these in explainer data")

        variable_splits_ = deepcopy(variable_splits)

        for key in variable_splits_.keys():
            if not isinstance(variable_splits_[key], (list, np.ndarray)):
                raise TypeError("variable_splits values have to be list or numpy.ndarrays")
            if isinstance(variable_splits_[key], list):
                variable_splits_[key] = np.array(variable_splits_[key])

    return variable_splits_


def check_processes(processes):
    from multiprocessing import cpu_count
    if processes > cpu_count():
        warnings.warn("You have asked for too many processes. Truncated to the number of physical CPUs.")

        return cpu_count()

    else:
        return processes


def check_variable_splits_type(arg):
    if arg not in ('uniform', 'quantile', 'quantiles'):
        raise ValueError("variable_splits_type has to be one of {'uniform', 'quantiles'}")
    return arg
