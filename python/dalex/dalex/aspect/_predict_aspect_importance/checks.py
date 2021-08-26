import numpy as np
import pandas as pd
from copy import deepcopy
from warnings import warn


def check_columns_in_new_observation(new_observation, explainer):
    if not set(new_observation.columns).issubset(explainer.data):
        raise ValueError(
            "Columns in new observation does not match these in training dataset."
        )


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
        elif new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_ = pd.DataFrame(new_observation_)
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, list):
        new_observation_ = pd.DataFrame(new_observation_).T
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, pd.DataFrame):
        if new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_.columns = explainer.data.columns
    else:
        raise TypeError(
            "new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame"
        )

    if pd.api.types.is_bool_dtype(new_observation_.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation_


def check_variable_groups(variable_groups, explainer):
    variable_groups_copy = deepcopy(variable_groups)
    if variable_groups_copy is not None:
        if not isinstance(variable_groups_copy, dict):
            raise TypeError("'variable_groups' should be of class dict")

        wrong_names = np.empty(len(variable_groups_copy))
        for i, key in enumerate(variable_groups_copy):
            if isinstance(variable_groups_copy[key], list):
                variable_groups_copy[key] = np.array(variable_groups_copy[key])
            elif not isinstance(variable_groups_copy[key], np.ndarray):
                raise TypeError(
                    "'variable_groups' is a dict of lists of variables"
                )

            if not isinstance(variable_groups_copy[key][0], str):
                raise TypeError(
                    "'variable_groups' is a dict of lists of variables"
                )

            wrong_names[i] = np.in1d(variable_groups_copy[key], explainer.data.columns).all()

        wrong_names = not wrong_names.all()

        if wrong_names:
            raise ValueError(
                "You have passed wrong variables names in variable_groups argument. "
                "'variable_groups' is a dict of lists of numeric variables."
            )

    return variable_groups_copy


def check_random_state(random_state):
    if random_state is not None:
        np.random.seed(random_state)

    return random_state


def check_method_type(type, types, aliases=None):
    if isinstance(type, tuple):
        ret = type[0]
    elif isinstance(type, str):
        ret = type
    else:
        raise TypeError("'type' is not a string")

    if ret not in types:
        if aliases:
            if ret not in aliases:
                raise ValueError("'type' must be one of: {}".format(', '.join(types+tuple(aliases))))
            else:
                return aliases[ret]
        else:
            if ret not in types:
                raise ValueError("'type' must be one of: {}".format(', '.join(types)))
    else:
        return ret


def check_processes(processes):
    from multiprocessing import cpu_count
    if processes > cpu_count():
        warn("You have asked for too many processes. Truncated to the number of physical CPUs.")
        return cpu_count()
    else:
        return processes

def check_method_depend(depend_method, corr_method, agg_method):
    depend_method_types = ('assoc', 'pps')
    depend_method_aliases = {'association': 'assoc', "PPS": 'pps', 'stats': 'assoc'}
    corr_method_types = ('spearman', 'pearson', 'kendall')
    agg_method_types = ('max', 'min', 'mean')
    agg_method_aliases = {'maximum': 'max', 'minimum': 'min', 'avg': 'mean', 'average': 'mean'}
    if isinstance(depend_method, str):
        if depend_method not in depend_method_types:
            if depend_method not in depend_method_aliases:
                raise ValueError("'depend_method' must be one of: {}".format(', '.join(depend_method_types+tuple(depend_method_aliases))))
            else:
                depend_method = depend_method_aliases[depend_method]
        if depend_method == "assoc":
            if corr_method not in corr_method_types:
                raise ValueError("'corr_method' must be one of: {}".format(', '.join(corr_method_types)))
        if depend_method == "pps":
            if agg_method not in agg_method_types:
                if agg_method not in agg_method_aliases:
                    raise ValueError("'agg_method' must be one of: {}".format(', '.join(agg_method_types)))
                else: 
                    agg_method = agg_method_aliases[agg_method]
    return depend_method, corr_method, agg_method


