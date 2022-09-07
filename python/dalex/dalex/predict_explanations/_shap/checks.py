from copy import deepcopy
from warnings import warn

import numpy as np
import pandas as pd


def check_columns_in_new_observation(new_observation,
                                     explainer):
    if not set(new_observation.columns).issubset(explainer.data):
        raise ValueError("Columns in new observation does not match these in training dataset.")


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
        raise TypeError("new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame")

    if pd.api.types.is_bool_dtype(new_observation_.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation_


def check_path(path):
    if not isinstance(path, (str, list, np.ndarray, pd.Series)):
        raise TypeError('path must be str or list or numpy.ndarray or pd.Series')
    if isinstance(path, str):
        if path != 'average':
            raise ValueError("if path is str, then it must be 'average'")
        else:
            return path

    path_ = np.array(path)
    if not np.issubdtype(path_[0], int):
        raise TypeError('if path is array-like, then must be array of integers')

    return path_


def check_processes(processes):
    from multiprocessing import cpu_count
    if processes > cpu_count():
        warn("You have asked for too many processes. Truncated to the number of physical CPUs.")

        return cpu_count()

    else:
        return processes


def check_random_state(random_state):
    if random_state is not None:
        np.random.seed(random_state)

    return random_state
