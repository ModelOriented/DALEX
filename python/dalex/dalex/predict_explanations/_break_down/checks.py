from ..._dataframe import column_names
from copy import deepcopy

import numpy as np
import pandas as pd
from ..._dataframe import native_observation


def check_new_observation(new_observation, explainer):
    if not isinstance(explainer.data, pd.DataFrame):
        return native_observation(new_observation, explainer.data, single=True)
    new_observation_ = deepcopy(new_observation)
    if isinstance(new_observation_, pd.Series):
        new_observation_ = new_observation_.to_frame().T
        new_observation_.columns = column_names(explainer.data)
    elif isinstance(new_observation_, np.ndarray):
        if new_observation_.ndim == 1:
            # make 1D array 2D
            new_observation_ = new_observation_.reshape((1, -1))
        elif new_observation_.ndim > 2:
            raise ValueError("Wrong new_observation dimension")

        elif new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_ = pd.DataFrame(new_observation_)
        new_observation_.columns = column_names(explainer.data)

    elif isinstance(new_observation_, list):
        new_observation_ = pd.DataFrame(new_observation_).T
        new_observation_.columns = column_names(explainer.data)

    elif isinstance(new_observation_, pd.DataFrame):
        if new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_.columns = column_names(explainer.data)
    else:
        raise TypeError("new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame")

    if pd.api.types.is_bool_dtype(new_observation_.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation_


def check_order(order):
    if order is not None and not isinstance(order, (list, np.ndarray)):
        raise TypeError('Wrong order type')

    if order is not None:
        return np.array(order)
    else:
        return None
