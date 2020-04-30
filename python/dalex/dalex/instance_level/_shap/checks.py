import numpy as np
import pandas as pd


def check_columns_in_new_observation(new_observation,
                                     explainer):
    if not set(new_observation.columns).issubset(explainer.data):
        raise ValueError("Columns in new observation does not match these in training dataset.")


def check_new_observation(new_observation, explainer):
    if isinstance(new_observation, (pd.Series, list, np.ndarray)):
        new_observation = np.array(new_observation)
        if new_observation.ndim == 1:
            # make 1D array 2D
            new_observation = new_observation.reshape((1, -1))
        elif new_observation.ndim > 2:
            raise ValueError("Wrong new_observation dimension")
        elif new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation = pd.DataFrame(new_observation, columns=explainer.data.columns)
    elif isinstance(new_observation, pd.DataFrame):
        if new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation.columns = explainer.data.columns
    else:
        raise TypeError("new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame")

    if pd.api.types.is_bool_dtype(new_observation.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation.copy()
