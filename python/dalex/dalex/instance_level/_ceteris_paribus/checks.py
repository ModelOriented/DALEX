import numpy as np
import pandas as pd


def check_variables(variables, explainer):
    if variables is not None and isinstance(variables, (list, np.ndarray, pd.Series)):
        if not set(variables).issubset(explainer.data.columns):
            raise ValueError('Invalid variable names')
    elif variables is not None and not isinstance(variables, (list, np.ndarray, pd.Series)):
        raise TypeError("Variables must by list or numpy.ndarray or pd.Series")
    else:
        variables = explainer.data.columns

    return list(variables)


def check_new_observation(new_observation, explainer):
    if isinstance(new_observation, (pd.Series, list, np.ndarray)):
        new_observation = np.array(new_observation)
        if new_observation.ndim == 1:
            # make 1D array 2D
            new_observation = new_observation.reshape((1, -1))
        elif new_observation.ndim > 2:
            raise ValueError("Wrong new_observation dimension")

        new_observation = pd.DataFrame(new_observation, columns=explainer.data.columns)
    elif isinstance(new_observation, pd.DataFrame):
        new_observation.columns = explainer.data.columns
    else:
        raise TypeError("new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame")

    if new_observation.index.unique().shape[0] != new_observation.shape[0]:
        raise ValueError("new_observation index is not unique")

    if pd.api.types.is_bool_dtype(new_observation.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation.copy()


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


def check_variable_splits(variable_splits, variables):
    """
    Validate variable splits
    """
    if variable_splits is not None and set(variable_splits.keys()) == set(variables):
        return True
    else:
        return False
