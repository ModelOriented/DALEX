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
    if not isinstance(new_observation, (pd.DataFrame,)):
        raise TypeError("new_observation must be the pandas.DataFrame")

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
