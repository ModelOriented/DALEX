import numpy as np
import pandas as pd


def check_variables(variables):
    # treating variables as list simplifies code
    if variables is not None and not isinstance(variables, (str, list, np.ndarray, pd.Series)):
        raise TypeError("variables must be None or str or list or np.ndarray or pd.Series")

    if variables is None:
        variables_ = None
    elif isinstance(variables, str):
        variables_ = [variables]
    else:
        variables_ = list(variables)

    return variables_


def check_variable_type(variable_type):
    if variable_type not in ['numerical', 'categorical']:
        raise ValueError("variable_type needs to be 'numerical' or 'categorical'")



def check_groups(groups):
    # treating groups as list simplifies code
    if groups is not None and not isinstance(groups, (str, list, np.ndarray, pd.Series)):
        raise TypeError("groups must be str or list or numpy.ndarray or pandas.Series or None")

    if groups is None:
        groups_ = []
    elif isinstance(groups, str):
        groups_ = [groups]
    else:
        groups_ = list(groups)

    return groups_
