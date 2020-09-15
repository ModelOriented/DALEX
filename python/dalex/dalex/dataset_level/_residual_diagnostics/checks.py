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
