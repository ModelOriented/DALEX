# helper functions for checks
import re
import numpy as np

from .._dataframe import iter_column_arrays


def verbose_cat(text, verbose):
    if verbose:
        print(text)


def is_y_in_data(data, y):
    for column in iter_column_arrays(data, numeric_only=y.dtype.kind in "biufc"):
        try:
            if np.array_equal(column, y):
                return True
        except (TypeError, ValueError):
            # Nested columns are valid model inputs, but cannot equal a numeric
            # scalar target vector. This optional leakage diagnostic must not
            # reject them or force a full object matrix conversion.
            continue
    return False


def get_model_info(model):
    model_package = re.search(r"(?<=<class ').*?(?=\.)", str(type(model)))[0]
    return {'model_package': model_package}
