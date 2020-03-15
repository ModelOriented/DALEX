import re

import numpy as np


def verbose_cat(text, verbose):
    if verbose:
        print(text)


def is_y_in_data(data, y):
    return (np.apply_along_axis(lambda x, y: (x == y).all(), 0, data, y)).any()


def yhat(model):
    if hasattr(model, 'predict_proba'):
        # check if model has predict_proba
        return lambda m, d: m.predict_proba(d)[:, 1]
    elif hasattr(model, 'predict'):
        # check if model has predict
        return lambda m, d: m.predict(d)

    return False


def get_model_info(model):
    model_package = re.search("(?<=<class ').*?(?=\.)", str(type(model)))[0]
    return {'model_package': model_package}
