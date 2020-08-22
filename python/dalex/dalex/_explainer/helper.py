# helper functions for checks
import re
import numpy as np


def verbose_cat(text, verbose):
    if verbose:
        print(text)


def is_y_in_data(data, y):
    return (np.apply_along_axis(lambda x, y: (x == y).all(), 0, data, y)).any()


def get_model_info(model):
    model_package = re.search("(?<=<class ').*?(?=\.)", str(type(model)))[0]
    return {'model_package': model_package}
