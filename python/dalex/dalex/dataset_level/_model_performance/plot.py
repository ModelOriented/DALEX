import numpy as np


def ecdf(x):
    # https://community.plot.ly/t/plot-the-empirical-cdf/29045
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size

    return result
