import numpy as np


def loss_root_mean_square(observed, predicted):
    return np.sqrt(((observed - predicted) ** 2).mean())
