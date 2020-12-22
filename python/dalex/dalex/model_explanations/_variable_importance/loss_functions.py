import numpy as np
from .._model_performance.utils import auc


def loss_root_mean_square(observed, predicted):
    return np.sqrt(((observed - predicted) ** 2).mean())


def loss_one_minus_auc(observed, predicted):
    return 1 - auc(y_true=observed, y_pred=predicted)
