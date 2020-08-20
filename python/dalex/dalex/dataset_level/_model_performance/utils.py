import numpy as np
import pandas as pd


def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def mae(y_pred, y_true):
    return np.abs(y_pred - y_true).mean()


def mad(y_pred, y_true):
    return np.median(np.abs(y_pred - y_true))


def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))


def r2(y_pred, y_true):
    return 1 - mse(y_pred, y_true) / mse(y_true.mean(), y_true)


def auc(y_pred, y_true):
    df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})
    df = df.sort_values('y_pred', ascending=False)

    # assumes that y = 0/1 where 1 is the positive label
    tpr = df.loc[:, 'y_true'].cumsum() / df.loc[:, 'y_true'].sum()
    fpr = (1 - df.loc[:, 'y_true']).cumsum() / (1 - df.loc[:, 'y_true']).sum()

    _auc = (np.diff(fpr) * (tpr.iloc[1:].values + tpr.iloc[:-1].values) / 2).sum()

    return _auc


def recall(tp, fp, tn, fn):
    return tp / (tp + fn)


def precision(tp, fp, tn, fn):
    return tp / (tp + fp)


def f1(tp, fp, tn, fn):
    _recall = recall(tp, fp, tn, fn)
    _precision = precision(tp, fp, tn, fn)
    return 2 * (_precision * _recall) / (_precision + _recall)


def accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + fp + tn + fn)
