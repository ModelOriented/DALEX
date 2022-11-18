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
    _df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})
    if _df.y_true.nunique() == 1:
        raise ValueError("Only one class present in `y`. ROC AUC score is not defined in that case.")
    tpr_temp = _df.groupby('y_pred').sum().reset_index().sort_values('y_pred', ascending=False)
    fpr_temp = _df.assign(y_true=1-_df.y_true).groupby('y_pred').sum().reset_index().sort_values('y_pred', ascending=False)

    # assumes that y = 0/1 where 1 is the positive label
    tpr = pd.concat([pd.Series([0]), tpr_temp.y_true.cumsum()]) / _df.y_true.sum()
    fpr = pd.concat([pd.Series([0]), fpr_temp.y_true.cumsum()]) / (1 - _df.y_true).sum()

    _auc = (np.diff(fpr) * (tpr.iloc[1:].values + tpr.iloc[:-1].values) / 2).sum()

    return _auc


def recall(tp, fp, tn, fn):
    if tp + fn == 0:
        raise ZeroDivisionError
    return tp / (tp + fn)


def precision(tp, fp, tn, fn):
    if tp + fp == 0:
        raise ZeroDivisionError
    return tp / (tp + fp)


def f1(tp, fp, tn, fn):
    _recall = recall(tp, fp, tn, fn)
    _precision = precision(tp, fp, tn, fn)
    return 2 * (_precision * _recall) / (_precision + _recall)


def accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + fp + tn + fn)
