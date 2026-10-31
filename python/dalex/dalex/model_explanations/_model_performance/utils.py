import numpy as np


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


def roc_curve(y_pred, y_true):
    """Legacy ROC coordinates without the plotting origin, shared with AUC."""
    scores = np.asarray(y_pred)
    observed = np.asarray(y_true, dtype=float)
    # ROC is a vector algorithm. Preserve the legacy tie groups, missing-score
    # omission and denominators without constructing intermediate dataframes.
    if scores.dtype.kind in 'biuf':
        valid = ~np.isnan(scores)
        values = scores[valid]
    else:
        def missing(value):
            if value is None:
                return True
            absent = value != value
            return bool(absent) if isinstance(absent, (bool, np.bool_)) else True
        valid = np.array([not missing(value) for value in scores], dtype=bool)
        # Retain Python integers when nullable/object storage cannot be safely
        # represented by one signed/unsigned NumPy dtype.
        values = scores[valid]
    # Avoid negation overflow or float coercion of integer decision scores.
    order = np.argsort(values, kind='mergesort')[::-1]
    sorted_scores = values[order]
    labels = observed[valid][order]
    if not len(sorted_scores):
        return np.empty(0), np.empty(0)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    positive = np.add.reduceat(np.where(np.isnan(labels), 0., labels), starts)
    negative = np.add.reduceat(np.where(np.isnan(labels), 0., 1 - labels), starts)
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = np.cumsum(positive) / np.nansum(observed)
        fpr = np.cumsum(negative) / np.nansum(1 - observed)
    return fpr, tpr


def auc(y_pred, y_true):
    observed = np.asarray(y_true, dtype=float)
    if np.unique(observed[~np.isnan(observed)]).size == 1:
        raise ValueError("Only one class present in `y`. ROC AUC score is not defined in that case.")
    fpr, tpr = roc_curve(y_pred, observed)
    fpr, tpr = np.r_[0., fpr], np.r_[0., tpr]
    return (np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2).sum()


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
