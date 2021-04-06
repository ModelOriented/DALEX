import numpy as np
import pandas as pd

from .exceptions import *
from ..._explainer import helper


def check_parameters(y, y_hat, protected, privileged, verbose):
    if not len(y) == len(y_hat) == len(protected):
        raise ParameterCheckError("protected and explainer attributes y, y_hat must have the same lengths")

    if not isinstance(verbose, bool):
        raise ParameterCheckError("verbose must be boolean, either False or True")

    if not isinstance(protected, np.ndarray):
        # if is not numpy array check what type it is and change to np array
        if isinstance(protected, list):
            try:
                helper.verbose_cat("protected list will be converted to np.ndarray", verbose)
                protected = np.array(protected, dtype='U')
            except:
                ParameterCheckError("failed to convert list to np.ndarray, try converting it manually", verbose)

        elif isinstance(protected, pd.Series):
            try:
                helper.verbose_cat("protected Series will be converted to np.ndarray", verbose)
                protected = np.array(protected, dtype='U')
            except ParameterCheckError:
                ParameterCheckError("failed to convert list to np.ndarray")
        else:
            ParameterCheckError("unsupported protected type provided. Please convert protected to flat np.ndarray")

    if protected.dtype.type is not np.str_:
        try:
            helper.verbose_cat("protected array is not string type, converting to string ", verbose)
            protected = protected.astype(str)
        except:
            ParameterCheckError('Could not convert protected to String type')

    if not isinstance(privileged, str):
        try:
            privileged = str(privileged)
        except:
            ParameterCheckError('Could not convert privileged to String')

    if privileged not in protected:
        raise ParameterCheckError("privileged parameter must be in protected vector")

    return y, y_hat, protected, privileged


def check_other_objects(fobject, other):
    class_type = fobject.__class__

    other_objects = []
    for obj in other:
        if isinstance(obj, class_type):
            other_objects.append(obj)

    return other_objects


def check_other_fairness_objects(fobject, other):
    """
    Checking compatibility of GroupFairnessClassification objects
    """
    for other_obj in other:

        if fobject.protected.shape != other_obj.protected.shape:
            raise FairnessObjectsDifferenceError('protected attributes have different shapes')

        if any(fobject.protected != other_obj.protected):
            raise FairnessObjectsDifferenceError('protected attributes are not the same')

        if fobject.privileged != other_obj.privileged:
            raise FairnessObjectsDifferenceError('privileged subgroups are not the same')

        if fobject.y.shape != other_obj.y.shape:
            raise FairnessObjectsDifferenceError('target variable (y) has different shape among Fairness objects')

        if any(fobject.y != other_obj.y):
            raise FairnessObjectsDifferenceError('target variable (y) is not the same among Fairness objects')

        if fobject.epsilon != other_obj.epsilon:
            raise FairnessObjectsDifferenceError('epsilon value is not the same among Fairness objects')

    # check uniqueness of label
    labels = [fobject.label]
    for other_obj in other:
        labels.append(other_obj.label)

    if len(labels) != len(set(labels)):
        raise FairnessObjectsDifferenceError(
            'Fairness labels are not unique and therefore objects cannot be plotted together'
        )
