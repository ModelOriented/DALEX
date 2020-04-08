import re

import numpy as np
import pandas as pd

from .helper import verbose_cat, is_y_in_data, get_model_info, yhat


def check_label(label, model, verbose):
    if label is None:
        # label not specified
        # try to extract something
        label = re.search(".*(?='>$)", str(type(model)).split('.')[-1])[0]
        verbose_cat("  -> label             : not specified, model's class taken instead!", verbose=verbose)
    else:
        verbose_cat("  -> target variable   :  " + str(label), verbose=verbose)

    return label


def check_data(data, verbose):
    if isinstance(data, np.ndarray):
        verbose_cat("data is numpy ndarray, columns are set as consecutive numbers", verbose=verbose)
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame) and data is not None:
        raise TypeError("data must be pandas.DataFrame or numpy.ndarray")
    elif data is None:
        verbose_cat("  -> data   :  not specified!", verbose=verbose)

    if data is not None:
        if data.index.unique().shape[0] != data.shape[0]:
            raise ValueError("Index is not unique")
        
        verbose_cat("  -> data              : " + str(data.shape[0]) + " rows " + str(data.shape[1]) + " cols",
                    verbose=verbose)

    return data


def check_y(y, data, verbose):
    if isinstance(y, pd.Series):
        y = np.array(y)
        verbose_cat("  -> target variable   :  Argument 'y' was a pandas.Series. Converted to a numpy.ndarray.",
                    verbose=verbose)
    elif isinstance(y, np.ndarray) and len(y.shape) != 1:
        raise ValueError("y must have only one dimension")
    elif not isinstance(y, np.ndarray) and data is not None:
        raise TypeError("y must be numpy.ndarray or pandas.Series")
    elif y is None:
        verbose_cat("  -> target variable   :  not specified!", verbose=verbose)

    if y is not None:
        verbose_cat("  -> target variable   : " + str(y.shape[0]) + " values", verbose=verbose)

    if y.shape[0] != data.shape[0]:
        verbose_cat("  -> target variable   :  length of 'y' is different than number of rows in 'data'",
                    verbose=verbose)

    if isinstance(y[0], str):
        verbose_cat("  -> target variable   :  Please note that 'y' is character vector.", verbose=verbose)
        verbose_cat("  -> target variable   :  Consider changing the 'y' to a logical or numerical vector.",
                    verbose=verbose)
        verbose_cat(
            "  -> target variable   :  Otherwise I will not be able to calculate residuals or loss function.",
            verbose=verbose)

    if data is not None and is_y_in_data(data, y):
        verbose_cat(
            "  -> data              :  A column identical to the target variable `y` has been found in the `data`.",
            verbose=verbose)
        verbose_cat(
            "  -> data              :  It is highly recommended to pass `data` without the target variable column",
            verbose=verbose)

    return y


def check_weights(weights, data, verbose):
    if weights is None:
        # weights not specified
        # do nothing
        pass
    else:
        if isinstance(weights, pd.Series):
            weights = np.array(weights)
            verbose_cat(
                "  -> sampling weights  :  Argument 'weights' was a pandas.Series. Converted to a numpy vector.",
                verbose=verbose)
        elif isinstance(weights, np.ndarray) and len(weights.shape) != 1:
            raise TypeError(
                "  -> sampling weights  :  Argument 'weights must be a 1D numpy.ndarray or pandas.Series")
        elif not isinstance(weights, np.ndarray):
            raise TypeError("  -> sampling weights  :  Argument 'weights must be numpy.ndarray or pandas.Series")

        verbose_cat("  -> sampling weights  : " + str(weights.shape[0]) + " values", verbose=verbose)

        if weights.shape[0] != data.shape[0]:
            verbose_cat(
                "  -> sampling weights  :  length of 'weights' is different than number of rows in 'data'",
                verbose=verbose)

    return weights


def check_predict_function(predict_function, model, data, precalculate, verbose):
    if predict_function is None:
        # predict_function not specified
        # try the default
        predict_function = yhat(model)

        if not predict_function:
            raise ValueError("  -> predict function  : predict_function not provided and cannot be extracted",
                             verbose=verbose)

    verbose_cat("  -> predict function  : " + str(predict_function) + " will be used", verbose=verbose)

    pred = None
    if data is not None and (verbose or precalculate):
        try:
            pred = predict_function(model, data)
            verbose_cat("  -> predicted values  : min = " + str(np.min(pred)) + ", mean = " + str(np.mean(pred)) +
                        ", max = " + str(np.max(pred)), verbose=verbose)

        except (Exception, ValueError, TypeError) as error:
            verbose_cat("  -> predicted values  :  the predict_function returns an error when executed",
                        verbose=verbose)
            print(error)

    return predict_function, pred


def check_residual_function(residual_function, predict_function, model, data, y, precalculate, verbose):
    if residual_function is None:
        # residual_function not specified
        # try the default
        def residual_function(_model, _data, _y):
            return _y - predict_function(_model, _data)
        verbose_cat("  -> residual function : difference between y and yhat", verbose=verbose)
    else:
        verbose_cat("  -> residual function : " + str(residual_function), verbose=verbose)

    # if data is specified then we may test residual_function
    residuals = None
    if data is not None and y is not None and (verbose or precalculate):
        try:
            residuals = residual_function(model, data, y)
            verbose_cat(
                "  -> residuals         : min = " + str(np.min(residuals)) + ", mean = " + str(np.mean(residuals)) +
                ", max = " + str(np.max(residuals)), verbose=verbose)
        except (Exception, ValueError, TypeError) as error:
            verbose_cat("  -> residuals         :  the residual_function returns an error when executed",
                        verbose=verbose)
            print(error)

    return residual_function, residuals


def check_model_info(model_info, model, verbose):
    if model_info is None:
        # extract defaults
        model_info = get_model_info(model)

        verbose_cat("  -> model_info        : package " + model_info['model_package'], verbose=verbose)
    else:
        verbose_cat("  -> model_info        : package " + model_info['model_package'] +
                    ", ver." + model_info['version'] + ", task" + model_info['type'], verbose=verbose)

    return model_info


def check_method_type(type, types):
    if isinstance(type, tuple):
        ret = type[0]
    else:
        ret = type
    if ret not in types:
        raise TypeError("'type' must be one of: {}".format(', '.join(types)))
    else:
        return ret

