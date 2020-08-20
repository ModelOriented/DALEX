# check functions for Explainer.__init__
import pandas as pd
from copy import deepcopy

from .helper import *
from .yhat import *


def check_path(path):
    if not isinstance(path, (str, list, np.ndarray, pd.Series)):
        raise TypeError('path must be str or list or numpy.ndarray or pd.Series')
    if isinstance(path, str):
        if path != 'average':
            raise ValueError("if path is str, then it must be 'average'")
        else:
            return path

    path_ = np.array(path)
    if not isinstance(path_[0], np.integer):
        raise TypeError('if path is array-like, then must be array of integers')

    return path_


def check_pred_data(data):
    if not isinstance(data, (pd.DataFrame,)):
        raise TypeError('data has to be pandas.DataFrame')

    # if isinstance(data, np.ndarray) and data.ndim != 2:
    #     raise ValueError("data must have 2 dimensions")


def check_label(label, model_class, model_info, verbose):
    if label is None:
        # label not specified
        # try to extract something

        label = model_class.split('.')[-1]
        verbose_cat("  -> label             : not specified, model's class short name is taken instead (default)",
                    verbose=verbose)

        model_info['label_default'] = True
    else:
        if not isinstance(label, str):
            raise TypeError("Label is not a string")

        verbose_cat("  -> label             : " + label, verbose=verbose)
        model_info['label_default'] = False

    return label, model_info


def check_data(data, verbose):
    if isinstance(data, np.ndarray):
        verbose_cat("data is converted to pd.DataFrame, columns are set as string numbers", verbose=verbose)
        data = pd.DataFrame(data)
        data.columns = data.columns.astype(str)
    elif not isinstance(data, pd.DataFrame) and data is not None:
        raise TypeError("data must be pandas.DataFrame or numpy.ndarray")
    elif data is None:
        verbose_cat("  -> data   : not specified!", verbose=verbose)

    if data is not None:
        if data.index.unique().shape[0] != data.shape[0]:
            raise ValueError("Index is not unique")

        verbose_cat("  -> data              : " + str(data.shape[0]) + " rows " + str(data.shape[1]) + " cols",
                    verbose=verbose)

    return data


def check_y(y, data, verbose):
    if isinstance(y, pd.Series):
        y = np.array(y)
        verbose_cat("  -> target variable   : Argument 'y' was a pandas.Series. Converted to a numpy.ndarray.",
                    verbose=verbose)
    elif isinstance(y, np.ndarray) and len(y.shape) != 1:
        raise ValueError("y must have only one dimension")
    elif not isinstance(y, np.ndarray) and data is not None:
        raise TypeError("y must be a 1D numpy.ndarray or pandas.Series")
    elif y is None:
        verbose_cat("  -> target variable   : not specified!", verbose=verbose)

    if y is not None:
        verbose_cat("  -> target variable   : " + str(y.shape[0]) + " values", verbose=verbose)

    if y.shape[0] != data.shape[0]:
        verbose_cat("  -> target variable   : length of 'y' is different than number of rows in 'data'",
                    verbose=verbose)

    if isinstance(y[0], str):
        verbose_cat("  -> target variable   : Please note that 'y' is character vector.", verbose=verbose)
        verbose_cat("  -> target variable   : Consider changing the 'y' to a logical or numerical vector.",
                    verbose=verbose)
        verbose_cat(
            "  -> target variable   : Otherwise I will not be able to calculate residuals or loss function.",
            verbose=verbose)

    if data is not None and is_y_in_data(data, y):
        verbose_cat(
            "  -> data              : A column identical to the target variable `y` has been found in the `data`.",
            verbose=verbose)
        verbose_cat(
            "  -> data              : It is highly recommended to pass `data` without the target variable column",
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


def check_predict_function(predict_function, model, data, model_class, model_info_, precalculate, verbose):
    if predict_function is None:
        # predict_function not specified
        # try the default
        predict_function, model_type_ = yhat(model, model_class)

        model_info_['predict_function_default'] = True

        if not predict_function:
            raise ValueError("  -> predict function  : predict_function not provided and cannot be extracted",
                             verbose=verbose)

        verbose_cat("  -> predict function  : " + str(predict_function) + " will be used (default)", verbose=verbose)

    else:
        model_type_ = None
        model_info_['predict_function_default'] = False
        verbose_cat("  -> predict function  : " + str(predict_function) + " will be used", verbose=verbose)

    pred = None
    if data is not None and (verbose or precalculate):
        try:
            pred = predict_function(model, data)
            verbose_cat(str.format("  -> predicted values  : min = {0:.3}, mean = {1:.3}, max = {2:.3}",
                                   np.min(pred), np.mean(pred), np.max(pred)), verbose=verbose)

        except (Exception, ValueError, TypeError) as error:
            verbose_cat("  -> predicted values  :  the predict_function returns an error when executed",
                        verbose=verbose)
            print(error)

    return predict_function, pred, model_info_, model_type_


def check_residual_function(residual_function, predict_function, model, data, y, model_info, precalculate, verbose):
    if residual_function is None:
        # residual_function not specified
        # try the default
        def residual_function(_model, _data, _y):
            return _y - predict_function(_model, _data)

        verbose_cat("  -> residual function : difference between y and yhat (default)", verbose=verbose)
        model_info['residual_function_default'] = True
    else:
        verbose_cat("  -> residual function : " + str(residual_function), verbose=verbose)
        model_info['residual_function_default'] = False

    # if data is specified then we may test residual_function
    residuals = None
    if data is not None and y is not None and (verbose or precalculate):
        try:
            residuals = residual_function(model, data, y)
            verbose_cat(str.format("  -> residuals         : min = {0:.3}, mean = {1:.3}, max = {2:.3}",
                                   np.min(residuals), np.mean(residuals), np.max(residuals)), verbose=verbose)
        except (Exception, ValueError, TypeError) as error:
            verbose_cat("  -> residuals         :  the residual_function returns an error when executed",
                        verbose=verbose)
            print(error)

    return residual_function, residuals, model_info


def check_model_info(model_info, model_info_, verbose):
    if model_info is not None:
        for key, value in model_info_.items():
            model_info[key] = value
    else:
        model_info = model_info_

    verbose_cat("  -> model_info        : package " + model_info['model_package'], verbose=verbose)

    return model_info


def check_method_type(type, types):
    if isinstance(type, tuple):
        ret = type[0]
    elif isinstance(type, str):
        ret = type
    else:
        raise TypeError("type is not a str")

    if ret not in types:
        raise ValueError("'type' must be one of: {}".format(', '.join(types)))
    else:
        return ret


def check_if_local_and_lambda(to_dump):
    import re

    pred_func = str(to_dump.predict_function)
    res_func = str(to_dump.residual_function)

    is_local = "<locals>"
    is_lambda = "<lambda>"

    if re.search(is_local, pred_func):
        print("  -> Predict function is local, thus has to be dropped.")
        to_dump.predict_function = None
    elif re.search(is_lambda, pred_func):
        print("  -> Predict function is lambda, thus has to be dropped.")
        to_dump.predict_function = None
    if re.search(is_local, res_func):
        print("  -> Residual function is local, thus has to be dropped.")
        to_dump.residual_function = None
    elif re.search(is_lambda, res_func):
        print("  -> Residual function is lambda, thus has to be dropped.")
        to_dump.residual_function = None

    return to_dump


def check_if_empty_fields(explainer):
    if explainer.predict_function is None:
        print("  -> Predict function is not present, setting to default")
        predict_function, pred = check_predict_function(None, explainer.model, None, False, False)
        explainer.predict_function = predict_function
    if explainer.residual_function is None:
        print("  -> Residual function is not present, setting to default")
        residual_function, residuals = check_residual_function(None, explainer.predict_function, explainer.model, None,
                                                               None,
                                                               False, False)
        explainer.residual_function = residual_function

    return explainer


def check_model_class(model_class, model, verbose):
    model_info = get_model_info(model)

    if model_class is None:
        model_class = str(type(model))
        from re import search
        model_class = search("(?<=<class ').*(?='>)", model_class)[0]
        model_info['model_class_default'] = True

        verbose_cat("  -> model_class       : " + model_class + " (default)", verbose=verbose)

    else:
        model_info['model_class_default'] = False

        verbose_cat("  -> model_class       : " + model_class, verbose=verbose)

    return model_class, model_info


def check_loss_function(explainer, loss_function):
    if loss_function is not None or explainer.model_type is None:  # user passed a function or type is not known
        return loss_function
    elif explainer.model_type == 'regression':
        return 'rmse'
    elif explainer.model_type == 'classification':
        return '1-auc'


def check_model_type(model_type, model_type_):
    return model_type_ if model_type is None else model_type


def check_new_observation_lime(new_observation):
    # lime accepts only np.array as data_row

    new_observation_ = deepcopy(new_observation)
    if isinstance(new_observation_, pd.Series):
        new_observation_ = new_observation_.to_numpy()
    elif isinstance(new_observation_, np.ndarray):
        if new_observation_.ndim == 2:
            if new_observation.shape[0] != 1:
                raise ValueError("Wrong new_observation dimension")
            # make 2D array 1D
            new_observation_ = new_observation_.flatten()
        elif new_observation_.ndim > 2:
            raise ValueError("Wrong new_observation dimension")
    elif isinstance(new_observation_, list):
        new_observation_ = np.array(new_observation_)
    elif isinstance(new_observation_, pd.DataFrame):
        if new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")
        else:
            new_observation_ = new_observation.to_numpy().flatten()
    else:
        raise TypeError("new_observation must be a list or numpy.ndarray or pandas.Series or pandas.DataFrame")

    return new_observation_
