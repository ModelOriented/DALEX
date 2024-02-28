# check functions for Explainer.__init__
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy

from .helper import verbose_cat, is_y_in_data, get_model_info
from .yhat import get_predict_function_and_model_type


def check_data(data, model, verbose):
    if data is None:
        verbose_cat("  -> data              : Not specified!", verbose=verbose)
    else:
        if isinstance(data, np.ndarray):
            verbose_cat("  -> data              : numpy.ndarray converted to pandas.DataFrame. Columns are set as string numbers.", verbose=verbose)
            data = pd.DataFrame(data)
            data.columns = data.columns.astype(str)
        elif type(data).__name__ == "H2OFrame":  # can't import h2o
            verbose_cat("  -> data              : h2o.H2OFrame converted to pandas.DataFrame. Column types are saved in the model.", verbose=verbose)
            model._column_types = data.types  # pandas attribute 'attrs' is experimental and may break in future versions        
            data = data.as_data_frame()
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be pandas.DataFrame or numpy.ndarray")

        if data.index.unique().shape[0] != data.shape[0]:
            raise ValueError("'data' index is not unique")

        verbose_cat("  -> data              : " + str(data.shape[0]) + " rows " + str(data.shape[1]) + " cols",
                    verbose=verbose)

    return data, model


def check_method_data(data):
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise TypeError("'data' has to be pandas.DataFrame or numpy.ndarray")

    if isinstance(data, np.ndarray) and data.ndim != 2:
        raise ValueError("'data' must have 2 dimensions")


def check_data_again(data):
    if data is None:
        raise ValueError("'data' attribute in Explainer is missing")


def check_y(y, data, verbose):
    if y is None:
        verbose_cat("  -> target variable   : Not specified!", verbose=verbose)
    else:
        if isinstance(y, pd.Series):
            y = np.array(y)
            verbose_cat("  -> target variable   : Parameter 'y' was a pandas.Series. Converted to a numpy.ndarray.",
                        verbose=verbose)
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("'y' must be only one column")
            y = y.to_numpy().flatten()
            verbose_cat("  -> target variable   : Parameter 'y' was a pandas.DataFrame. Converted to a numpy.ndarray.",
                        verbose=verbose)
        elif type(y).__name__ == "H2OFrame":  # can't import h2o
            verbose_cat("  -> target variable   : Parameter 'y' was a h2o.H2OFrame. Converted to a numpy.ndarray.",
                        verbose=verbose)
            y = y.as_data_frame().to_numpy().flatten()
        elif isinstance(y, np.ndarray) and len(y.shape) != 1:
            raise ValueError("'y' must have only one dimension")
        elif not isinstance(y, np.ndarray):
            raise TypeError("'y' must be a numpy.ndarray (1d) or pandas.Series")
        
        verbose_cat("  -> target variable   : " + str(y.shape[0]) + " values", verbose=verbose)

        if isinstance(y[0], str):
            verbose_cat("  -> target variable   : Please note that 'y' is a string array.", verbose=verbose)
            verbose_cat("  -> target variable   : 'y' should be a numeric or boolean array.",
                        verbose=verbose)
            verbose_cat(
                "  -> target variable   : Otherwise an Error may occur in calculating residuals or loss.",
                verbose=verbose)

        if data is not None:
            if y.shape[0] != data.shape[0]:
                verbose_cat("  -> target variable   : Length of 'y' is different than the number of rows in 'data'.",
                            verbose=verbose)

            if is_y_in_data(data, y):
                verbose_cat(
                    "  -> data              : A column identical to the target variable" +
                    " `y` has been found in the `data`.",
                    verbose=verbose)
                verbose_cat(
                    "  -> data              : It is highly recommended to pass `data` without" +
                    " the target variable column.",
                    verbose=verbose)

    return y


def check_y_again(y):
    if y is None:
        raise ValueError("'y' attribute in Explainer is missing")
    if isinstance(y[0], str):
        raise ValueError("'y' attribute in Explainer is of string type and it should be numerical "
                         "to allow for calculations")


def check_weights(weights, data, verbose):
    if weights is None:
        # weights not specified
        # do nothing
        pass
    else:
        check_data_again(data)

        if isinstance(weights, pd.Series):
            weights = np.array(weights)
            verbose_cat(
                "  -> sampling weights  :  Parameter 'weights' was a pandas.Series. Converted to a numpy.ndarray.",
                verbose=verbose)
        elif isinstance(weights, np.ndarray) and len(weights.shape) != 1:
            raise TypeError("'weights' must have only one dimension")
        elif not isinstance(weights, np.ndarray):
            raise TypeError("'weights' must be a numpy.ndarray (1d) or pandas.Series")

        verbose_cat("  -> sampling weights  : " + str(weights.shape[0]) + " values", verbose=verbose)

        if weights.shape[0] != data.shape[0]:
            verbose_cat(
                "  -> sampling weights  :  Length of 'weights' is different than the number of rows in 'data'.",
                verbose=verbose)

    return weights


def check_model_class(model_class, model, verbose):
    model_info = get_model_info(model)

    if model_class is None:
        if hasattr(model, "_final_estimator"):
            model_class = str(type(model._final_estimator))
            model_info['Pipeline'] = True
        else:
            model_class = str(type(model))
            model_info['Pipeline'] = False
        from re import search
        model_class = search("(?<=<class ').*(?='>)", model_class)[0]
        model_info['model_class_default'] = True

        verbose_cat("  -> model_class       : " + model_class + " (default)", verbose=verbose)

    else:
        model_info['model_class_default'] = False

        verbose_cat("  -> model_class       : " + model_class, verbose=verbose)

    return model_class, model_info


def check_label(label, model_class, model_info, verbose):
    if label is None:
        # label not specified
        # try to extract something

        label = model_class.split('.')[-1]
        verbose_cat("  -> label             : Not specified, model's class short name will be used. (default)",
                    verbose=verbose)

        model_info['label_default'] = True
    else:
        if not isinstance(label, str):
            raise TypeError("'label' should be a string.")

        verbose_cat("  -> label             : " + label, verbose=verbose)
        model_info['label_default'] = False

    return label, model_info


def check_predict_function_and_model_type(predict_function, model_type,
                                          model, data, model_class, model_info_, precalculate, verbose):
    if predict_function is None:
        # predict_function not specified
        # try the default
        predict_function, model_type_ = get_predict_function_and_model_type(model, model_class)

        model_info_['predict_function_default'] = True

        if not predict_function:
            raise ValueError("  -> predict function  : 'predict_function' not provided and cannot be extracted.")

        verbose_cat("  -> predict function  : " + str(predict_function) + " will be used (default)", verbose=verbose)
    else:
        _, model_type_ = get_predict_function_and_model_type(model, model_class)
        model_info_['predict_function_default'] = False
        verbose_cat("  -> predict function  : " + str(predict_function) + " will be used", verbose=verbose)

    y_hat = None
    if data is not None:
        # check if predict_function accepts arrays
        try:
            data_values = data.values[[0]]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') # ignore warnings about feature names in scikit-learn
                predict_function(model, data_values)
            model_info_['arrays_accepted'] = True
            verbose_cat("  -> predict function  : Accepts pandas.DataFrame and numpy.ndarray.",
                        verbose=verbose)
        except:
            model_info_['arrays_accepted'] = False
            verbose_cat("  -> predict function  : Accepts only pandas.DataFrame, numpy.ndarray causes problems.",
                        verbose=verbose)

        if verbose or precalculate:
            try:
                y_hat = predict_function(model, data)
                verbose_cat(str.format("  -> predicted values  : min = {0:.3}, mean = {1:.3}, max = {2:.3}",
                                       np.min(y_hat).astype(float), np.mean(y_hat).astype(float),
                                       np.max(y_hat).astype(float)), verbose=verbose)

            except (Exception, ValueError, TypeError) as error:
                # verbose_cat("  -> predicted values  : the predict_function returns an error when executed \n",
                #             verbose=verbose)

                warnings.warn("\n  -> predicted values  : 'predict_function' returns an Error when executed: \n" +
                     str(error), stacklevel=2)

            if not isinstance(y_hat, np.ndarray) or y_hat.shape != (data.shape[0], ):
                warnings.warn("\n  -> predicted values  : 'predict_function' must return numpy.ndarray (1d)", stacklevel=2)

    if model_type is None:
        # model_type not specified
        if model_type_ is None:
            verbose_cat("  -> model type        : 'model_type' not provided and cannot be extracted.", verbose=verbose)
            verbose_cat("  -> model type        : Some functionalities won't be available.", verbose=verbose)
        else:
            # use extracted model_type
            model_type = model_type_
            model_info_['model_type_default'] = True
            verbose_cat("  -> model type        : " + str(model_type) + " will be used (default)", verbose=verbose)
    else:
        model_info_['model_type_default'] = False
        verbose_cat("  -> model type        : " + str(model_type) + " will be used", verbose=verbose)

    return predict_function, model_type, y_hat, model_info_


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
                                   np.min(residuals).astype(float), np.mean(residuals).astype(float),
                                   np.max(residuals).astype(float)), verbose=verbose)
        except (Exception, ValueError, TypeError) as error:
            verbose_cat("  -> residuals         :  'residual_function' returns an Error when executed:",
                        verbose=verbose)
            print(error)

    return residual_function, residuals, model_info


def check_model_info(model_info, model_info_, verbose):
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            model_info_[key] = value

    verbose_cat("  -> model_info        : package " + model_info_['model_package'], verbose=verbose)

    return model_info_


def check_method_type(type, types, aliases=None):
    if isinstance(type, tuple):
        ret = type[0]
    elif isinstance(type, str):
        ret = type
    else:
        raise TypeError("'type' is not a string")

    if ret not in types:
        if aliases:
            if ret not in aliases:
                raise ValueError("'type' must be one of: {}".format(', '.join(types+tuple(aliases))))
            else:
                return aliases[ret]
        else:
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
        print("  -> 'predict_function' attribute is a local function; thus, has to be dropped.")
        to_dump.predict_function = None
    elif re.search(is_lambda, pred_func):
        print("  -> 'predict_function' attribute is a lambda; thus, has to be dropped.")
        to_dump.predict_function = None
    if re.search(is_local, res_func):
        print("  -> 'residual_function' attribute is a local function; thus, has to be dropped.")
        to_dump.residual_function = None
    elif re.search(is_lambda, res_func):
        print("  -> 'residual_function' attribute is a lambda; thus, has to be dropped.")
        to_dump.residual_function = None

    # for R compatibility
    to_dump.model_info['type'] = to_dump.model_type

    return to_dump


def check_if_empty_fields(explainer):
    if explainer.predict_function is None:
        print("  -> 'predict_function' is not present, setting to default.")
        predict_function, model_type, y_hat, model_info_ = \
            check_predict_function_and_model_type(predict_function=None,
                                                  model_type=explainer.model_type,
                                                  model=explainer.model,
                                                  data=explainer.data,
                                                  model_class=explainer.model_class,
                                                  model_info_=explainer.model_info,
                                                  precalculate=True,
                                                  verbose=False)
        explainer.predict_function = predict_function
        explainer.model_type = model_type
        explainer.y_hat = y_hat
    else:
        model_info_ = {}
    if explainer.residual_function is None:
        print("  -> 'residual_function' is not present, setting to default.")
        residual_function, residuals, model_info = \
            check_residual_function(residual_function=None,
                                    predict_function=explainer.predict_function,
                                    model=explainer.model,
                                    data=explainer.data,
                                    y=explainer.y,
                                    model_info=explainer.model_info,
                                    precalculate=True,
                                    verbose=False)
        explainer.residual_function = residual_function
        explainer.residuals = residuals
    else:
        model_info = explainer.model_info
    
    explainer.model_info = check_model_info(model_info, model_info_, verbose=False)

    return explainer


def check_method_loss_function(explainer, loss_function):
    if loss_function is not None or explainer.model_type is None:  # user passed a function or type is not known
        return loss_function
    elif explainer.model_type == 'regression':
        return 'rmse'
    elif explainer.model_type == 'classification':
        return '1-auc'


def check_new_observation_lime(new_observation):
    # lime accepts only np.array as data_row

    new_observation_ = deepcopy(new_observation)
    if isinstance(new_observation_, pd.Series):
        new_observation_ = new_observation_.to_numpy()
    elif isinstance(new_observation_, np.ndarray):
        if new_observation_.ndim == 2:
            if new_observation_.shape[0] != 1:
                raise ValueError("'new_observation' should be a single row")
            # make 2D array 1D
            new_observation_ = new_observation_.flatten()
        elif new_observation_.ndim > 2:
            raise ValueError("'new_observation' has too many dimensions")
    elif isinstance(new_observation_, list):
        new_observation_ = np.array(new_observation_)
    elif isinstance(new_observation_, pd.DataFrame):
        if new_observation_.shape[0] != 1:
            raise ValueError("'new_observation' should be a single row")
        else:
            new_observation_ = new_observation_.to_numpy().flatten()
    else:
        raise TypeError("'new_observation' must be a list or numpy.ndarray (1d) or pandas.Series or pandas.DataFrame")

    return new_observation_
