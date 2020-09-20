from warnings import warn
import numpy as np


def yhat_default(m, d):
    return m.predict(d)


def yhat_proba_default(m, d):
    return m.predict_proba(d)[:, 1]


def yhat_xgboost(m, d):
    from xgboost import DMatrix
    return m.predict(DMatrix(d))


def get_tf_prediction_function(model):
    if model.output_shape[1] == 1:
        return yhat_tf_regression, "regression"
    elif model.output_shape[1] == 2:
        return yhat_tf_classification, "classification"
    else:
        warn("Tensorflow: Output shape of the predict method should not be greater than 2")
        return yhat_tf_classification, "classification"


def yhat_tf_regression(m, d):
    return m.predict(np.array(d)).reshape(-1, )


def yhat_tf_classification(m, d):
    return m.predict(np.array(d))[:, 1]


def get_predict_function_and_model_type(model, model_class):
    # check for exceptions
    yhat_exception_dict = {
        "xgboost.core.Booster": (yhat_xgboost, None),  # there is no way of checking for regr/classif
        "tensorflow.python.keras.engine.sequential.Sequential": get_tf_prediction_function(model)
    }
    if yhat_exception_dict.get(model_class, None) is not None:
        return yhat_exception_dict.get(model_class)

    model_type_ = None
    # additional extraction sklearn
    if hasattr(model, '_estimator_type'):
        if model._estimator_type == 'classifier':
            model_type_ = 'classification'
        elif model._estimator_type == 'regressor':
            model_type_ = 'regression'

    # default extraction
    if hasattr(model, 'predict_proba'):
        # check if model has predict_proba
        return yhat_proba_default, 'classification' if model_type_ is None else model_type_
    elif hasattr(model, 'predict'):
        # check if model has predict
        return yhat_default, 'regression' if model_type_ is None else model_type_
    else:
        # this path results in an error later
        return False, None