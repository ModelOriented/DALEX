def yhat_default(m, d):
    return m.predict(d)


def yhat_proba_default(m, d):
    return m.predict_proba(d)[:, 1]


def yhat_xgboost(m, d):
    from xgboost import DMatrix
    return m.predict(DMatrix(d))


def get_predict_function_and_model_type(model, model_class):
    # check for exceptions
    yhat_exception_dict = {
        "xgboost.core.Booster": (yhat_xgboost, None)  # there is no way of checking for regr/classif
    }
    if yhat_exception_dict.get(model_class, None) is not None:
        return yhat_exception_dict.get(model_class)

    model_type = None
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