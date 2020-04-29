def yhat_default(m, d):
    return m.predict(d)


def yhat_proba_default(m, d):
    return m.predict_proba(d)[:, 1]


def yhat_proba_xgboost(m, d):
    from xgboost import DMatrix
    return m.predict(DMatrix(d))


def yhat(model, model_class):
    yhat_exception_dict = {
        "xgboost.core.Booster": yhat_proba_xgboost
    }

    func = yhat_exception_dict.get(model_class, None)

    if func is not None:
        return func

    if hasattr(model, 'predict_proba'):
        # check if model has predict_proba
        return yhat_proba_default
    elif hasattr(model, 'predict'):
        # check if model has predict
        return yhat_default
    else:
        return False
