def yhat_default(m, d):
    return m.predict(d)


def yhat_proba(m, d):
    return m.predict_proba(d)[:, 1]


def yhat(model):
    if hasattr(model, 'predict_proba'):
        # check if model has predict_proba
        return yhat_proba
    elif hasattr(model, 'predict'):
        # check if model has predict
        return yhat_default

    return False