# utility functions for Explainer methods
import numpy as np


def unpack_kwargs_lime(explainer, new_observation, **kwargs):
    # utility function for predict_surrogate(type='lime')
    # use https://stackoverflow.com/a/58543357 to unpack the **kwargs into multiple functions
    from lime.lime_tabular import LimeTabularExplainer
    import inspect

    explainer_args = [k for k, v in inspect.signature(LimeTabularExplainer).parameters.items()]
    explainer_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in explainer_args}
    explanation_args = [k for k, v in inspect.signature(
        LimeTabularExplainer.explain_instance).parameters.items()]
    explanation_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in explanation_args}

    if 'training_data' not in explainer_dict:
        explainer_dict['training_data'] = explainer.data.to_numpy()
    if 'mode' not in explainer_dict:
        explainer_dict['mode'] = explainer.model_type
    if 'data_row' not in explanation_dict:
        explanation_dict['data_row'] = new_observation
    if 'predict_fn' not in explanation_dict:
        if hasattr(explainer.model, 'predict_proba'):
            explanation_dict['predict_fn'] = explainer.model.predict_proba
        elif hasattr(explainer.model, 'predict'):
            explanation_dict['predict_fn'] = explainer.model.predict
        else:
            raise ValueError("Pass a `predict_fn` parameter to the `predict_surrogate` method. "
                             "See https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_tabular.LimeTabularExplainer.explain_instance")

    return explainer_dict, explanation_dict


def create_surrogate_model(explainer, type, max_vars, max_depth, **kwargs):
    # utility function for model_surrogate(type=('tree', 'linear'))

    y_hat = explainer.predict(explainer.data) if explainer.y_hat is None else explainer.y_hat

    # init a proper model
    if explainer.model_type == 'regression':
        if type == 'tree':
            from sklearn.tree import DecisionTreeRegressor
            surrogate_model = DecisionTreeRegressor(max_depth=max_depth, **kwargs)
        elif type == 'linear':
            from sklearn.linear_model import LinearRegression
            surrogate_model = LinearRegression(**kwargs)
    elif explainer.model_type == 'classification':
        y_hat = y_hat.round().astype(int)  # modify prob to classes
        if type == 'tree':
            from sklearn.tree import DecisionTreeClassifier
            surrogate_model = DecisionTreeClassifier(max_depth=max_depth, **kwargs)
        elif type == 'linear':
            from sklearn.linear_model import LogisticRegression
            surrogate_model = LogisticRegression(**kwargs)
    else:
        raise ValueError("'explainer.model_type' must be 'regression' or 'classification'")

    # find the most important variables
    if max_vars is not None and max_vars < len(explainer.data.columns):
        # use model native feature importance if possible
        if hasattr(explainer.model, "feature_importances_"):  # scikit, xgboost, lightgbm
            # take last max_vars of the variables sorted from least to most important
            vi = explainer.model.feature_importances_
            variables = explainer.data.columns[np.argsort(vi)][-max_vars:]
        else:
            # calculate model_parts and take max_vars most important variables
            vi = explainer.model_parts(N=500, B=15)
            variables = vi.result.variable[~vi.result.variable.isin(['_baseline_', '_full_model_'])].tail(max_vars)
        X = explainer.data.loc[:, variables]
    else:
        variables = explainer.data.columns
        X = explainer.data

    surrogate_model.fit(X, y_hat)

    # add additional attributes to the surrogate model object
    surrogate_model.feature_names = variables.tolist()
    surrogate_model.class_names = \
        surrogate_model.classes_.astype(str).tolist() if hasattr(surrogate_model, 'classes_') else None

    from .object import Explainer
    surrogate_explainer = Explainer(surrogate_model, X, y_hat, model_type=explainer.model_type, verbose=False)
    surrogate_model_performance = surrogate_explainer.model_performance()
    surrogate_model.performance = surrogate_model_performance.result

    # add the plot method to the surrogate model object
    if type == 'tree':
        # it uses feature_names and class_names if needed
        import types
        surrogate_model.plot = types.MethodType(plot_tree_custom, surrogate_model)

    # TODO: add plot method for the linear model

    return surrogate_model


def plot_tree_custom(model, **kwargs):
    # wrapper for the plot_tree function, that makes the plot look useful
    # it does not return the plot (because plot_tree works so)
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(16, 10))
    _ = plot_tree(model,
                  filled=True,
                  fontsize=10,
                  ax=ax,
                  feature_names=model.feature_names,
                  class_names=model.class_names,
                  proportion=True,
                  **kwargs)
