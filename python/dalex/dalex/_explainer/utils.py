# utility functions for Explainer methods
import numpy as np
import pandas as pd

import types, inspect, warnings
    
def create_lime_explanation(explainer, new_observation, **kwargs):
    # utility function for predict_surrogate(type='lime')    
    from lime.lime_tabular import LimeTabularExplainer
    explainer_dict, explanation_dict = unpack_kwargs_lime(explainer, new_observation, **kwargs)
    lime_tabular_explainer = LimeTabularExplainer(**explainer_dict)
    explanation = lime_tabular_explainer.explain_instance(**explanation_dict)
    
    explanation.plot = types.MethodType(plot_lime_custom, explanation)
    explanation.result = pd.DataFrame(explanation.as_list(), columns=['variable', 'effect'])
    return explanation
    
def unpack_kwargs_lime(explainer, new_observation, **kwargs):
    # utility function for predict_surrogate(type='lime')
    # use https://stackoverflow.com/a/58543357 to unpack the **kwargs into multiple functions
    from lime.lime_tabular import LimeTabularExplainer

    explainer_args = [k for k, v in inspect.signature(LimeTabularExplainer).parameters.items()]
    explainer_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in explainer_args}
    explanation_args = [k for k, v in inspect.signature(
        LimeTabularExplainer.explain_instance).parameters.items()]
    explanation_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in explanation_args}

    if 'training_data' not in explainer_dict:
        explainer_dict['training_data'] = explainer.data.to_numpy()
    if 'mode' not in explainer_dict:
        explainer_dict['mode'] = explainer.model_type
    if 'feature_names' not in explainer_dict:
        explainer_dict['feature_names'] = explainer.data.columns
    if 'data_row' not in explanation_dict:
        explanation_dict['data_row'] = new_observation
    if 'predict_fn' not in explanation_dict:
        if explainer_dict['mode'] == 'regression':
            explanation_dict['predict_fn'] = lambda x: explainer.predict(pd.DataFrame(x, columns=explainer.data.columns))
        elif explainer_dict['mode'] == 'classification':
            explanation_dict['predict_fn'] = \
                lambda x: np.concatenate([1 - explainer.predict(pd.DataFrame(x, columns=explainer.data.columns)).reshape(-1, 1),
                                          explainer.predict(pd.DataFrame(x, columns=explainer.data.columns)).reshape(-1, 1)],
                                         axis=1)
        else:
            raise ValueError("Pass a 'mode' parameter to the `predict_surrogate` method. "
                             "See https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_tabular.LimeTabularExplainer")
        try:
            explanation_dict['predict_fn'](explainer_dict['training_data'])
        except:
            raise ValueError("Pass a proper `predict_fn` parameter to the `predict_surrogate` method. "
                             "See https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_tabular.LimeTabularExplainer.explain_instance")
    return explainer_dict, explanation_dict


def plot_lime_custom(explanation, label=1, return_figure=False, **kwargs):
    # return_figure=True makes two plots in IPython
    if return_figure:
        return explanation.as_pyplot_figure(label=label, **kwargs)
    else:
        _ = explanation.as_pyplot_figure(label=label, **kwargs)

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
        else:
            raise TypeError("'type' parameter must be one of {'tree', 'linear'}")
    elif explainer.model_type == 'classification':
        y_hat = y_hat.round().astype(int)  # modify prob to classes
        if type == 'tree':
            from sklearn.tree import DecisionTreeClassifier
            surrogate_model = DecisionTreeClassifier(max_depth=max_depth, **kwargs)
        elif type == 'linear':
            from sklearn.linear_model import LogisticRegression
            surrogate_model = LogisticRegression(**kwargs)
        else:
            raise TypeError("'type' parameter must be one of {'tree', 'linear'}")
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
    if hasattr(surrogate_model, 'classes_'):
        surrogate_model.class_names = surrogate_model.classes_.astype(str).tolist()
    else:
        surrogate_model.class_names = None

    from .object import Explainer
    surrogate_explainer = Explainer(surrogate_model, X, y_hat, model_type=explainer.model_type, verbose=False)
    surrogate_model_performance = surrogate_explainer.model_performance()
    surrogate_model.performance = surrogate_model_performance.result

    # add the plot method to the surrogate model object
    if type == 'tree':
        # it uses feature_names and class_names if needed
        surrogate_model.plot = types.MethodType(plot_tree_custom, surrogate_model)

    # TODO: add plot method for the linear model

    return surrogate_model


def plot_tree_custom(model, figsize=(16, 10), fontsize=10, filled=True, proportion=True, return_figure=False, **kwargs):
    # wrapper for the plot_tree function, that makes the plot look useful
    # it does not return the plot (because plot_tree works so)
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=figsize)
    
    # return_figure=True makes two plots in IPython
    if return_figure:
        return plot_tree(model,
                    filled=filled,
                    fontsize=fontsize,
                    ax=ax,
                    feature_names=model.feature_names,
                    class_names=model.class_names,
                    proportion=proportion,
                    **kwargs)  
    else:
        _ = plot_tree(model,
                    filled=filled,
                    fontsize=fontsize,
                    ax=ax,
                    feature_names=model.feature_names,
                    class_names=model.class_names,
                    proportion=proportion,
                    **kwargs)     
