import re

import numpy as np


def verbose_cat(text, verbose):
    if verbose:
        print(text)


def is_y_in_data(data, y):
    return (np.apply_along_axis(lambda x, y: (x == y).all(), 0, data, y)).any()


def get_model_info(model):
    model_package = re.search("(?<=<class ').*?(?=\.)", str(type(model)))[0]
    return {'model_package': model_package}


def unpack_kwargs_lime(explainer, new_observation, **kwargs):
    # helper function for predict_surrogate(type='lime')
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
