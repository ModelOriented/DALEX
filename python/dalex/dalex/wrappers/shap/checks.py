import re
from collections.abc import Iterable


def check_explainer_type(explainer_type, model):
    if explainer_type is not None and not isinstance(explainer_type, str):
        raise ValueError("'explainer_type' must be one of {'TreeExplainer', 'DeepExplainer', 'GradientExplainer', 'LinearExplainer', 'KernelExplainer'}")

    if isinstance(explainer_type, str):
        return explainer_type

    # https://github.com/slundberg/shap/blob/8c18d6e3b56fe6675b04f6bccef47885f843ae43/shap/explainers/pytree.py#L138
    model_type = str(type(model))
    if model_type.endswith("sklearn.ensemble.forest.RandomForestRegressor'>") or\
        model_type.endswith("sklearn.ensemble.forest.RandomForestClassifier'>") or\
        model_type.endswith("xgboost.core.Booster'>") or\
        model_type.endswith("lightgbm.basic.Booster'>"):
        explainer_type = "TreeExplainer"
    elif model_type.endswith("'keras.engine.training.Model'>") or\
            model_type.endswith("nn.Module'>"):
        explainer_type = "DeepExplainer"
    # elif model_type.endswith("keras.engine.sequential.Sequential'>"):
    #     explainer_type = "GradientExplainer"
    elif re.search(".*sklearn\.linear_model.*", model_type):
        explainer_type = "LinearExplainer"
    # else:
    #     raise Exception("Could not determine the proper 'shap' 'explainer_type',"
    #                     " please use parameter 'explainer_type'")
    else:
        explainer_type = "KernelExplainer"

    return explainer_type


def check_compatibility(explainer):
    """Placeholder for more specific checks"""

    if explainer.model_info['arrays_accepted']:
        return True
    else:
        raise TypeError("Model not compatible with 'shap' package")
