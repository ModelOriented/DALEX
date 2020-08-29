import re


def check_shap_explainer_type(shap_explainer_type, model):
    if shap_explainer_type is not None and not isinstance(shap_explainer_type, str):
        raise ValueError("'shap_explainer_type' must be one of {'TreeExplainer', 'DeepExplainer', 'GradientExplainer', 'LinearExplainer', 'KernelExplainer'}")

    if isinstance(shap_explainer_type, str):
        return shap_explainer_type

    # https://github.com/slundberg/shap/blob/8c18d6e3b56fe6675b04f6bccef47885f843ae43/shap/explainers/pytree.py#L138
    model_type = str(type(model))
    if model_type.endswith("sklearn.ensemble._forest.RandomForestRegressor'>") or\
        model_type.endswith("sklearn.ensemble._forest.RandomForestClassifier'>") or\
        model_type.endswith("xgboost.core.Booster'>") or\
        model_type.endswith("lightgbm.basic.Booster'>"):
        shap_explainer_type = "TreeExplainer"
    elif model_type.endswith("'keras.engine.training.Model'>") or\
            model_type.endswith("nn.Module'>"):
        shap_explainer_type = "DeepExplainer"
    # elif model_type.endswith("keras.engine.sequential.Sequential'>"):
    #     explainer_type = "GradientExplainer"
    elif re.search(".*sklearn\.linear_model.*", model_type):
        shap_explainer_type = "LinearExplainer"
    # else:
    #     raise Exception("Could not determine the proper 'shap' 'explainer_type',"
    #                     " please use parameter 'explainer_type'")
    else:
        shap_explainer_type = "KernelExplainer"

    return shap_explainer_type


def check_compatibility(explainer):
    """Placeholder for more specific checks"""

    if explainer.model_info['arrays_accepted']:
        return True
    else:
        raise TypeError("Model not compatible with 'shap' package")
