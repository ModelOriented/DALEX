import re
import numpy as np
from copy import deepcopy
import pandas as pd


def check_shap_explainer_type(shap_explainer_type, model):
    if shap_explainer_type is not None and not isinstance(shap_explainer_type, str):
        raise ValueError("'shap_explainer_type' parameter in  must be one of {'TreeExplainer', 'DeepExplainer',"
                         " 'GradientExplainer', 'LinearExplainer', 'KernelExplainer'}")

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
    elif re.search(".*sklearn\\.linear_model.*", model_type):
        shap_explainer_type = "LinearExplainer"
    # else:
    #     raise Exception("Could not determine the proper '_shap' 'explainer_type',"
    #                     " please use parameter 'explainer_type'")
    else:
        shap_explainer_type = "KernelExplainer"

    return shap_explainer_type


def check_compatibility(explainer):
    """Placeholder for more specific checks"""

    if 'arrays_accepted' in explainer.model_info:
        if not explainer.model_info['arrays_accepted']:
            raise TypeError("'predict_function' not compatible with the 'shap' package")
    else:
        # data was None but now is available
        try:
            data_values = explainer.data.values[[0]]
            explainer.predict(data_values)
            explainer.model_info['arrays_accepted'] = True
        except:
            raise TypeError("'predict_function' not compatible with the 'shap' package")


def check_new_observation_predict_parts(new_observation, explainer):
    new_observation_ = deepcopy(new_observation)
    if isinstance(new_observation_, pd.Series):
        new_observation_ = new_observation_.to_frame().T
        new_observation_.columns = explainer.data.columns
    elif isinstance(new_observation_, np.ndarray):
        if new_observation_.ndim == 1:
            # make 1D array 2D
            new_observation_ = new_observation_.reshape((1, -1))
        elif new_observation_.ndim > 2:
            raise ValueError("Wrong new_observation dimension")

        elif new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_ = pd.DataFrame(new_observation_)
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, list):
        new_observation_ = pd.DataFrame(new_observation_).T
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, pd.DataFrame):
        if new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_.columns = explainer.data.columns
    else:
        raise TypeError("new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame")

    if pd.api.types.is_bool_dtype(new_observation_.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation_
