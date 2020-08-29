import shap
from .checks import *
from copy import deepcopy


class ShapWrapper:
    """Calculate instance level variable attributions as Shapley Values

    Parameters
    ----------
    type : str
        {'predict_parts', 'model_parts'}

    Attributes
    ----------
    result : list or numpy.ndarray
        list in case of multiple outputs
    shap_explainer_type : 'shap' package explainer
        explainer type from 'shap' package, {'TreeExplainer', 'DeepExplainer', 'GradientExplainer', 'LinearExplainer', 'KernelExplainer'}
    new_observation : pandas.Series or pandas.DataFrame
        observations for which shap values will be calculated
    type : str
        {'predict_parts', 'model_parts'}

    https://github.com/slundberg/shap
    """

    def __init__(self, type):
        self.shap_explainer = None
        self.type = type
        self.result = None
        self.new_observation = None
        self.shap_explainer_type = None

    def fit(self,
            explainer,
            new_observation,
            shap_explainer_type=None,
            **kwargs):

        check_compatibility(explainer)

        shap_explainer_type = check_shap_explainer_type(shap_explainer_type, explainer.model)

        if shap_explainer_type == "TreeExplainer":
            self.shap_explainer = shap.TreeExplainer(explainer.model, explainer.data.values)
        elif shap_explainer_type == "DeepExplainer":
            self.shap_explainer = shap.DeepExplainer(explainer.model, explainer.data.values)
        elif shap_explainer_type == "GradientExplainer":
            self.shap_explainer = shap.GradientExplainer(explainer.model, explainer.data.values)
        elif shap_explainer_type == "LinearExplainer":
            self.shap_explainer = shap.LinearExplainer(explainer.model, explainer.data.values)
        elif shap_explainer_type == "KernelExplainer":
            self.shap_explainer = shap.KernelExplainer(
                lambda x: explainer.predict(x), explainer.data.values)

        self.result = self.shap_explainer.shap_values(new_observation.values, **kwargs)
        self.new_observation = deepcopy(new_observation)
        self.shap_explainer_type = shap_explainer_type

    def plot(self, **kwargs):
        """Plot the shapley values

        Parameters
        ----------
        **kwargs : key word arguments
            key word arguments passed to 'shap's 'force_plot' in case of
            type == 'predict_parts' or passed to 'shap's 'summary_plot' in case of
            type == 'model_parts'

        https://github.com/slundberg/shap
        """

        if self.type == 'predict_parts':
            shap.force_plot(self.shap_explainer.expected_value[1],
                            self.result[1] if isinstance(self.result, list) else self.result,
                            self.new_observation.values,
                            matplotlib=True,
                            **kwargs)
        elif self.type == 'model_parts':
            shap.summary_plot(self.result,
                              self.new_observation,
                              **kwargs)
