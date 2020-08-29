import shap
from .checks import *
from copy import deepcopy


class ShapWrapper:
    def __init__(self, type):
        self.shap_explainer = None
        self.type = type
        self.result = None
        self.new_observation = None
        self.explainer_type = None

    def fit(self,
            explainer,
            new_observation,
            explainer_type=None,
            **kwargs):

        check_compatibility(explainer)

        explainer_type = check_explainer_type(explainer_type, explainer.model)

        if explainer_type == "TreeExplainer":
            self.shap_explainer = shap.TreeExplainer(explainer.model, explainer.data.values)
        elif explainer_type == "DeepExplainer":
            self.shap_explainer = shap.DeepExplainer(explainer.model, explainer.data.values)
        elif explainer_type == "GradientExplainer":
            self.shap_explainer = shap.GradientExplainer(explainer.model, explainer.data.values)
        elif explainer_type == "LinearExplainer":
            self.shap_explainer = shap.LinearExplainer(explainer.model, explainer.data.values)
        elif explainer_type == "KernelExplainer":
            self.shap_explainer = shap.KernelExplainer(
                lambda x: explainer.predict(x), explainer.data.values)

        self.result = self.shap_explainer.shap_values(new_observation.values, **kwargs)
        self.new_observation = deepcopy(new_observation)
        self.explainer_type = explainer_type

    def plot(self, **kwargs):
        if self.type == 'shap':
            shap.force_plot(self.shap_explainer.expected_value,
                            self.result,
                            self.new_observation,
                            matplotlib=True,
                            **kwargs)
        elif self.type == 'feature_importance':
            shap.summary_plot(self.result,
                              self.new_observation,
                              **kwargs)
