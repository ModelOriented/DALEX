import shap
from .checks import *


class ShapWrapper:
    """Explanation wrapper for the 'shap' package

    This object uses the shap package to create the model explanation.
    See ttps://github.com/slundberg/shap

    Parameters
    ----------
    type : {'predict_parts', 'model_parts'}


    Attributes
    ----------
    result : list or numpy.ndarray
        Calculated shap values for `new_observation` data.
    shap_explainer : {shap.TreeExplainer, shap.DeepExplainer,
        shap.GradientExplainer, shap.LinearExplainer, shap.KernelExplainer}
        Explainer object from the 'shap' package.
    shap_explainer_type : {'TreeExplainer', 'DeepExplainer',
        'GradientExplainer', 'LinearExplainer', 'KernelExplainer'}
        String name of the Explainer class.
    new_observation : pandas.Series or pandas.DataFrame
        Observations for which the shap values will be calculated
        (later stored in `result`).
    type : {'predict_parts', 'model_parts'}


    Notes
    ----------
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
        """Calculate the result of explanation

        Fit method makes calculations in place and changes the attributes.

        Parameters
        -----------
        explainer : Explainer object
            Model wrapper created using the Explainer class.
        new_observation : pd.Series or np.ndarray
            An observation for which a prediction needs to be explained.
        shap_explainer_type : {'TreeExplainer', 'DeepExplainer',
            'GradientExplainer', 'LinearExplainer', 'KernelExplainer'}
            String name of the Explainer class (default is None, which automatically
            chooses an Explainer to use).

        Returns
        -----------
        None
        """

        check_compatibility(explainer)
        shap_explainer_type = check_shap_explainer_type(shap_explainer_type, explainer.model)

        if self.type == 'predict_parts':
            new_observation = check_new_observation_predict_parts(new_observation, explainer)

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
        self.new_observation = new_observation
        self.shap_explainer_type = shap_explainer_type

    def plot(self, **kwargs):
        """Plot the Shap Wrapper

        Parameters
        ----------
        kwargs :
            Keyword arguments passed to one of the:
                - shap.force_plot when type is 'predict_parts'
                - shap.summary_plot when type is 'model_parts'
            Exceptions are: `base_value`, `shap_values`,
            `features` and `feature_names`.
            Other parameters: https://github.com/slundberg/shap

        Returns
        -----------
        None

        Notes
        --------
        https://github.com/slundberg/shap
        """

        if self.type == 'predict_parts':
            if isinstance(self.shap_explainer.expected_value, (np.ndarray, list)):
                base_value = self.shap_explainer.expected_value[1]
            else:
                base_value = self.shap_explainer.expected_value

            shap_values = self.result[1] if isinstance(self.result, list) else self.result
            shap.force_plot(base_value=base_value,
                            shap_values=shap_values,
                            features=self.new_observation.values,
                            feature_names=self.new_observation.columns,
                            matplotlib=True,
                            **kwargs)
        elif self.type == 'model_parts':
            shap.summary_plot(shap_values=self.result,
                              features=self.new_observation,
                              **kwargs)
