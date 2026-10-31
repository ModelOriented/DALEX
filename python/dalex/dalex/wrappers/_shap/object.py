from ..._dataframe import to_numpy, column_names, native_observation, frame_from_values, as_eager_dataframe
import pandas as pd
import numpy as np
import sys
from contextlib import nullcontext
from ..._random import get_random_state, numpy_state_scope
from . import checks


def model_matrix(data, model, shap_type):
    """Honor model-specific tensor boundaries without changing feature backends."""
    values = to_numpy(data)
    if isinstance(data, pd.DataFrame) and values.dtype.kind == "O" and all(
        dtype.kind in "biuf" for dtype in data.dtypes
    ):
        # Nullable pandas/Arrow storage can export numeric features as an object
        # matrix containing pd.NA. SHAP's numeric consumers need NumPy NaN, but
        # keep exact integer storage when no missing value requires promotion.
        missing = pd.isna(values)
        dtypes = [np.dtype(getattr(dtype, "numpy_dtype", dtype)) for dtype in data.dtypes]
        if missing.any():
            dtypes.append(np.dtype("float32"))
            values = np.where(missing, np.nan, values)
        values = values.astype(np.result_type(*dtypes))
    torch = sys.modules.get("torch")
    if shap_type in ("DeepExplainer", "GradientExplainer") and torch is not None and isinstance(model, torch.nn.Module):
        parameter = next(model.parameters(), None)
        if parameter is None:
            parameter = next(model.buffers(), None)
        if parameter is not None:
            return torch.tensor(values, dtype=parameter.dtype, device=parameter.device)
        return torch.tensor(values, dtype=torch.get_default_dtype())
    return values
from ..._explanation import Explanation, CompatibilityTable
from ..._plotting import result_frame


class ShapWrapper(Explanation):
    """Explanation wrapper for the `shap` package

    This object uses the shap package to create the model explanation.
    See ttps://github.com/slundberg/shap

    Parameters
    ----------
    type : {'predict_parts', 'model_parts'}
        Method type for calculations.

    Attributes
    ----------
    result : list or numpy.ndarray
        Calculated shap values for `new_observation` data.
    shap_explainer : {shap.TreeExplainer, shap.DeepExplainer, shap.GradientExplainer, shap.LinearExplainer, shap.KernelExplainer}
        Explainer object from the `shap` package.
    shap_explainer_type : {'TreeExplainer', 'DeepExplainer',
        'GradientExplainer', 'LinearExplainer', 'KernelExplainer'}
        String name of the Explainer class.
    new_observation : pandas.Series or pandas.DataFrame
        Observations for which the shap values will be calculated
        (later stored in `result`).
    type : {'predict_parts', 'model_parts'}
        Method type for calculations.

    Notes
    ----------
    - https://github.com/slundberg/shap
    """

    new_observation = CompatibilityTable()

    def __init__(self, type):
        self.shap_explainer = None
        self.type = type
        self.result = None
        self.new_observation = None
        self.shap_explainer_type = None
        self.expected_value = None
        self._rng_state = None

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
        shap_explainer_type : {'TreeExplainer', 'DeepExplainer', 'GradientExplainer', 'LinearExplainer', 'KernelExplainer'}
            String name of the Explainer class (default is `None`, which automatically
            chooses an Explainer to use).
        kwargs: dict
            Keyword parameters passed to the `shapley_values` method.

        Returns
        -----------
        None
        """
        from shap import TreeExplainer, DeepExplainer, GradientExplainer, LinearExplainer, KernelExplainer

        shap_explainer_type = checks.check_shap_explainer_type(shap_explainer_type, explainer.model)
        # Model-specific SHAP explainers consume the estimator directly. Kernel
        # SHAP uses the explicit native-frame callback below; neither needs a
        # predictor that accepts unlabelled NumPy arrays.

        if self.type == 'predict_parts':
            if isinstance(explainer.data, pd.DataFrame):
                new_observation = checks.check_new_observation_predict_parts(new_observation, explainer)
            else:
                new_observation = native_observation(new_observation, explainer.data, single=True)

        background = model_matrix(explainer.data, explainer.model, shap_explainer_type)
        if shap_explainer_type == "TreeExplainer":
            try:
                self.shap_explainer = TreeExplainer(explainer.model, background)
            except: # https://github.com/ModelOriented/DALEX/issues/371
                self.shap_explainer = TreeExplainer(explainer.model)
        elif shap_explainer_type == "DeepExplainer":
            self.shap_explainer = DeepExplainer(explainer.model, background)
        elif shap_explainer_type == "GradientExplainer":
            model = explainer.model
            torch = sys.modules.get("torch")
            if torch is not None and isinstance(model, torch.nn.Module):
                from ._torch import TorchInputAdapter
                model = TorchInputAdapter(model)
            self.shap_explainer = GradientExplainer(model, background)
        elif shap_explainer_type == "LinearExplainer":
            self.shap_explainer = LinearExplainer(explainer.model, background)
        elif shap_explainer_type == "KernelExplainer":
            self.shap_explainer = KernelExplainer(
                lambda x: explainer.predict(frame_from_values(x, explainer.data)), background
            )

        state = self._rng_state
        if state is None and shap_explainer_type == "GradientExplainer" and kwargs.get("rseed") is not None:
            state = get_random_state(kwargs["rseed"]).get_state()
        rng_scope = numpy_state_scope(state) if shap_explainer_type in ("KernelExplainer", "GradientExplainer") else nullcontext()
        with rng_scope:
            self.result = self.shap_explainer.shap_values(model_matrix(new_observation, explainer.model, shap_explainer_type), **kwargs)
        self.expected_value = getattr(self.shap_explainer, "expected_value", None)
        if self.expected_value is None:
            # GradientExplainer does not expose expected_value, but force plots
            # need the same background-prediction baseline as other explainers.
            self.expected_value = explainer.predict(explainer.data).mean()
        # Preserve typed observation metadata until the legacy pandas attribute
        # is explicitly read. Plotting can consume native arrays directly.
        self.new_observation = (new_observation if isinstance(new_observation, pd.DataFrame)
                                else as_eager_dataframe(new_observation).clone().to_native())
        self.shap_explainer_type = shap_explainer_type

    def plot(self, **kwargs):
        """Plot the Shap Wrapper

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to one of the:
                - shap.force_plot when type is `'predict_parts'`
                - shap.summary_plot when type is `'model_parts'`
            Exceptions are: `base_value`, `shap_values`,
            `features` and `feature_names`.

        Returns
        -----------
        None

        Notes
        --------
        - https://github.com/slundberg/shap
        """
        from shap import force_plot, summary_plot

        observation = result_frame(self, table='new_observation').to_native()
        
        if self.type == 'predict_parts':
            expected_value = getattr(self, "expected_value", None)
            if expected_value is None:
                expected_value = self.shap_explainer.expected_value
            # TensorFlow's DeepExplainer can return an EagerTensor here.
            flat_expected = np.asarray(expected_value).reshape(-1)
            base_value = flat_expected[0 if flat_expected.size == 1 else 1]

            if isinstance(self.result, list):
                shap_values = self.result[1] 
            elif isinstance(self.result, np.ndarray):
                if len(self.result.shape) == 3:
                    shap_values = self.result[:, :, 0 if self.result.shape[-1] == 1 else 1]
                else:
                    shap_values = self.result

            force_plot(base_value=base_value,
                       shap_values=shap_values,
                       features=to_numpy(observation),
                       feature_names=column_names(observation),
                       matplotlib=True,
                       **kwargs)
        elif self.type == 'model_parts':
            values = self.result
            if isinstance(values, np.ndarray) and values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            if not isinstance(observation, pd.DataFrame):
                kwargs.setdefault('feature_names', column_names(observation))
                observation = to_numpy(observation)
            summary_plot(shap_values=values,
                         features=observation,
                         **kwargs)
