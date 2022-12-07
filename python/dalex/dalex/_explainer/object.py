import numpy as np

from dalex.model_explanations import ModelPerformance, VariableImportance, \
    AggregatedProfiles, ResidualDiagnostics
from dalex.predict_explanations import BreakDown, Shap, CeterisParibus, UnbiasedKernelShap
from dalex.wrappers import ShapWrapper
from dalex.fairness import GroupFairnessClassification, GroupFairnessRegression

from . import checks
from . import utils
from . import helper
from .. import _global_checks


class Explainer:
    """ Create Model Explainer

    Black-box models may have very different structures. This class creates a unified
    representation of a model, which can be further processed by various explanations.
    Methods of this class produce explanation objects, that contain the main result
    attribute, and can be visualised using the plot method.

    The `model` is the only required parameter, but most of the explanations require
    that other parameters are provided (See `data`, `y`, `predict_function`, `model_type`).

    Parameters
    ----------
    model : object
        Model to be explained.
    data : pd.DataFrame or np.ndarray (2d)
        Data which will be used to calculate the explanations. It shouldn't contain
        the target column (See `y`).
        NOTE: If target variable is present in the data, some of the functionalities may
        not work properly.
    y : pd.Series or pd.DataFrame or np.ndarray (1d)
        Target variable with outputs / scores. It shall have the same length as `data`.
    predict_function : function, optional
        Function that takes two parameters (model, data) and returns a np.ndarray (1d)
        with model predictions (default is predict method extracted from the model).
        NOTE: This function needs to work with `data` as pd.DataFrame.
    residual_function : function, optional
        Function that takes three parameters (model, data, y) and returns a np.ndarray (1d)
        with model residuals (default is a function constructed from `predict_function`).
    weights : pd.Series or np.ndarray (1d), optional
        Sampling weights for observations in `data`. It shall have the same length as
        `data` (default is `None`).
    label : str, optional
        Model name to appear in result and plots
        (default is last element of the class attribute extracted from the model).
    model_class : str, optional
        Class of the model that is used e.g. to choose the `predict_function`
        (default is the class attribute extracted from the model).
        NOTE: Use if your model is wrapped with Pipeline.
    verbose : bool
        Print diagnostic messages during the Explainer initialization (default is `True`).
    precalculate : bool
        Calculate y_hat (predicted values) and residuals during the Explainer
        initialization (default is `True`).
    model_type : {'regression', 'classification', None}
        Model task type that is used e.g. in `model_performance()` and `model_parts()`
        (default is try to extract the information from the model, else `None`).
    model_info: dict, optional
        Dict `{'model_package', 'model_package_version', ...}` containing additional
        information to be stored.

    Attributes
    --------
    model : object
        A model to be explained.
    data : pd.DataFrame
        Data which will be used to calculate the explanations.
    y : np.ndarray (1d)
        Target variable with outputs / scores.
    predict_function : function
        Function that takes two arguments (model, data) and returns np.ndarray (1d)
        with model predictions.
    y_hat : np.ndarray (1d)
        Model predictions for `data`.
    residual_function : function
        Function that takes three arguments (model, data, y) and returns np.ndarray (1d)
        with model residuals.
    residuals : np.ndarray (1d)
        Model residuals for `data`.
    weights : np.ndarray (1d)
        Sampling weights for observations in `data`.
    label : str
        Name to appear in result and plots.
    model_class : str
        Class of the model.
    model_type : {'regression', 'classification', `None`}
        Model task type.
    model_info: dict
        Dict `{'model_package', 'model_package_version', ...}` containing additional
        information.

    Notes
    --------
    - https://pbiecek.github.io/ema/dataSetsIntro.html#ExplainersTitanicPythonCode

    """

    def __init__(self,
                 model,
                 data=None,
                 y=None,
                 predict_function=None,
                 residual_function=None,
                 weights=None,
                 label=None,
                 model_class=None,
                 verbose=True,
                 precalculate=True,
                 model_type=None,
                 model_info=None):

        # TODO: colorize
        
        helper.verbose_cat("Preparation of a new explainer is initiated\n", verbose=verbose)

        # REPORT: checks for data
        data, model = checks.check_data(data, model, verbose)

        # REPORT: checks for y
        y = checks.check_y(y, data, verbose)

        # REPORT: checks for weights
        weights = checks.check_weights(weights, data, verbose)

        # REPORT: checks for model_class
        model_class, _model_info = checks.check_model_class(model_class, model, verbose)

        # REPORT: checks for label
        label, _model_info = checks.check_label(label, model_class, _model_info, verbose)

        # REPORT: checks for predict_function and model_type
        # these two are together only because of `yhat_exception_dict`
        predict_function, model_type, y_hat, _model_info = \
            checks.check_predict_function_and_model_type(predict_function, model_type,
                                                         model, data, model_class, _model_info,
                                                         precalculate, verbose)

        # if data is specified then we may test predict_function
        # at this moment we have predict function

        # REPORT: checks for residual_function
        residual_function, residuals, _model_info = checks.check_residual_function(
            residual_function, predict_function, model, data, y, _model_info, precalculate, verbose
        )

        # REPORT: checks for model_info
        _model_info = checks.check_model_info(model_info, _model_info, verbose)

        # READY to create an explainer
        self.model = model
        self.data = data
        self.y = y
        self.predict_function = predict_function
        self.y_hat = y_hat
        self.residual_function = residual_function
        self.residuals = residuals
        self.model_class = model_class
        self.label = label
        self.model_info = _model_info
        self.weights = weights
        self.model_type = model_type

        helper.verbose_cat("\nA new explainer has been created!", verbose=verbose)

    def predict(self, data):
        """Make a prediction

        This function uses the `predict_function` attribute.

        Parameters
        ----------
        data : pd.DataFrame, np.ndarray (2d)
            Data which will be used to make a prediction.

        Returns
        ----------
        np.ndarray (1d)
            Model predictions for given `data`.
        """

        checks.check_method_data(data)

        return self.predict_function(self.model, data)

    def residual(self, data, y):
        """Calculate residuals

        This function uses the `residual_function` attribute.

        Parameters
        -----------
        data : pd.DataFrame
            Data which will be used to calculate residuals.
        y : pd.Series or np.ndarray (1d)
            Target variable which will be used to calculate residuals.

        Returns
        -----------
        np.ndarray (1d)
            Model residuals for given `data` and `y`.
        """

        checks.check_method_data(data)

        return self.residual_function(self.model, data, y)

    def predict_parts(self,
                      new_observation,
                      type=('break_down_interactions', 'break_down', 'shap', 'shap_wrapper', 'unbiased_kernel_shap'),
                      order=None,
                      interaction_preference=1,
                      path="average",
                      N=None,
                      B=25,
                      keep_distributions=False,
                      label=None,
                      processes=1,
                      random_state=None,
                      n_samples=25,
                      **kwargs):
        """Calculate predict-level variable attributions as Break Down, Shapley Values or Shap Values

        Parameters
        -----------
        new_observation : pd.Series or np.ndarray (1d) or pd.DataFrame (1,p)
            An observation for which a prediction needs to be explained.
        type : {'break_down_interactions', 'break_down', 'shap', 'shap_wrapper'}
            Type of variable attributions (default is `'break_down_interactions'`).
        order : list of int or str, optional
            Parameter specific for `break_down_interactions` and `break_down`. Use a fixed
            order of variables for attribution calculation. Use integer values  or string
            variable names (default is `None`, which means order by importance).
        interaction_preference : int, optional
            Parameter specific for `break_down_interactions` type. Specify which interactions
            will be present in an explanation. The larger the integer, the more frequently
            interactions will be presented (default is `1`).
        path : list of int, optional
            Parameter specific for `shap`. If specified, then attributions for this path
            will be plotted (default is `'average'`, which plots attribution means for
            `B` random paths).
        N : int, optional
            Number of observations that will be sampled from the `data` attribute before
            the calculation of variable attributions. Default is `None` which means all `data`.
        B : int, optional
            Parameter specific for `shap`. Number of random paths to calculate
            variable attributions (default is `25`).
        keep_distributions :  bool, optional
            Save the distribution of partial predictions (default is `False`).
        label : str, optional
            Name to appear in result and plots. Overrides default.
        processes : int, optional
            Parameter specific for `shap`. Number of parallel processes to use in calculations.
            Iterated over `B` (default is `1`, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).
        kwargs : dict
            Used only for `'shap_wrapper'`. Pass `shap_explainer_type` to specify, which
            Explainer shall be used: `{'TreeExplainer', 'DeepExplainer', 'GradientExplainer',
            'LinearExplainer', 'KernelExplainer'}` (default is `None`, which automatically
            chooses an Explainer to use).
            Also keyword arguments passed to one of the: `shap.TreeExplainer.shap_values,
            shap.DeepExplainer.shap_values, shap.GradientExplainer.shap_values,
            shap.LinearExplainer.shap_values, shap.KernelExplainer.shap_values`.
            See https://github.com/slundberg/shap

        Returns
        -----------
        BreakDown, Shap or ShapWrapper class object
            Explanation object containing the main result attribute and the plot method.
            Object class, its attributes, and the plot method depend on the `type` parameter.

        Notes
        --------
        - https://pbiecek.github.io/ema/breakDown.html
        - https://pbiecek.github.io/ema/iBreakDown.html
        - https://pbiecek.github.io/ema/shapley.html
        - https://github.com/slundberg/shap
        """

        checks.check_data_again(self.data)

        types = ('break_down_interactions', 'break_down', 'shap', 'shap_wrapper', 'unbiased_kernel_shap')
        _type = checks.check_method_type(type, types)

        if isinstance(N, int):
            # temporarly overwrite data in the Explainer (fastest way)
            # at the end of predict_parts fix the Explainer (add original data)
            if isinstance(random_state, int):
                np.random.seed(random_state)
            N = min(N, self.data.shape[0])
            I = np.random.choice(np.arange(self.data.shape[0]), N, replace=False)
            from copy import deepcopy
            _data = deepcopy(self.data)
            self.data = self.data.iloc[I, :]

        if _type == 'break_down_interactions' or _type == 'break_down':
            _predict_parts = BreakDown(
                type=_type,
                keep_distributions=keep_distributions,
                order=order,
                interaction_preference=interaction_preference
            )
        elif _type == 'shap':
            _predict_parts = Shap(
                keep_distributions=keep_distributions,
                path=path,
                B=B,
                processes=processes,
                random_state=random_state
            )
        elif _type == 'shap_wrapper':
            _global_checks.global_check_import('shap', 'SHAP explanations')
            _predict_parts = ShapWrapper('predict_parts')
        elif _type == 'unbiased_kernel_shap':
            _predict_parts = UnbiasedKernelShap(
                keep_distributions=keep_distributions,
                processes=processes,
                random_state=random_state,
                n_samples=n_samples
            )
        else:
            raise TypeError("Wrong type parameter.")

        _predict_parts.fit(self, new_observation, **kwargs)
        
        if label:
            _predict_parts.result['label'] = label

        if isinstance(N, int):
            self.data = _data

        return _predict_parts

    def predict_profile(self,
                        new_observation,
                        type=('ceteris_paribus',),
                        y=None,
                        variables=None,
                        grid_points=101,
                        variable_splits=None,
                        variable_splits_type='uniform',
                        variable_splits_with_obs=True,
                        processes=1,
                        label=None,
                        verbose=True):
        """Calculate predict-level variable profiles as Ceteris Paribus

        Parameters
        -----------
        new_observation : pd.DataFrame or np.ndarray or pd.Series
            Observations for which predictions need to be explained.
        type : {'ceteris_paribus', TODO: 'oscilations'}
            Type of variable profiles (default is `'ceteris_paribus'`).
        y : pd.Series or np.ndarray (1d), optional
            Target variable with the same length as `new_observation`.
        variables : str or array_like of str, optional
            Variables for which the profiles will be calculated
            (default is `None`, which means all of the variables).
        grid_points : int, optional
            Maximum number of points for profile calculations (default is `101`).
            NOTE: The final number of points may be lower than `grid_points`,
            eg. if there is not enough unique values for a given variable.
        variable_splits : dict of lists, optional
            Split points for variables e.g. `{'x': [0, 0.2, 0.5, 0.8, 1], 'y': ['a', 'b']}`
            (default is `None`, which means that they will be calculated using one of
            `variable_splits_type` and the `data` attribute).
        variable_splits_type : {'uniform', 'quantiles'}, optional
            Way of calculating `variable_splits`. Set `'quantiles'` for percentiles.
            (default is `'uniform'`, which means uniform grid of points).
        variable_splits_with_obs: bool, optional
            Add variable values of `new_observation` data to the `variable_splits`
            (default is `True`).
        label : str, optional
            Name to appear in result and plots. Overrides default.
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `variables`
            (default is `1`, which means no parallel computation).
        verbose : bool, optional
            Print tqdm progress bar (default is `True`).

        Returns
        -----------
        CeterisParibus class object
            Explanation object containing the main result attribute and the plot method.

        Notes
        --------
        - https://pbiecek.github.io/ema/ceterisParibus.html
        """

        checks.check_data_again(self.data)

        types = ('ceteris_paribus',)
        _type = checks.check_method_type(type, types)

        if _type == 'ceteris_paribus':
            _predict_profile = CeterisParibus(
                variables=variables,
                grid_points=grid_points,
                variable_splits=variable_splits,
                variable_splits_type=variable_splits_type,
                variable_splits_with_obs=variable_splits_with_obs,
                processes=processes
            )
        else:
            raise TypeError("Wrong type parameter.")

        _predict_profile.fit(self, new_observation, y, verbose)

        if label:
            _predict_profile.result['_label_'] = label
            
        return _predict_profile

    def predict_surrogate(self,
                          new_observation,
                          type='lime',
                          **kwargs):
        """Wrapper for surrogate model explanations

        This function uses the lime package to create the model explanation.
        See https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular

        Parameters
        -----------
        new_observation : pd.Series or np.ndarray (1d) or pd.DataFrame (1,p)
            An observation for which a prediction needs to be explained.
        type : {'lime'}
            Type of explanation method
            (default is `'lime'`, which uses the lime package to create an explanation).
        kwargs : dict
            Keyword arguments passed to the lime.lime_tabular.LimeTabularExplainer object
            and the LimeTabularExplainer.explain_instance method. Exceptions are:
            `training_data`, `mode`, `data_row` and `predict_fn`. Other parameters:
            https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular

        Returns
        -----------
        lime.explanation.Explanation
            Explanation object.

        Notes
        -----------
        - https://github.com/marcotcr/lime
        """

        checks.check_data_again(self.data)

        if type == 'lime':
            _global_checks.global_check_import('lime', 'LIME explanations')
            _new_observation = checks.check_new_observation_lime(new_observation)
            _explanation = utils.create_lime_explanation(self, _new_observation, **kwargs)
        else:
            raise TypeError("Wrong 'type' parameter.")

        return _explanation

    def model_performance(self,
                          model_type=None,
                          cutoff=0.5,
                          label=None):
        """Calculate model-level model performance measures

        Parameters
        -----------
        model_type : {'regression', 'classification', None}
            Model task type that is used to choose the proper performance measures
            (default is `None`, which means try to extract from the `model_type` attribute).
        cutoff : float, optional
            Cutoff for predictions in classification models. Needed for measures like
            recall, precision, acc, f1 (default is `0.5`).
        label : str, optional
            Name to appear in result and plots. Overrides default.

        Returns
        -----------
        ModelPerformance class object
            Explanation object containing the main result attribute and the plot method.

        Notes
        --------
        - https://pbiecek.github.io/ema/modelPerformance.html
        """

        checks.check_data_again(self.data)
        checks.check_y_again(self.y)

        if model_type is None and self.model_type is None:
            raise TypeError("if self.model_type is None, then model_type must be not None")
        elif model_type is None:
            model_type = self.model_type

        _model_performance = ModelPerformance(
            model_type=model_type,
            cutoff=cutoff
        )
        _model_performance.fit(self)
        
        if label:
            _model_performance.result['label'] = label

        return _model_performance

    def model_parts(self,
                    loss_function=None,
                    type=('variable_importance', 'ratio', 'difference', 'shap_wrapper'),
                    N=1000,
                    B=10,
                    variables=None,
                    variable_groups=None,
                    keep_raw_permutations=True,
                    label=None,
                    processes=1,
                    random_state=None,
                    **kwargs):

        """Calculate model-level variable importance

        Parameters
        -----------
        loss_function : {'rmse', '1-auc', 'mse', 'mae', 'mad'} or function, optional
            If string, then such loss function will be used to assess variable importance
            (default is `'rmse'` or `'1-auc'`, depends on `model_type` attribute).
        type : {'variable_importance', 'ratio', 'difference', 'shap_wrapper'}
            Type of transformation that will be applied to dropout loss.
            (default is `'variable_importance'`, which is Permutational Variable Importance).
        N : int, optional
            Number of observations that will be sampled from the `data` attribute before
            the calculation of variable importance. `None` means all `data` (default is `1000`).
        B : int, optional
            Number of permutation rounds to perform on each variable (default is `10`).
        variables : array_like of str, optional
            Variables for which the importance will be calculated
            (default is `None`, which means all of the variables).
            NOTE: Ignored if `variable_groups` is not `None`.
        variable_groups : dict of lists, optional
            Group the variables to calculate their joint variable importance
            e.g. `{'X': ['x1', 'x2'], 'Y': ['y1', 'y2']}` (default is `None`).
        keep_raw_permutations: bool, optional
            Save results for all permutation rounds (default is `True`).
        label : str, optional
            Name to appear in result and plots. Overrides default.
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `B`
            (default is `1`, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).
        kwargs : dict
            Used only for 'shap_wrapper'. Pass `shap_explainer_type` to specify, which
            Explainer shall be used: `{'TreeExplainer', 'DeepExplainer', 'GradientExplainer',
            'LinearExplainer', 'KernelExplainer'}`.
            Also keyword arguments passed to one of the: `shap.TreeExplainer.shap_values,
            shap.DeepExplainer.shap_values, shap.GradientExplainer.shap_values,
            shap.LinearExplainer.shap_values, shap.KernelExplainer.shap_values`.
            See https://github.com/slundberg/shap

        Returns
        -----------
        VariableImportance or ShapWrapper class object
            Explanation object containing the main result attribute and the plot method.
            Object class, its attributes, and the plot method depend on the `type` parameter.

        Notes
        --------
        - https://pbiecek.github.io/ema/featureImportance.html
        - https://github.com/slundberg/shap
        """

        checks.check_data_again(self.data)

        types = ('variable_importance', 'ratio', 'difference', 'shap_wrapper')
        aliases = {'permutational': 'variable_importance', 'feature_importance': 'variable_importance'}
        _type = checks.check_method_type(type, types, aliases)

        loss_function = checks.check_method_loss_function(self, loss_function)

        if _type != 'shap_wrapper':
            checks.check_y_again(self.y)

            _model_parts = VariableImportance(
                loss_function=loss_function,
                type=_type,
                N=N,
                B=B,
                variables=variables,
                variable_groups=variable_groups,
                processes=processes,
                random_state=random_state,
                keep_raw_permutations=keep_raw_permutations,
            )
            _model_parts.fit(self)
            
            if label:
                _model_parts.result['label'] = label
                 
        elif _type == 'shap_wrapper':
            _global_checks.global_check_import('shap', 'SHAP explanations')
            _model_parts = ShapWrapper('model_parts')
            if isinstance(N, int):
                if isinstance(random_state, int):
                    np.random.seed(random_state)
                N = min(N, self.data.shape[0])
                I = np.random.choice(np.arange(self.data.shape[0]), N, replace=False)
                _new_observation = self.data.iloc[I, :]
            else:
                _new_observation = self.data

            _model_parts.fit(self, _new_observation, **kwargs)
        else:
            raise TypeError("Wrong type parameter");

        return _model_parts

    def model_profile(self,
                      type=('partial', 'accumulated', 'conditional'),
                      N=300,
                      variables=None,
                      variable_type='numerical',
                      groups=None,
                      span=0.25,
                      grid_points=101,
                      variable_splits=None,
                      variable_splits_type='uniform',
                      center=True,
                      label=None,
                      processes=1,
                      random_state=None,
                      verbose=True):

        """Calculate model-level variable profiles as Partial or Accumulated Dependence

        Parameters
        -----------
        type : {'partial', 'accumulated', 'conditional'}
            Type of model profiles
            (default is `'partial'` for Partial Dependence Profiles).
        N : int, optional
            Number of observations that will be sampled from the `data` attribute before
            the calculation of variable profiles. `None` means all `data` (default is `300`).
        variables : str or array_like of str, optional
            Variables for which the profiles will be calculated
            (default is `None`, which means all of the variables).
        variable_type : {'numerical', 'categorical'}
            Calculate the profiles for numerical or categorical variables
            (default is `'numerical'`).
        groups : str or array_like of str, optional
            Names of categorical variables that will be used for profile grouping
            (default is `None`, which means no grouping).
        span : float, optional
            Smoothing coefficient used as sd for gaussian kernel (default is `0.25`).
        grid_points : int, optional
            Maximum number of points for profile calculations (default is `101`).
            NOTE: The final number of points may be lower than `grid_points`,
            e.g. if there is not enough unique values for a given variable.
        variable_splits : dict of lists, optional
            Split points for variables e.g. `{'x': [0, 0.2, 0.5, 0.8, 1], 'y': ['a', 'b']}`
            (default is `None`, which means that they will be distributed uniformly).
        variable_splits_type : {'uniform', 'quantiles'}, optional
            Way of calculating `variable_splits`. Set 'quantiles' for percentiles.
            (default is `'uniform'`, which means uniform grid of points).
        center : bool, optional
            Theoretically Accumulated Profiles start at `0`, but are centered to compare
            them with Partial Dependence Profiles (default is `True`, which means center
            around the average `y_hat` calculated on the data sample).
        label : str, optional
            Name to appear in result and plots. Overrides default.
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `variables`
            (default is `1`, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).
        verbose : bool, optional
            Print tqdm progress bar (default is `True`).

        Returns
        -----------
        AggregatedProfiles class object
            Explanation object containing the main result attribute and the plot method.

        Notes
        --------
        - https://pbiecek.github.io/ema/partialDependenceProfiles.html
        - https://pbiecek.github.io/ema/accumulatedLocalProfiles.html
        """

        checks.check_data_again(self.data)

        types = ('partial', 'accumulated', 'conditional')
        aliases = {'pdp': 'partial', 'ale': 'accumulated'}
        _type = checks.check_method_type(type, types, aliases)

        _ceteris_paribus = CeterisParibus(
            grid_points=grid_points,
            variables=variables,
            variable_splits=variable_splits,
            variable_splits_type=variable_splits_type,
            processes=processes
        )

        if isinstance(N, int):
            if isinstance(random_state, int):
                np.random.seed(random_state)
            N = min(N, self.data.shape[0])
            I = np.random.choice(np.arange(self.data.shape[0]), N, replace=False)
            _y = self.y[I] if self.y is not None else self.y
            _new_observation = self.data.iloc[I, :]
        else:
            _y = self.y
            _new_observation = self.data

        _ceteris_paribus.fit(self, _new_observation, _y, verbose=verbose)

        _model_profile = AggregatedProfiles(
            type=_type,
            variables=variables,
            variable_type=variable_type,
            groups=groups,
            span=span,
            center=center,
            random_state=random_state
        )

        _model_profile.fit(_ceteris_paribus, verbose)

        if label:
            _model_profile.result['_label_'] = label
                
        return _model_profile

    def model_diagnostics(self,
                          variables=None,
                          label=None):
        """Calculate model-level residuals diagnostics

        Parameters
        -----------
        variables : str or array_like of str, optional
            Variables for which the data will be calculated
            (default is `None`, which means all of the variables).
        label : str, optional
            Name to appear in result and plots. Overrides default.

        Returns
        -----------
        ResidualDiagnostics class object
            Explanation object containing the main result attribute and the plot method.

        Notes
        --------
        - https://pbiecek.github.io/ema/residualDiagnostic.html
        """

        checks.check_data_again(self.data)
        checks.check_y_again(self.y)

        _residual_diagnostics = ResidualDiagnostics(
            variables=variables
        )
        _residual_diagnostics.fit(self)

        if label:
            _residual_diagnostics.result['label'] = label
            
        return _residual_diagnostics

    def model_surrogate(self,
                        type=('tree', 'linear'),
                        max_vars=5,
                        max_depth=3,
                        **kwargs):
        """Create a surrogate interpretable model from the black-box model

        This method uses the scikit-learn package to create a surrogate
        interpretable model (e.g. decision tree) from the black-box model.
        It aims to use the most important features and add a plot method to
        the model, so that it can be easily interpreted. See Notes section
        for references.

        Parameters
        -----------
        type : {'tree', 'linear'}
            Type of a surrogate model. This can be a decision tree or a linear model
            (default is `'tree'`).
        max_vars : int, optional
            Maximum number of variables that will be used in surrogate model training.
            These are the most important variables to the black-box model (default is `5`).
        max_depth : int, optional
            The maximum depth of the tree. If `None`, then nodes are expanded until all
            leaves are pure or until all leaves contain less than min_samples_split
            samples (default is `3` for interpretable plot).
        kwargs : dict
            Keyword arguments passed to one of the: `sklearn.tree.DecisionTreeClassifier,
            sklearn.tree.DecisionTreeRegressor, sklearn.linear_model.LogisticRegression,
            sklearn.linear_model.LinearRegression`


        Returns
        -----------
        One of: sklearn.tree.DecisionTreeClassifier, sklearn.tree.DecisionTreeRegressor, sklearn.linear_model.LogisticRegression, sklearn.linear_model.LinearRegression
        A surrogate model with additional:
            - `plot` method
            - `performance` attribute
            - `feature_names` attribute
            - `class_names` attribute

        Notes
        -----------
        - https://christophm.github.io/interpretable-ml-book/global.html
        - https://github.com/scikit-learn/scikit-learn
        """

        _global_checks.global_check_import('scikit-learn', 'surrogate models')
        checks.check_data_again(self.data)

        types = ('tree', 'linear')
        _type = checks.check_method_type(type, types)

        surrogate_model = utils.create_surrogate_model(explainer=self,
                                                       type=_type,
                                                       max_vars=max_vars,
                                                       max_depth=max_depth,
                                                       **kwargs)

        return surrogate_model

    def model_fairness(self,
                       protected,
                       privileged,
                       cutoff=0.5,
                       epsilon=0.8,
                       label=None,
                       **kwargs):
        """Creates a model-level fairness explanation that enables bias detection

        This method returns a GroupFairnessClassification or a GroupFairnessRegression
        object depending of the type of predictor. They work as a wrapper of the
        protected attribute and the Explainer from which `y` and `y_hat`
        attributes were extracted. Along with an information about
        privileged subgroup (value in the `protected` parameter), those 3 vectors
        create triplet `(y, y_hat, protected)` which is a base for all further
        fairness calculations and visualizations.

        The GroupFairnessRegression should be treated as experimental tool.
        It was implemented according to Fairness Measures for Regression via
        Probabilistic Classification - Steinberg et al. (2020).

        Parameters
        -----------
        protected : np.ndarray (1d)
            Vector, preferably 1-dimensional np.ndarray containing strings,
            which denotes the membership to a subgroup. It doesn't have to be binary.
            It doesn't need to be in data. It is sometimes suggested not to use
            sensitive attributes in modelling, but still check model bias for them.
            NOTE: List and pd.Series are also supported; however, if provided,
            they will be transformed into a np.ndarray (1d) with dtype 'U'.
        privileged : str
            Subgroup that is suspected to have the most privilege.
            It needs to be a string present in `protected`.
        cutoff : float or dict, optional
            Only for classification models.
            Threshold for probabilistic output of a classifier.
            It might be: a `float` - same for all subgroups from `protected`,
            or a `dict` - individually adjusted for each subgroup;
            must have values from `protected` as keys.
        epsilon : float
            Parameter defines acceptable fairness scores. The closer to `1` the
            more strict the verdict is. If the ratio of certain unprivileged
            and privileged subgroup is within the `(epsilon, 1/epsilon)` range,
            then there is no discrimination in this metric and for this subgroups
            (default is `0.8`).
        label : str, optional
            Name to appear in result and plots. Overrides default.
        kwargs : dict
            Keyword arguments. It supports `verbose`, which is a boolean
            value telling if additional output should be printed
            (`True`) or not (`False`, default).

        Returns
        -----------
        GroupFairnessClassification class object (a subclass of _FairnessObject)
            Explanation object containing the main result attribute and the plot method.
            
        It has the following main attributes:
            - result : `pd.DataFrame`
                Scaled `metric_scores`. The scaling is performed by
                dividing all metric scores by scores of the privileged subgroup.
            - metric_scores : `pd.DataFrame`
                Raw metric scores for each subgroup.
            - parity_loss : `pd.Series`
                It is a summarised `result`. From each metric (column) a logarithm
                is calculated, then the absolute value is taken and summarised.
                Therefore, for metric M:
                    `parity_loss` is a `sum(abs(log(M_i / M_privileged)))`
                        where `M_i` is the metric score for subgroup `i`.
            - label : `str`
                `label` attribute from the Explainer object.
                    Labels must be unique when plotting.
            - cutoff : `dict`
                A float value for each subgroup (key in dict).

        Notes
        -----------
        - Verma, S. & Rubin, J. (2018) https://fairware.cs.umass.edu/papers/Verma.pdf
        - Zafar, M.B., et al. (2017) https://arxiv.org/pdf/1610.08452.pdf
        - Hardt, M., et al. (2016) https://arxiv.org/pdf/1610.02413.pdf
        - Steinberg, D., et al. (2020) https://arxiv.org/pdf/2001.06089.pdf
        """

        if self.model_type == 'classification':
            fobject = GroupFairnessClassification(y=self.y,
                                                  y_hat=self.y_hat,
                                                  protected=protected,
                                                  privileged=privileged,
                                                  cutoff=cutoff,
                                                  epsilon=epsilon,
                                                  label=self.label,
                                                  **kwargs)

        elif self.model_type == 'regression':
            fobject = GroupFairnessRegression(y=self.y,
                                              y_hat=self.y_hat,
                                              protected=protected,
                                              privileged=privileged,
                                              epsilon=epsilon,
                                              label=self.label,
                                              **kwargs)

        else :
            raise ValueError("'model_type' must be either 'classification' or 'regression'")

        if label:
             fobject.label = label

        return fobject

    def dumps(self, *args, **kwargs):
        """Return the pickled representation (bytes object) of the Explainer

        This method uses the pickle package. See
        https://docs.python.org/3/library/pickle.html#pickle.dumps

        NOTE: local functions and lambdas cannot be pickled.
        Attribute `residual_function` by default contains lambda; thus,
        if not provided by the user, it will be dropped before the dump.

        Parameters
        -----------
        args : dict
            Positional arguments passed to the pickle.dumps function.
        kwargs : dict
            Keyword arguments passed to the pickle.dumps function.

        Returns
        -----------
        bytes object
        """

        from copy import deepcopy
        to_dump = deepcopy(self)
        to_dump = checks.check_if_local_and_lambda(to_dump)

        import pickle
        return pickle.dumps(to_dump, *args, **kwargs)

    def dump(self, file, *args, **kwargs):
        """Write the pickled representation of the Explainer to the file (pickle)

        This method uses the pickle package. See
        https://docs.python.org/3/library/pickle.html#pickle.dump

        NOTE: local functions and lambdas cannot be pickled.
        Attribute `residual_function` by default contains lambda; thus,
        if not provided by the user, it will be dropped before the dump.

        Parameters
        -----------
        file : ...
            A file object opened for binary writing, or an io.BytesIO instance.
        args : dict
            Positional arguments passed to the pickle.dump function.
        kwargs : dict
            Keyword arguments passed to the pickle.dump function.
        """

        from copy import deepcopy
        to_dump = deepcopy(self)
        to_dump = checks.check_if_local_and_lambda(to_dump)

        import pickle
        return pickle.dump(to_dump, file, *args, **kwargs)

    @staticmethod
    def loads(data, use_defaults=True, *args, **kwargs):
        """Load the Explainer from the pickled representation (bytes object)

        This method uses the pickle package. See
        https://docs.python.org/3/library/pickle.html#pickle.loads

        NOTE: local functions and lambdas cannot be pickled.
        If `use_defaults` is set to `True`, then dropped functions are set to defaults.

        Parameters
        -----------
        data : bytes object
            Binary representation of the Explainer.
        use_defaults : bool
            Replace empty `predict_function` and `residual_function` with default
            values like in Explainer initialization (default is `True`).
        args : dict
            Positional arguments passed to the pickle.loads function.
        kwargs : dict
            Keyword arguments passed to the pickle.loads function.

        Returns
        -----------
        Explainer object
        """

        import pickle
        exp = pickle.loads(data, *args, **kwargs)

        if use_defaults:
            exp = checks.check_if_empty_fields(exp)

        return exp

    @staticmethod
    def load(file, use_defaults=True, *args, **kwargs):
        """Read the pickled representation of the Explainer from the file (pickle)

        This method uses the pickle package. See
        https://docs.python.org/3/library/pickle.html#pickle.load

        NOTE: local functions and lambdas cannot be pickled.
        If `use_defaults` is set to `True`, then dropped functions are set to defaults.

        Parameters
        -----------
        file : ...
            A binary file object opened for reading, or an io.BytesIO object.
        use_defaults : bool
            Replace empty `predict_function` and `residual_function` with default
            values like in Explainer initialization (default is `True`).
        args : dict
            Positional arguments passed to the pickle.load function.
        kwargs : dict
            Keyword arguments passed to the pickle.load function.

        Returns
        -----------
        Explainer object
        """

        import pickle
        exp = pickle.load(file, *args, **kwargs)

        if use_defaults:
            exp = checks.check_if_empty_fields(exp)

        return exp
