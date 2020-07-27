from dalex.dataset_level import ModelPerformance, VariableImportance, AggregatedProfiles
from dalex.instance_level import BreakDown, Shap, CeterisParibus
from .checks import *
from .helper import get_model_info


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
    y : pd.Series or np.ndarray (1d)
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
        `data` (default is None).
    label : str, optional
        Model name to appear in result and plots
        (default is last element of the class attribute extracted from the model).
    model_class : str, optional
        Class of the model that is used e.g. to choose the `predict_function`
        (default is the class attribute extracted from the model).
        NOTE: Use if your model is wrapped with Pipeline.
    verbose : bool
        Print diagnostic messages during the Explainer initialization (default is True).
    precalculate : bool
        Calculate y_hat (predicted values) and residuals during the Explainer
        initialization (default is True).
    model_type : {'regression', 'classification', None}
        Model task type that is used e.g. in `model_performance` and `model_parts`
        (default is try to extract the information from the model, else None).
    model_info: dict, optional
        Dict {'model_package', 'model_package_version', ...} containing additional
        information to be stored.
    colorize : TODO

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
        Model name to appear in result and plots.
    model_class : str
        Class of the model.
    model_type : {'regression', 'classification', None}
        Model task type.
    model_info: dict
        Dict {'model_package', 'model_package_version', ...} containing additional
        information.

    Notes
    --------
    https://pbiecek.github.io/ema/dataSetsIntro.html#ExplainersTitanicPythonCode

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
                 model_info=None,
                 colorize=True):

        verbose_cat("Preparation of a new explainer is initiated\n", verbose=verbose)

        # if requested, remove colors
        if not colorize:
            color_codes = {
                'yellow_start': "",
                'yellow_end': "",
                'red_start': "",
                'red_end': "",
                'green_start': "",
                'green_end': ""}

        # REPORT: checks for data
        """
        Contrary to the R package, data cannot be retrieved from model, thus data is necessary.
        If data is None, tell user about the lack of the data.
        """

        data = check_data(data, verbose)

        # REPORT: checks for y
        y = check_y(y, data, verbose)

        # REPORT: checks for weights
        weights = check_weights(weights, data, verbose)

        model_info_ = get_model_info(model)

        model_class, model_info_ = check_model_class(model_class, model_info_, model, verbose)

        label, model_info_ = check_label(label, model_class, model_info_, verbose)

        # REPORT: checks for predict_function
        predict_function, pred, model_info_, model_type_ = check_predict_function(predict_function, model, data,
                                                                                  model_class, model_info_,
                                                                                  precalculate, verbose)

        model_type = check_model_type(model_type, model_type_)
        # if data is specified then we may test predict_function
        # at this moment we have predict function

        # REPORT: checks for residual_function
        residual_function, residuals, model_info_ = check_residual_function(residual_function, predict_function,
                                                                            model, data, y,
                                                                            model_info_, precalculate, verbose)

        model_info = check_model_info(model_info, model_info_, verbose)

        # READY to create an explainer
        self.model = model
        self.data = data
        self.y = y
        self.predict_function = predict_function
        self.y_hat = pred
        self.residual_function = residual_function
        self.residuals = residuals
        self.model_class = model_class
        self.label = label
        self.model_info = model_info
        self.weights = weights
        self.model_type = model_type

        verbose_cat("\nA new explainer has been created!", verbose=verbose)

    def predict(self, data):
        """Make a prediction

        This function uses the `predict_function` attribute.

        Parameters
        ----------
        data : pd.DataFrame (2d)
            Data which will be used to make a prediction.

        Returns
        ----------
        np.ndarray (1d)
            Model predictions for given `data`.
        """

        check_pred_data(data)

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

        check_pred_data(data)

        return self.residual_function(self.model, data, y)

    def predict_parts(self,
                      new_observation,
                      type=('break_down_interactions', 'break_down', 'shap'),
                      order=None,
                      interaction_preference=1,
                      path="average",
                      B=25,
                      keep_distributions=False,
                      processes=1,
                      random_state=None):
        """Calculate instance level variable attributions as Break Down or Shapley Values

        Parameters
        -----------
        new_observation : pd.Series or np.ndarray
            An observation for which a prediction needs to be explained.
        type : {'break_down_interactions', 'break_down', 'shap'}
            Type of variable attributions (default is 'break_down_interactions').
        order : list of int or str, optional
            Use a fixed order of variables for attribution calculation. Use integer values
            or string variable names (default is None which means order by importance).
        interaction_preference : int, optional
            Specify which interactions will be present in an explanation. The larger the
            integer, the more frequently interactions will be presented (default is 1).
        path : list of int, optional
            If specified, then attributions for this path will be plotted
            (default is 'average', which plots attribution means for `B` random paths).
        B : int, optional
            Number of random paths to calculate variable attributions (default is 25).
        keep_distributions :  bool, optional
            Save the distribution of partial predictions (default is False).
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `B`
            (default is 1, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).

        Returns
        -----------
        BreakDown or Shap class object
            Explanation object containing the main result attribute and the plot method.
            Object class, its attributes, and the plot method depend on the `type` parameter.
        """

        types = ('break_down_interactions', 'break_down', 'shap')
        type = check_method_type(type, types)
        path_ = check_path(path)

        if type == 'break_down_interactions' or type == 'break_down':
            predict_parts_ = BreakDown(
                type=type,
                keep_distributions=keep_distributions,
                order=order,
                interaction_preference=interaction_preference
            )
        elif type == 'shap':
            predict_parts_ = Shap(
                keep_distributions=keep_distributions,
                path=path_,
                B=B,
                processes=processes,
                random_state=random_state
            )

        predict_parts_.fit(self, new_observation)

        return predict_parts_

    def predict_profile(self,
                        new_observation,
                        type=('ceteris_paribus',),
                        y=None,
                        variables=None,
                        grid_points=101,
                        variable_splits=None,
                        variable_splits_type='uniform',
                        include_new_observation=True,
                        processes=1,
                        verbose=True):
        """Calculate instance level variable profiles as Ceteris Paribus

        Parameters
        -----------
        new_observation : pd.DataFrame or np.ndarray
            Observations for which predictions need to be explained.
        type : {'ceteris_paribus', TODO: 'oscilations'}
            Type of variable profiles (default is 'ceteris_paribus').
        y : pd.Series or np.ndarray (1d), optional
            Target variable with the same length as `new_observation`.
        variables : array_like of str, optional
            Variables for which the profiles will be calculated
            (default is None, which means all of the variables).
        grid_points : int, optional
            Maximum number of points for profile calculations (default is 101).
            NOTE: The final number of points may be lower than `grid_points`,
            eg. if there is not enough unique values for a given variable.
        variable_splits : dict of lists, optional
            Split points for variables e.g. {'x': [0, 0.2, 0.5, 0.8, 1], 'y': ['a', 'b']}
            (default is None, which means that they will be calculated using one of
            `variable_splits_type` and the `data` attribute).
        variable_splits_type : {'uniform', 'quantiles'}, optional
            Way of calculating `variable_splits`. Set 'quantiles' for percentiles.
            (default is 'uniform', which means uniform grid of points).
        include_new_observation: bool, optional
            Add variable values of `new_observation` data to the `variable_splits`
            (default is True).
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `variables`
            (default is 1, which means no parallel computation).
        verbose : bool, optional
            Print tqdm progress bar (default is True).

        Returns
        -----------
        CeterisParibus class object
            Explanation object containing the main result attribute and the plot method.
        """

        types = ('ceteris_paribus', )
        type = check_method_type(type, types)

        if type == 'ceteris_paribus':
            predict_profile_ = CeterisParibus(
                variables=variables,
                grid_points=grid_points,
                variable_splits=variable_splits,
                variable_splits_type=variable_splits_type,
                include_new_observation=include_new_observation,
                processes=processes
            )

        predict_profile_.fit(self, new_observation, y, verbose)

        return predict_profile_

    def model_performance(self,
                          model_type=None,
                          cutoff=0.5):
        """Calculate dataset level model performance measures

        Parameters
        -----------
        model_type : {'regression', 'classification', None}
            Model task type that is used to choose the proper performance measures
            (default is None, which means try to extract from the `model_type` attribute).
        cutoff : float, optional
            Cutoff for predictions in classification models. Needed for measures like
            recall, precision, acc, f1 (default is 0.5).

        Returns
        -----------
        ModelPerformance class object
            Explanation object containing the main result attribute and the plot method.
        """

        if model_type is None and self.model_type is None:
            raise TypeError("if self.model_type is None, then model_type must be not None")
        elif model_type is None:
            model_type = self.model_type

        model_performance_ = ModelPerformance(
            model_type=model_type,
            cutoff=cutoff
        )
        model_performance_.fit(self)

        return model_performance_

    def model_parts(self,
                    loss_function=None,
                    type=('variable_importance', 'ratio', 'difference'),
                    N=1000,
                    B=10,
                    variables=None,
                    variable_groups=None,
                    keep_raw_permutations=True,
                    processes=1,
                    random_state=None):

        """Calculate dataset level variable importance

        Parameters
        -----------
        loss_function : {'rmse', '1-auc', 'mse', 'mae', 'mad'} or function, optional
            If string, then such loss function will be used to assess variable importance
            (default is 'rmse' or `1-auc`, depends on `model_type` attribute).
        type : {'variable_importance', 'ratio', 'difference'}, optional
            Type of transformation that will be applied to dropout loss.
        N : int, optional
            Number of observations that will be sampled from the `data` attribute before
            the calculation of variable importance. None means all `data` (default is 1000).
        B : int, optional
            Number of permutation rounds to perform on each variable (default is 10).
        variables : array_like of str, optional
            Variables for which the importance will be calculated
            (default is None, which means all of the variables).
            NOTE: Ignored if `variable_groups` is not None.
        variable_groups : dict of lists, optional
            Group the variables to calculate their joint variable importance
            e.g. {'X': ['x1', 'x2'], 'Y': ['y1', 'y2']} (default is None).
        keep_raw_permutations: bool, optional
            Save results for all permutation rounds (default is True).
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `B`
            (default is 1, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).

        Returns
        -----------
        VariableImportance class object
            Explanation object containing the main result attribute and the plot method.
        """

        types = ('variable_importance', 'ratio', 'difference')
        type = check_method_type(type, types)

        loss_function = check_loss_function(self, loss_function)

        model_parts_ = VariableImportance(
            loss_function=loss_function,
            type=type,
            N=N,
            B=B,
            variables=variables,
            variable_groups=variable_groups,
            processes=processes,
            random_state=random_state,
            keep_raw_permutations=keep_raw_permutations,
        )

        model_parts_.fit(self)

        return model_parts_

    def model_profile(self,
                      type=('partial', 'accumulated', 'conditional'),
                      N=300,
                      variables=None,
                      variable_type='numerical',
                      groups=None,
                      span=0.25,
                      grid_points=101,
                      variable_splits=None,
                      center=True,
                      processes=1,
                      random_state=None,
                      verbose=True):

        """Calculate dataset level variable profiles as Partial or Accumulated Dependence

        Parameters
        -----------
        type : {'partial', 'accumulated', 'conditional'}
            Type of model profiles (default is 'partial' for Partial Dependence Profiles).
        N : int, optional
            Number of observations that will be sampled from the `data` attribute before
            the calculation of variable profiles. None means all `data` (default is 300).
        variables : str or array_like of str, optional
            Variables for which the profiles will be calculated
            (default is None, which means all of the variables).
        variable_type : {'numerical', 'categorical'}
            Calculate the profiles for numerical or categorical variables
            (default is 'numerical').
        groups : str or array_like of str, optional
            Names of categorical variables that will be used for profile grouping
            (default is None, which means no grouping).
        span : float, optional
            Smoothing coefficient used as sd for gaussian kernel (default is 0.25).
        grid_points : int, optional
            Maximum number of points for profile calculations (default is 101).
            NOTE: The final number of points may be lower than `grid_points`,
            eg. if there is not enough unique values for a given variable.
        variable_splits : dict of lists, optional
            Split points for variables e.g. {'x': [0, 0.2, 0.5, 0.8, 1], 'y': ['a', 'b']}
            (default is None, which means that they will be distributed uniformly).
        center : bool, optional
            Theoretically Accumulated Profiles starts at 0. If True, then they are centered
            around average response like Partial Profiles (default is True).
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `variables`
            (default is 1, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).
        verbose : bool, optional
            Print tqdm progress bar (default is True).

        Returns
        -----------
        AggregatedProfiles class object
            Explanation object containing the main result attribute and the plot method.
        """

        types = ('partial', 'accumulated', 'conditional')
        type = check_method_type(type, types)

        if N is None:
            N = self.data.shape[0]
        else:
            N = min(N, self.data.shape[0])

        if random_state is not None:
            np.random.seed(random_state)

        I = np.random.choice(np.arange(N), N, replace=False)

        ceteris_paribus = CeterisParibus(grid_points=grid_points,
                                         variable_splits=variable_splits,
                                         variable_splits_type='uniform',
                                         processes=processes)
        ceteris_paribus.fit(self, self.data.iloc[I, :], self.y[I], verbose=verbose)

        model_profile_ = AggregatedProfiles(
            type=type,
            variables=variables,
            variable_type=variable_type,
            groups=groups,
            span=span,
            center=center,
            random_state=random_state
        )

        model_profile_.fit(ceteris_paribus, verbose)

        return model_profile_

    def dumps(self, *args, **kwargs):
        """Return the pickled representation (bytes object) of the Explainer

        This function uses the pickle package. See
        https://docs.python.org/3/library/pickle.html#pickle.dumps

        NOTE: local functions and lambdas cannot be pickled.
        Attribute `residual_function` by default contains lambda; thus,
        if not provided by the user, it will be dropped before the dump.

        Parameters
        -----------
        args :
            Positional arguments passed to the pickle.dumps function
        kwargs :
            Keyword arguments passed to the pickle.dumps function

        Returns
        -----------
        bytes object
        """

        from copy import deepcopy
        to_dump = deepcopy(self)
        to_dump = check_if_local_and_lambda(to_dump)

        import pickle
        return pickle.dumps(to_dump, *args, **kwargs)

    def dump(self, file, *args, **kwargs):
        """Write the pickled representation of the Explainer to the file (pickle)

        This function uses the pickle package. See
        https://docs.python.org/3/library/pickle.html#pickle.dump

        NOTE: local functions and lambdas cannot be pickled.
        Attribute `residual_function` by default contains lambda; thus,
        if not provided by the user, it will be dropped before the dump.

        Parameters
        -----------
        file :
            A file object opened for binary writing, or an io.BytesIO instance.
        args :
            Positional arguments passed to the pickle.dump function
        kwargs :
            Keyword arguments passed to the pickle.dump function
        """

        from copy import deepcopy
        to_dump = deepcopy(self)
        to_dump = check_if_local_and_lambda(to_dump)

        import pickle
        return pickle.dump(to_dump, file, *args, **kwargs)

    @staticmethod
    def loads(data, use_defaults=True, *args, **kwargs):
        """Load the Explainer from the pickled representation (bytes object)

        This function uses the pickle package. See
        https://docs.python.org/3/library/pickle.html#pickle.loads

        Note, that local functions and lambdas cannot be pickled.
        If use_defaults is set to True, then dropped functions are set to defaults.

        Parameters
        -----------
        data : bytes object
            Binary representation of the Explainer.
        use_defaults : bool
            Replace empty `predict_function` and `residual_function` with default
            values like in Explainer initialization (default is True).
        args :
            Positional arguments passed to the pickle.loads function
        kwargs :
            Keyword arguments passed to the pickle.loads function

        Returns
        -----------
        Explainer object
        """

        import pickle
        exp = pickle.loads(data, *args, **kwargs)

        if use_defaults:
            exp = check_if_empty_fields(exp)

        return exp

    @staticmethod
    def load(file, use_defaults=True, *args, **kwargs):
        """Read the pickled representation of the Explainer from the file (pickle)

        This function uses the pickle package. See
        https://docs.python.org/3/library/pickle.html#pickle.load

        Note, that local functions and lambdas cannot be pickled.
        If use_defaults is set to True, then dropped functions are set to defaults.

        Parameters
        -----------
        file :
            A binary file object opened for reading, or an io.BytesIO object.
        use_defaults : bool
            Replace empty `predict_function` and `residual_function` with default
            values like in Explainer initialization (default is True).
        args :
            Positional arguments passed to the pickle.load function
        kwargs :
            Keyword arguments passed to the pickle.load function

        Returns
        -----------
        Explainer object
        """

        import pickle
        exp = pickle.load(file, *args, **kwargs)

        if use_defaults:
            exp = check_if_empty_fields(exp)

        return exp
