from dalex.dataset_level import ModelPerformance, VariableImportance, AggregatedProfiles
from dalex.instance_level import BreakDown, Shap, CeterisParibus
from .checks import *
from .helper import get_model_info


class Explainer:
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

        """Create Model Explainer

        Black-box models may have very different structures.
        This function creates a unified representation of a model, which can be further processed by various explainers.
        Please NOTE, that the model is the only required argument.
        But some explainers may require that other arguments will be provided too.

        :param model: a model to be explained
        :param data: data.frame or matrix - data that was used for fitting. If not provided then will be extracted from the model. Data should be passed without target column (this shall be provided as the y argument). NOTE: If target variable is present in the data, some of the functionalities my not work properly.
        :param y: numeric vector with outputs / scores. If provided then it shall have the same size as data
        :param predict_function: function that takes two arguments: model and new data and returns numeric vector with predictions
        :param residual_function: function that takes three arguments: model, data and response vector y. It should return a numeric vector with model residuals for given data. If not provided, response residuals y-yhat are calculated.
        :param weights: weights numeric vector with sampling weights. By default it's None. If provided then it shall have the same length as data
        :param label: the name of the model. By default it's extracted from the 'class' attribute of the model
        :param model_class: str, class of actual model, use if your model is wrapped
        :param verbose: if True (default) then diagnostic messages will be printed
        :param precalculate: if True (default) then predicted_values and residual are calculated when explainer is created.
        :param model_type: type of a model, either classification or regression.
        :param model_info: list containing additional information about the model
        :param colorize: TODO
        """

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
        Contrary to DALEX in R, data cannot be retrieved from model, thus data is necessary.
        if data is None
        tell user about the lack of the data
        """

        """
        Input numpy/pandas
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
        predict_function, pred, model_info_ = check_predict_function(predict_function, model, data, model_class,
                                                                     model_info_, precalculate, verbose)

        # if data is specified then we may test predict_function
        # at this moment we have predict function

        # REPORT: checks for residual_function
        residual_function, residuals, model_info_ = check_residual_function(residual_function, predict_function, model,
                                                                            data, y,
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

        """This is a generic function for making a prediction.
        
        :param data: pd.DataFrame
        :return: array-like, prediction of the model
        """

        check_pred_data(data)

        return self.predict_function(self.model, data)

    def residual(self, data, y):

        """This is a generic function for calculating residuals.

        :param data: pd.DataFrame
        :param y: numeric vector with outputs / scores. If provided then it shall have the same size as data
        :return: array-like, prediction of the model
        """

        return self.residual_function(self.model, data, y)

    def predict_parts(self,
                      new_observation,
                      type=('break_down_interactions', 'break_down', 'shap'),
                      order=None,
                      interaction_preference=1,
                      path="average",
                      B=25,
                      keep_distributions=False):

        """Instance Level Variable Attribution as Break Down or SHAP Explanations

        :param new_observation: pd.Series, a new observarvation for which predictions need to be explained
        :param type: type the type of variable attributions. Either 'shap', 'break_down' or 'break_down_interactions'
        :param order: if not `NULL`, then it will be a fixed order of variables. It can be a numeric vector or vector with names of variables.
        :param interaction_preference: an integer specifying which interactions will be present in an explanation. The larger the integer, the more frequently interactions will be presented.
        :param path: if specified, then this path will be highlighted on the plot. Use `average` in order to show an average effect
        :param B: number of random paths
        :param keep_distributions: if `TRUE`, then distribution of partial predictions is stored and can be plotted with the generic `plot()`.
        :return: BreakDown / Shap
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
                B=B
            )

        predict_parts_.fit(self, new_observation)

        return predict_parts_

    def predict_profile(self,
                        new_observation,
                        type=('ceteris_paribus'),
                        y=None,
                        variables=None,
                        grid_points=101,
                        variable_splits=None):

        """Creates CeterisParibus object

        :param new_observation: DataFrame with observations for which the profiles will be calculated
        :param type TODO
        :param y: pandas Series with labels for the observations
        :param variables: variables for which the profiles are calculated
        :param grid_points: number of points in a single variable split if calculated automatically
        :param variable_splits: mapping of variables into points the profile will be calculated, if None then calculate with the function `_calculate_variable_splits`
        :return CeterisParibus object
        """

        types = ('ceteris_paribus')
        type = check_method_type(type, types)

        if type == 'ceteris_paribus':
            predict_profile_ = CeterisParibus(
                variables=variables,
                grid_points=grid_points,
                variable_splits=variable_splits
            )

        predict_profile_.fit(self, new_observation, y)

        return predict_profile_

    def model_performance(self,
                          model_type=None,
                          cutoff=0.5):

        """Model Performance Measures

        :param model_type: type of prediction regression/classification
        :param cutoff: a cutoff for classification models, needed for measures like recall, precision, ACC, F1. By default 0.5.
        :return: ModelPerformance object
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
                    loss_function='rmse',
                    type=('variable_importance', 'ratio', 'difference'),
                    N=None,
                    B=10,
                    keep_raw_permutations=None,
                    variables=None,
                    variable_groups=None,
                    random_state=None):

        """Creates VariableImportance object

        :param loss_function: a function that will be used to assess variable importance
        :param type: 'variable_importance'/'ratio'/'difference' type of transformation that should be applied for dropout loss
        :param N: number of observations that should be sampled for calculation of variable importance
        :param B: number of permutation rounds to perform on each variable
        :param keep_raw_permutations: TODO
        :param variables: vector of variables. If None then variable importance will be tested for each variable from the data separately, ignored if variable_groups is not None
        :param variable_groups: dict of lists of variables. Each list is treated as one group. This is for testing joint variable importance
        :param label: TODO
        :param random_state: random state for the permutations
        :return: FeatureImportance object
        """

        types = ('variable_importance', 'ratio', 'difference')
        type = check_method_type(type, types)

        model_parts_ = VariableImportance(
            loss_function=loss_function,
            type=type,
            N=N,
            B=B,
            variables=variables,
            variable_groups=variable_groups,
            random_state=random_state,
            keep_raw_permutations=keep_raw_permutations,
        )

        model_parts_.fit(self)

        return model_parts_

    def model_profile(self,
                      type=('partial', 'accumulated', 'conditional'),
                      N=500,
                      variables=None,
                      variable_type='numerical',
                      groups=None,
                      span=0.25,
                      grid_points=101,
                      intercept=True):

        """Dataset Level Variable Effect as Partial Dependency Profile or Accumulated Local Effects

        :param ceteris_paribus: a ceteris paribus explainer or list of ceteris paribus explainers
        :param N: number of observations used for calculation of partial dependency profiles. By default, 500 observations will be chosen randomly. If None then all observations will be used.
        :param variables: str or list or numpy.ndarray or pandas.Series, if not None then aggregate only for selected variables will be calculated, if None all will be selected
        :param variable_type: TODO If "numerical" then only numerical variables will be calculated. If "categorical" then only categorical variables will be calculated.
        :param groups: str or list or numpy.ndarray or pandas.Series, a variable names that will be used for grouping
        :param type: either partial/conditional/accumulated for partial dependence, conditional profiles of accumulated local effects
        :param span: smoothing coeffcient, by default 0.25.It's the sd for gaussian kernel
        :param grid_points: number of points for profile
        :param intercept: False if center data on 0
        :return: VariableEffect object
        """

        types = ('partial', 'accumulated', 'conditional')
        type = check_method_type(type, types)

        if N is None:
            N = self.data.shape[0]
        else:
            N = min(N, self.data.shape[0])

        I = np.random.choice(np.arange(N), N, replace=False)

        ceteris_paribus = CeterisParibus(grid_points=grid_points)
        ceteris_paribus.fit(self, self.data.iloc[I, :], self.y[I])

        model_profile_ = AggregatedProfiles(
            type=type,
            variables=variables,
            variable_type=variable_type,
            groups=groups,
            span=span,
            intercept=intercept
        )

        model_profile_.fit(ceteris_paribus)

        return model_profile_

    def dumps(self, *args, **kwargs):
        """Return binary (pickled) representation of the explainer.

        Note, that local functions and lambdas cannot be pickled.
        Residual function by default contains lambda, thus, if default, is always dropped.
        Original explainer is not changed.

        :param args: Positional arguments passed to pickle.dumps
        :param kwargs: Keyword arguments passed to pickle.dumps
        :return: binary (pickled) representation of the explainer
        """

        from copy import deepcopy
        to_dump = deepcopy(self)
        to_dump = check_if_local_and_lambda(to_dump)

        import pickle
        return pickle.dumps(to_dump, *args, **kwargs)

    def dump(self, file, *args, **kwargs):
        """Save binary (pickled) representation of the explainer to the file.

        Note, that local functions and lambdas cannot be pickled.
        Residual function by default contains lambda, thus, if default, is always dropped.
        Original explainer is not changed.

        :param file: It can be a file object opened for binary writing, an io.BytesIO instance.
        :param args: Positional arguments passed to pickle.dump
        :param kwargs: Keyword arguments passed to pickle.dump
        :return: binary (pickled) representation of the explainer
        """

        from copy import deepcopy
        to_dump = deepcopy(self)
        to_dump = check_if_local_and_lambda(to_dump)

        import pickle
        return pickle.dump(to_dump, file, *args, **kwargs)

    @staticmethod
    def loads(data, use_defaults=True, *args, **kwargs):
        """Load explainer from binary (pickled) representation.

        Note, that local functions and lambdas cannot be pickled.
        If use_defaults is set to True, then dropped functions are set to defaults.

        :param data: binary representation of the explainer
        :param use_defaults: True, if dropped functions should be replaced with defaults
        :param args: Positional arguments passed to pickle.loads
        :param kwargs: Keyword arguments passed to pickle.loads
        :return: Explainer
        """

        import pickle
        exp = pickle.loads(data, *args, **kwargs)

        if use_defaults:
            exp = check_if_empty_fields(exp)

        return exp

    @staticmethod
    def load(file, use_defaults=True, *args, **kwargs):
        """Load explainer from binary file (pickled).

        Note, that local functions and lambdas cannot be pickled.
        If use_defaults is set to True, then dropped functions are set to defaults.

        :param file: It can be a binary file object opened for reading, an io.BytesIO object
        :param use_defaults: True, if dropped functions should be replaced with defaults
        :param args: Positional arguments passed to pickle.load
        :param kwargs: Keyword arguments passed to pickle.load
        :return: Explainer
        """

        import pickle
        exp = pickle.load(file, *args, **kwargs)

        if use_defaults:
            exp = check_if_empty_fields(exp)

        return exp
