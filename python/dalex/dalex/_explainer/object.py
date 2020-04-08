from dalex.instance_level import BreakDown, Shap, CeterisParibus
from dalex.dataset_level import ModelPerformance, VariableImportance, AggregatedProfiles
from .checks import *


class Explainer:
    def __init__(self,
                 model,
                 data=None,
                 y=None,
                 predict_function=None,
                 residual_function=None,
                 weights=None,
                 label=None,
                 verbose=True,
                 precalculate=True,
                 colorize=True,
                 model_info=None,
                 model_type=None):

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
        :param verbose: if True (default) then diagnostic messages will be printed
        :param precalculate: if True (default) then predicted_values and residual are calculated when explainer is created.
        :param colorize: TODO
        :param model_info: list containg information about the model
        :param model_type: type of a model, either classification or regression.
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

        # REPORT: checks for model label
        label = check_label(label, model, verbose)

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

        # REPORT: checks for predict_function
        predict_function, pred = check_predict_function(predict_function, model, data, precalculate, verbose)

        # if data is specified then we may test predict_function
        # at this moment we have predict function

        # REPORT: checks for residual_function
        residual_function, residuals = check_residual_function(residual_function, predict_function, model, data, y,
                                                               precalculate, verbose)

        model_info = check_model_info(model_info, model, verbose)

        # READY to create an explainer
        self.model = model
        self.data = data
        self.y = y
        self.predict_function = predict_function
        self.y_hat = pred
        self.residual_function = residual_function
        self.residuals = residuals
        self.model_class = type(model)
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

        return self.predict_function(self.model, data)

    def predict_parts(self,
                     new_observation,
                     type=('break_down_interactions','break_down','shap'),
                     order=None,
                     interaction_preference=1,
                     path=None,
                     B=25,
                     keep_distributions=False):

        """Instance Level Variable Attribution as Break Down or SHAP Explanations

        :param new_observation: pd.Series, a new observarvation for which predictions need to be explained
        :param type: type the type of variable attributions. Either 'shap', 'break_down' or 'break_down_interactions'
        :param order: if not `NULL`, then it will be a fixed order of variables. It can be a numeric vector or vector with names of variables.
        :param interaction_preference: an integer specifying which interactions will be present in an explanation. The larger the integer, the more frequently interactions will be presented.
        :param path: if specified, then this path will be highlighed on the plot. Use `average` in order to show an average effect
        :param B: number of random paths
        :param keep_distributions: if `TRUE`, then distribution of partial predictions is stored and can be plotted with the generic `plot()`.
        :return: BreakDown / Shap
        """

        types = ('break_down_interactions','break_down','shap')
        type = check_method_type(type, types)

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
                path=path,
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

        model_performance_ = ModelPerformance(model_type, cutoff)
        model_performance_.fit(self)

        return model_performance_

    def model_parts(self,
                    loss_function='loss_root_mean_square',
                    type=('variable_importance','ratio','difference'),
                    n_sample=None,
                    B=10,
                    keep_raw_permutations=None,
                    variables=None,
                    variable_groups=None,
                    random_state=None):

        """Creates VariableImportance object

        :param loss_function: a function thet will be used to assess variable importance
        :param type: type of transformation that should be applied for dropout loss
        :param n_sample: number of observations that should be sampled for calculation of variable importance
        :param B: number of permutation rounds to perform on each variable
        :param keep_raw_permutations: TODO
        :param variables: vector of variables. If None then variable importance will be tested for each variable from the data separately
        :param variable_groups: list of variables names vectors. This is for testing joint variable importance
        :param label: TODO
        :param random_state: random state for the permutations
        :return: FeatureImportance object
        """

        types = ('variable_importance','ratio','difference')
        type = check_method_type(type, types)

        model_parts_ = VariableImportance(
            loss_function=loss_function,
            type=type,
            n_sample=n_sample,
            B=B,
            variables=variables,
            variable_groups=variable_groups,
            random_state=random_state,
            keep_raw_permutations=keep_raw_permutations,
        )

        model_parts_.fit(self)

        return model_parts_

    def model_profile(self,
                      type=('partial','accumulated','conditional'),
                      N=500,
                      variables=None,
                      variable_type='numerical',
                      groups=None,
                      span=0.25,
                      grid_points=101,
                      intercept=True):

        """Dataset Level Variable Effect as Partial Dependency Profile or Accumulated Local Effects

        :param ceteris_paribus: a ceteris paribus explainer or list of ceteris paribus explainers
        :param N: number of observations used for calculation of partial dependency profiles. By default, 500 observations will be chosen randomly.
        :param variables: names of variables for which profiles shall be calculated.
        :param variable_type: TODO If "numerical" then only numerical variables will be calculated. If "categorical" then only categorical variables will be calculated.
        :param groups: a variable name that will be used for grouping.
        :param type: either partial/conditional/accumulated for partial dependence, conditional profiles of accumulated local effects
        :param span: smoothing coeffcient, by default 0.25.It's the sd for gaussian kernel
        :param grid_points: number of points for profile
        :param intercept: False if center data on 0
        :return: VariableEffect object
        """

        types = ('partial','accumulated','conditional')
        type = check_method_type(type, types)

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
