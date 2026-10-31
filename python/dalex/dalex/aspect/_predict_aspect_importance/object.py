from .._plotting import plot_data
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from dalex import _theme, _global_checks, _global_utils
from dalex._explanation import Explanation, EmptyTable
import narwhals.stable.v2 as nw
from .._table import AspectTable
from ..._dataframe import copy_for_edit
from dalex.aspect.utils import calculate_min_depend, get_min_depend_from_matrix

from . import checks, utils, plot


class PredictAspectImportance(Explanation):
    """Calculate predict-level aspect importance

    Parameters
    -----------
    variable_groups : dict of lists 
        Variables grouped in aspects to calculate their importance. 
    type : {'default', 'shap'}, optional
        Type of aspect importance/attributions (default is `'default'`, which means 
        the use of simplified LIME method).
    N : int, optional
        Number of observations that will be sampled from the `data` attribute
        before the calculation of aspect importance (default is `2000`).
    B : int, optional
        Parameter specific for `type == 'shap'`. Number of random paths to calculate aspect
        attributions (default is `25`).
        NOTE: Ignored if `type` is not `'shap'`.
    n_aspects : int, optional
        Parameter specific for `type == 'default'`. Maximum number of non-zero importances, i.e.
        coefficients after lasso fitting (default is `None`, which means the linear regression is used).
        NOTE: Ignored if `type` is not `'default'`.
    sample_method : {'default', 'binom'}, optional
        Parameter specific for `type == 'default'`. Sampling method for creating binary matrix 
        used as mask for replacing aspects in sampled data (default is `'default'`, which means 
        it randomly replaces one or two zeros per row; `'binom'` replaces random number of zeros 
        per row).
        NOTE: Ignored if `type` is not `'default'`.
    f : int, optional
        Parameter specific for `type == 'default'` and `sample_method == 'binom'`. Parameter 
        controlling average number of replaced zeros for binomial sampling (default is `2`). 
        NOTE: Ignored if `type` is not `'default'` or `sample_method` is not `'binom'`.
    depend_method: {'assoc', 'pps'} or function, optional
        The method of calculating the dependencies between variables (i.e. the dependency 
        matrix). Default is `'assoc'`, which means the use of statistical association 
        (correlation coefficient, Cramér's V and eta-quared); 
        `'pps'` stands for Power Predictive Score.
        NOTE: When a function is passed, it is called with the `data` and it 
        must return a symmetric dependency matrix (`pd.DataFrame` with variable names as 
        columns and rows).
    corr_method : {'spearman', 'pearson', 'kendall'}, optional
        The method of calculating correlation between numerical variables 
        (default is `'spearman'`).
        NOTE: Ignored if `depend_method` is not `'assoc'`.
    agg_method : {'max', 'min', 'avg'}, optional
        The method of aggregating the PPS values for pairs of variables 
        (default is `'max'`).
        NOTE: Ignored if `depend_method` is not `'pps'`. 
    processes : int, optional
        Parameter specific for `type == 'shap'`. Number of parallel processes to use in calculations.
        Iterated over `B` (default is `1`, which means no parallel computation).
    random_state : int, optional
        Set seed for random number generator (default is random seed).

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    prediction : float
        Prediction for `new_observation`.
    intercept : float
        Average prediction for `data`.
    variable_groups : dict of lists 
        Variables grouped in aspects to calculate their importance. 
    type : {'default', 'shap'}
        Type of aspect importance/attributions to calculate.
    N : int
        Number of observations that will be sampled from the `data` attribute
        before the calculation of aspect importance.
    B : int
        Number of random paths to calculate aspect attributions.
    n_aspects : int
        Maximum number of non-zero importances.
    sample_method : {'default', 'binom'}
        Sampling method for creating binary matrix used as mask for replacing aspects in sampled data.
    f : int
        Average number of replaced zeros for binomial sampling.
    depend_method : {'assoc', 'pps'}
        The method of calculating the dependencies between variables.
    corr_method : {'spearman', 'pearson', 'kendall'}
        The method of calculating correlation between numerical variables.
    agg_method : {'max', 'min', 'avg'}
        The method of aggregating the PPS values for pairs of variables.
    processes : int
        Number of parallel processes to use in calculations. Iterated over `B`.
    random_state : int
        Set seed for random number generator.
    
    Notes
    -----
    - https://arxiv.org/abs/2104.03403
    """
    def __init__(
        self,
        variable_groups,
        type="default",
        N=2000,
        B=25,
        n_aspects=None,
        sample_method="default",
        f=2,
        depend_method="assoc",
        corr_method="spearman",
        agg_method="max",
        processes=1,
        random_state=None,
        **kwargs
    ):
        
        types = ('default', 'shap')
        aliases = {'simplified_lime': 'default', 'lime': 'default', 'shapley_values': 'shap'}
        _type = checks.check_method_type(type, types, aliases)
        _processes = checks.check_processes(processes)
        _random_state = checks.check_random_state(random_state)
        _depend_method, _corr_method, _agg_method = checks.check_method_depend(depend_method, corr_method, agg_method)
        self._depend_matrix = None
        if "_depend_matrix" in kwargs:
            self._depend_matrix = kwargs.get("_depend_matrix")
        
        self.variable_groups = variable_groups
        self.type = _type
        self.N = N
        self.B = B
        self.n_aspects = n_aspects
        self.sample_method = sample_method
        self.f = f
        self.depend_method = _depend_method
        self.corr_method = _corr_method
        self.agg_method = _agg_method
        self.random_state = _random_state
        self.processes = _processes
        self.prediction = None
        self.intercept = None
        self.result = EmptyTable()

    def _repr_html_(self):
        return self.result._repr_html_()

    def fit(self, explainer, new_observation, *, _rng=None):
        """Calculate the result of explanation
        Fit method makes calculations in place and changes the attributes.

        Parameters
        ----------
        explainer : Explainer object
            Model wrapper created using the Explainer class.
        new_observation : pd.Series or np.ndarray (1d) or pd.DataFrame (1,p)
            An observation for which a prediction needs to be explained.
        
        Returns
        -----------
        None
        """
        _new_observation = checks.check_new_observation(new_observation, explainer)
        checks.check_columns_in_new_observation(_new_observation, explainer)
        _variable_groups = checks.check_variable_groups(self.variable_groups, explainer)
        
        self.prediction = explainer.predict(_new_observation)[0]
        self.intercept = (explainer.y_hat if explainer.y_hat is not None
                          else explainer.predict(explainer.data)).mean()
        
        from .._native_importance import calculate
        data, names, value_lists = calculate(explainer, _new_observation, _variable_groups,
                                             kind=self.type, N=self.N, n_aspects=self.n_aspects,
                                             sample_method=self.sample_method, f=self.f, B=self.B,
                                             processes=self.processes, random_state=self.random_state, rng=_rng)
        if self._depend_matrix is not None:
            pairs, minima = get_min_depend_from_matrix(self._depend_matrix, names)
        else:
            pairs, minima = calculate_min_depend(names, explainer.data, self.depend_method,
                                                 self.corr_method, self.agg_method)
        frame = nw.from_native(data, eager_only=True)
        frame = frame.with_columns(nw.new_series('min_depend', minima, backend=nw.get_native_namespace(frame)))
        observation_snapshot = nw.from_native(copy_for_edit(_new_observation), eager_only=True).clone().to_native()
        self.result = AspectTable(frame.to_native(), names, pairs, observation_snapshot,
                                  value_lists=value_lists)


    def plot(
        self,
        objects=None,
        baseline=None,
        max_aspects=10,
        show_variable_names=True,
        digits=3,
        rounding_function=np.around,
        bar_width=25,
        min_max=None,
        vcolors=None,
        title="Predict Aspect Importance",
        vertical_spacing=None,
        show=True,
    ):
        """Plot the Predict Aspect Importance explanation.

        Parameters
        ----------
        objects : PredictAspectImportance object or array_like of PredictAspectImportance objects
            Additional objects to plot in subplots (default is `None`).
        baseline: float, optional
            Starting x point for bars 
            (default is 0 if `type` is `'default'` and average prediction if `type` is `'shap'`).
        max_aspects : int, optional
            Maximum number of aspects that will be presented for for each subplot
            (default is `10`).
        show_variable_names : bool, optional
            `True` shows names of variables grouped in aspects; `False` shows names of aspects
            (default is `True`).
        digits : int, optional
            Number of decimal places (`np.around`) to round contributions.
            See `rounding_function` parameter (default is `3`).
        rounding_function : function, optional
            A function that will be used for rounding numbers (default is `np.around`).
        bar_width : float, optional
            Width of bars in px (default is `16`).
        min_max : 2-tuple of float, optional
            Range of OX axis (default is `[min-0.15*(max-min), max+0.15*(max-min)]`).
        vcolors : 2-tuple of str, optional
            Color of bars (default is `["#8bdcbe", "#f05a71"]`).
        title : str, optional
            Title of the plot (default is `"Predict Aspect Importance"`).
        vertical_spacing : float <0, 1>, optional
            Ratio of vertical space between the plots (default is `0.2/number of rows`).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object that can 
            be edited or saved using the `write_image()` method (default is `True`).

        Returns
        -------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """

        if not isinstance(max_aspects, (int, np.integer)) or max_aspects < 1:
            raise ValueError('max_aspects must be a positive integer')
        _result_list = [plot_data(self)]
        _intercept_list = [self.intercept]
        # are there any other objects to plot?
        if objects is None:
            n = 1
        elif isinstance(objects, self.__class__):
            n = 2
            _result_list += [plot_data(objects)]
            _intercept_list += [objects.intercept]
        elif isinstance(objects, (list, tuple)):
            n = len(objects) + 1
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _result_list += [plot_data(ob)]
                _intercept_list += [ob.intercept]
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)

        model_names = [
            result.frame['label'][0] for result in _result_list
        ]

        if vertical_spacing is None:
            vertical_spacing = 0.2 / n

        # generate plot
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            x_title="aspect importance",
            subplot_titles=model_names,
        )

        plot_height = 78 + 71

        if vcolors is None:
            vcolors = _theme.get_aspect_importance_colors()

        if min_max is None:
            temp_min_max = [np.inf, -np.inf]
        else:
            temp_min_max = min_max

        for i, _result in enumerate(_result_list):
            if len(_result.frame) <= max_aspects:
                m = len(_result.frame)
            else:
                m = max_aspects + 1

            if baseline is None:
                if self.type == 'shap':
                    baseline = _intercept_list[i]
                else: 
                    baseline = 0 
            
            _data = _result.take(range(min(max_aspects, len(_result.frame)))).rounded('importance', rounding_function, digits)
            _rows = _data.rows()
            _result = _data.frame
            colors = [0 if value > 0 else 1 for value in _result['importance'].to_list()]
            tooltip = [plot.tooltip_text(row, rounding_function, digits, self.type) for row in _rows]
            labels = _global_utils.convert_float_to_str(_result['importance'].to_numpy(), '+')

            fig.add_shape(
                type="line",
                x0=baseline,
                x1=baseline,
                y0=-1,
                y1=m,
                yref="paper",
                xref="x",
                line={"color": "#371ea3", "width": 1.5, "dash": "dot"},
                row=i + 1,
                col=1,
            )

            fig.add_bar(
                orientation="h",
                y=[
                    ", ".join(variables_list)
                    for variables_list in [row['variable_names'] for row in _rows]
                ]
                if show_variable_names
                else _result["aspect_name"].to_list(),
                x=_result["importance"].to_list(),
                textposition="outside",
                text=labels,
                marker_color=[vcolors[int(c)] for c in colors],
                base=baseline,
                hovertext=tooltip,
                hoverinfo="text",
                hoverlabel={"bgcolor": "rgba(0,0,0,0.8)"},
                showlegend=False,
                row=i + 1,
                col=1,
            )

            fig.update_yaxes(
                {
                    "type": "category",
                    "autorange": "reversed",
                    "gridwidth": 2,
                    "automargin": True,
                    "ticks": "outside",
                    "tickcolor": "white",
                    "ticklen": 10,
                    "fixedrange": True,
                },
                row=i + 1,
                col=1,
            )

            fig.update_xaxes(
                {
                    "type": "linear",
                    "gridwidth": 2,
                    "zeroline": False,
                    "automargin": True,
                    "ticks": "outside",
                    "tickcolor": "white",
                    "ticklen": 3,
                    "fixedrange": True,
                },
                row=i + 1,
                col=1,
            )

            plot_height += m * bar_width + (m + 1) * bar_width / 4

            if min_max is None:
                cum = _result['importance'].to_numpy() + baseline
                min_max_margin =  np.ptp(cum) * 0.15 
                temp_min_max[0] = np.min(
                    [
                        temp_min_max[0],
                        cum.min() - min_max_margin,
                    ]
                )
                temp_min_max[1] = np.max(
                    [
                        temp_min_max[1],
                        cum.max() + min_max_margin,
                    ]
                )

        plot_height += (n - 1) * 70

        fig.update_xaxes({"range": temp_min_max})
        fig.update_layout(
            title_text=title,
            title_x=0.15,
            font={"color": "#371ea3"},
            template="none",
            height=plot_height,
            margin={"t": 78, "b": 71, "r": 30},
        )

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
