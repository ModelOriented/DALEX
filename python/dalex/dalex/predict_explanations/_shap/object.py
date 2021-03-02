import numpy as np
from plotly.subplots import make_subplots

from . import checks, plot, utils
from ... import _theme, _global_checks
from ..._explanation import Explanation


class Shap(Explanation):
    """Calculate predict-level variable attributions as Shapley Values

    Parameters
    -----------
    path : list of int, optional
        If specified, then attributions for this path will be plotted
        (default is `'average'`, which plots attribution means for `B` random paths).
    B : int, optional
        Number of random paths to calculate variable attributions (default is `25`).
    keep_distributions : bool, optional
        Save the distribution of partial predictions (default is `False`).
    processes : int, optional
        Number of parallel processes to use in calculations. Iterated over `B`
        (default is `1`, which means no parallel computation).
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
    path : list of int or 'average'
        Path for which the attributions will be plotted.
    B : int
        Number of random paths to calculate variable attributions.
    keep_distributions : bool
        Save the distribution of partial predictions.
    yhats_distributions : pd.DataFrame or None
        The distribution of partial predictions.
    processes : int
        Number of parallel processes to use in calculations. Iterated over `B`.
    random_state : int or None
        Seed that was set for random number generator.

    Notes
    --------
    - https://pbiecek.github.io/ema/shapley.html
    """

    def __init__(self,
                 path="average",
                 B=25,
                 keep_distributions=False,
                 processes=1,
                 random_state=None):

        _path = checks.check_path(path)
        _processess = checks.check_processes(processes)
        _random_state = checks.check_random_state(random_state)

        self.path = _path
        self.keep_distributions = keep_distributions
        self.B = B
        self.result = None
        self.yhats_distributions = None
        self.prediction = None
        self.intercept = None
        self.processes = _processess
        self.random_state = _random_state

    def _repr_html_(self):
        return self.result._repr_html_()

    def fit(self,
            explainer,
            new_observation):
        """Calculate the result of explanation

        Fit method makes calculations in place and changes the attributes.

        Parameters
        -----------
        explainer : Explainer object
            Model wrapper created using the Explainer class.
        new_observation : pd.Series or np.ndarray
            An observation for which a prediction needs to be explained.

        Returns
        -----------
        None
        """

        _new_observation = checks.check_new_observation(new_observation, explainer)
        checks.check_columns_in_new_observation(_new_observation, explainer)
        self.result, self.prediction, self.intercept, self.yhats_distributions = utils.shap(
            explainer,
            _new_observation,
            self.path,
            self.keep_distributions,
            self.B,
            self.processes,
            self.random_state
        )

    def plot(self,
             objects=None,
             baseline=None,
             max_vars=10,
             digits=3,
             rounding_function=np.around,
             bar_width=16,
             min_max=None,
             vcolors=None,
             title="Shapley Values",
             vertical_spacing=None,
             show=True):
        """Plot the Shapley Values explanation

        Parameters
        -----------
        objects : Shap object or array_like of Shap objects
            Additional objects to plot in subplots (default is `None`).
        baseline: float, optional
            Starting x point for bars (default is average prediction).
        max_vars : int, optional
            Maximum number of variables that will be presented for for each subplot
            (default is `10`).
        digits : int, optional
            Number of decimal places (`np.around`) to round contributions.
            See `rounding_function` parameter (default is `3`).
        rounding_function : function, optional
            A function that will be used for rounding numbers (default is `np.around`).
        bar_width : float, optional
            Width of bars in px (default is `16`).
        min_max : 2-tuple of float, optional
            Range of OX axis (default is `[min-0.15*(max-min), max+0.15*(max-min)]`).
        vcolors : 3-tuple of str, optional
            Color of bars (default is `["#8bdcbe", "#f05a71"]`).
        title : str, optional
            Title of the plot (default is `"Shapley Values"`).
        vertical_spacing : float <0, 1>, optional
            Ratio of vertical space between the plots (default is `0.2/number of rows`).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object that can 
            be edited or saved using the `write_image()` method (default is `True`).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """

        # are there any other objects to plot?
        if objects is None:
            n = 1
            _result_list = [self.result.loc[self.result['B'] == 0,].copy()]
            _intercept_list = [self.intercept]
            _prediction_list = [self.prediction]
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            n = 2
            _result_list = [self.result.loc[self.result['B'] == 0,].copy(),
                            objects.result.loc[objects.result['B'] == 0,].copy()]
            _intercept_list = [self.intercept, objects.intercept]
            _prediction_list = [self.prediction, objects.prediction]
        elif isinstance(objects, (list, tuple)):  # objects as tuple or array
            n = len(objects) + 1
            _result_list = [self.result.loc[self.result['B'] == 0,].copy()]
            _intercept_list = [self.intercept]
            _prediction_list = [self.prediction]
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _result_list += [ob.result.loc[ob.result['B'] == 0,].copy()]
                _intercept_list += [ob.intercept]
                _prediction_list += [ob.prediction]
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)

        # TODO: add intercept and prediction list update for multi-class
        # deleted_indexes = []
        # for i in range(n):
        #     result = _result_list[i]
        #
        #     if len(result['label'].unique()) > 1:
        #         n += len(result['label'].unique()) - 1
        #         # add new data frames to list
        #         _result_list += [v for k, v in result.groupby('label', sort=False)]
        #         deleted_indexes += [i]
        #
        # _result_list = [j for i, j in enumerate(_result_list) if i not in deleted_indexes]
        model_names = [result.iloc[0, result.columns.get_loc("label")] for result in _result_list]

        if vertical_spacing is None:
            vertical_spacing = 0.2 / n

        fig = make_subplots(rows=n, cols=1,
                            shared_xaxes=True, vertical_spacing=vertical_spacing,
                            x_title='contribution', subplot_titles=model_names)
        plot_height = 78 + 71

        if vcolors is None:
            vcolors = _theme.get_break_down_colors()

        if min_max is None:
            temp_min_max = [np.Inf, -np.Inf]
        else:
            temp_min_max = min_max

        for i, _result in enumerate(_result_list):
            if _result.shape[0] <= max_vars:
                m = _result.shape[0]
            else:
                m = max_vars + 1

            if baseline is None:
                baseline = _intercept_list[i]
            prediction = _prediction_list[i]

            df = plot.prepare_data_for_shap_plot(_result, baseline, prediction, max_vars, rounding_function, digits)

            fig.add_shape(
                type='line',
                x0=baseline,
                x1=baseline,
                y0=-1,
                y1=m,
                yref="paper",
                xref="x",
                line={'color': "#371ea3", 'width': 1.5, 'dash': 'dot'},
                row=i + 1, col=1
            )

            fig.add_bar(
                orientation="h",
                y=df['variable'].tolist(),
                x=df['contribution'].tolist(),
                textposition="outside",
                text=df['label_text'].tolist(),
                marker_color=[vcolors[int(c)] for c in df['sign'].tolist()],
                base=baseline,
                hovertext=df['tooltip_text'].tolist(),
                hoverinfo='text',
                hoverlabel={'bgcolor': 'rgba(0,0,0,0.8)'},
                showlegend=False,
                row=i + 1, col=1
            )

            fig.update_yaxes({'type': 'category', 'autorange': 'reversed', 'gridwidth': 2, 'automargin': True,
                              'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True},
                             row=i + 1, col=1)

            fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                              'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                             row=i + 1, col=1)

            plot_height += m * bar_width + (m + 1) * bar_width / 4

            if min_max is None:
                cum = df.contribution.values + baseline
                min_max_margin = cum.ptp() * 0.15
                temp_min_max[0] = np.min([temp_min_max[0], cum.min() - min_max_margin])
                temp_min_max[1] = np.max([temp_min_max[1], cum.max() + min_max_margin])

        plot_height += (n - 1) * 70

        fig.update_xaxes({'range': temp_min_max})
        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          height=plot_height, margin={'t': 78, 'b': 71, 'r': 30})

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
