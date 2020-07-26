from warnings import warn

from plotly.subplots import make_subplots

from .plot import *
from .checks import *
from .utils import local_interactions
from ..._explainer.theme import get_break_down_colors


class BreakDown:
    def __init__(self,
                 type='break_down',
                 keep_distributions=False,
                 order=None,
                 interaction_preference=1):

        order = check_order(order)

        self.type = type
        self.keep_distributions = keep_distributions
        self.order = order
        self.interaction_preference = interaction_preference
        self.result = None
        self.yhats_distributions = None

    def fit(self,
            explainer,
            new_observation):

        new_observation = check_new_observation(new_observation, explainer)
        if new_observation.shape[0] != 1:
            warn("You should pass only one new_observation, taken only first")
            new_observation = new_observation.iloc[0, :]

        if self.type == 'break_down_interactions':
            result, yhats_distributions = local_interactions(explainer,
                                                             new_observation,
                                                             self.interaction_preference,
                                                             '2d',
                                                             self.order,
                                                             self.keep_distributions)
        elif self.type == 'break_down':
            result, yhats_distributions = local_interactions(explainer,
                                                             new_observation,
                                                             self.interaction_preference,
                                                             '1d',
                                                             self.order,
                                                             self.keep_distributions)
        else:
            raise ValueError("'type' must be one of 'break_down_interactions', 'break_down'")

        self.result = result
        self.yhats_distributions = yhats_distributions

    def plot(self,
             objects=None,
             baseline=None,
             max_vars=10,
             digits=3,
             rounding_function=np.around,
             bar_width=16,
             min_max=None,
             vcolors=None,
             title="Break Down",
             vertical_spacing=None,
             show=True):
        """Plot the Break Down explanation

        Parameters
        -----------
        objects : BreakDown object or array_like of BreakDown objects
            Additional objects to plot in subplots (default is None).
        baseline: float, optional
            Starting x point for bars (default is average prediction).
        max_vars : int, optional
            Maximum number of variables that will be presented for for each subplot
            (default is 10).
        digits : int, optional
            Number of decimal places (np.around) to round contributions.
            See `rounding_function` parameter (default is 3).
        rounding_function : function, optional
            A funciton that will be used for rounding numbers (default is np.around).
        bar_width : float, optional
            Width of bars in px (default is 16).
        min_max : 2-tuple of float, optional
            Range of x-axis (default is [min - 0.15*(max-min), max + 0.15*(max-min)]).
        vcolors : 3-tuple of str, optional
            Color of bars (default is ["#371ea3", "#8bdcbe", "#f05a71"]).
        title : str, optional
            Title of the plot (default is "Break Down").
        vertical_spacing : float <0, 1>, optional
            Ratio of vertical space between the plots (default is 0.2/number of subplots).
        show : bool, optional
            True shows the plot; False returns the plotly Figure object that can be
            edited or saved using the `write_image()` method (default is True).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """

        # are there any other objects to plot?
        if objects is None:
            n = 1
            _result_list = [self.result.copy()]
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            n = 2
            _result_list = [self.result.copy(), objects.result.copy()]
        else:  # objects as tuple or array
            n = len(objects) + 1
            _result_list = [self.result.copy()]
            for ob in objects:
                if not isinstance(ob, self.__class__):
                    raise TypeError("Some explanations aren't of Break Down class")
                _result_list += [ob.result.copy()]

        deleted_indexes = []
        for i in range(n):
            _result = _result_list[i]

            if len(_result['label'].unique()) > 1:
                n += len(_result['label'].unique()) - 1
                # add new data frames to list
                _result_list += [v for k, v in _result.groupby('label', sort=False)]

                deleted_indexes += [i]

        _result_list = [j for i, j in enumerate(_result_list) if i not in deleted_indexes]
        model_names = [result.iloc[0, result.columns.get_loc("label")] for result in _result_list]

        if vertical_spacing is None:
            vertical_spacing = 0.2 / n

        fig = make_subplots(rows=n, cols=1,
                            shared_xaxes=True, vertical_spacing=vertical_spacing,
                            x_title='contribution', subplot_titles=model_names)
        plot_height = 78 + 71

        if vcolors is None:
            vcolors = get_break_down_colors()

        if min_max is None:
            temp_min_max = [np.Inf, -np.Inf]
        else:
            temp_min_max = min_max

        for i in range(n):
            _result = _result_list[i]

            if _result.shape[0] - 2 <= max_vars:
                m = _result.shape[0]
            else:
                m = max_vars + 3

            if baseline is None:
                baseline = _result.iloc[0, _result.columns.get_loc("cumulative")]

            df = prepare_data_for_break_down_plot(_result, baseline, max_vars, rounding_function, digits)

            measure = ["relative"] * m
            measure[m - 1] = "total"

            fig.add_shape(
                type='line',
                x0=baseline,
                x1=baseline,
                y0=0,
                y1=m - 1,
                yref="paper",
                xref="x",
                line={'color': "#371ea3", 'width': 1.5, 'dash': 'dot'},
                row=i + 1, col=1
            )

            fig.add_waterfall(
                orientation="h",
                measure=measure,
                y=df['variable'].tolist(),
                x=df['contribution'].tolist(),
                textposition="outside",
                text=df['label_text'].tolist(),
                connector={"mode": "spanning", "line": {"width": 1, "color": "#371ea3", "dash": "solid"}},
                decreasing={"marker": {"color": vcolors[-1]}},
                increasing={"marker": {"color": vcolors[1]}},
                totals={"marker": {"color": vcolors[0]}},
                base=baseline,
                hovertext=df['tooltip_text'].tolist(),
                hoverinfo='text+delta',
                hoverlabel={'bgcolor': 'rgba(0,0,0,0.8)'},
                showlegend=False,
                row=i + 1, col=1
            )

            fig.update_yaxes({'type': 'category', 'autorange': 'reversed', 'gridwidth': 2, 'automargin': True,
                              'ticks': "outside", 'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True},
                             row=i + 1, col=1)

            if min_max is None:
                cum = df.cumulative.values
                min_max_margin = cum.ptp() * 0.15
                temp_min_max[0] = np.min([temp_min_max[0], cum.min() - min_max_margin])
                temp_min_max[1] = np.max([temp_min_max[1], cum.max() + min_max_margin])

            fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                              'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                             row=i + 1, col=1)

            plot_height += m * bar_width + (m + 1) * bar_width / 4

        plot_height += (n - 1) * 70

        fig.update_xaxes({'range': temp_min_max})
        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          height=plot_height, margin={'t': 78, 'b': 71, 'r': 30})

        if show:
            fig.show(config={
                'displaylogo': False,
                'staticPlot': False,
                'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d', 'zoom2d', 'pan2d',
                                           'zoomIn2d', 'zoomOut2d', 'resetScale2d', 'toggleSpikelines',
                                           'hoverCompareCartesian', 'hoverClosestCartesian']
            })
        else:
            return fig
