from warnings import warn
from plotly.subplots import make_subplots

from .checks import *
from .utils import local_interactions
from ..._explainer.theme import get_break_down_colors


class BreakDown:
    def __init__(self,
                 type='break_down',
                 keep_distributions=False,
                 order=None,
                 interaction_preference=1):
        # TODO interactions / interactions preference
        # TODO checks

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
             bd_list=None,
             baseline=None,
             max_vars=10,
             digits=3,
             rounding_function=np.around,
             bar_width=16,
             min_max=None,
             vcolors=None,
             title="Break Down"):
        """
        Plot function for BreakDown class.

        :param bd_list: object of BreakDown class or list or tuple containing such objects
        :param baseline: float, starting point of bars
        :param max_vars: int, maximum number of variables that shall be presented for for each model
        :param digits: int, number of columns in the plot grid
        :param rounding_function: a function to be used for rounding numbers
        :param bar_width: float, width of bars
        :param min_max: 2-tuple of float values, range of x-axis
        :param vcolors: 3-tuple of str values, color of bars
        :param title: str, the plot's title
        """

        deleted_indexes = []

        # are there any other explanations to plot?
        if bd_list is None:
            n = 1
            _result_list = [self.result]
        elif isinstance(bd_list, BreakDown):  # allow for list to be a single element
            n = 2
            _result_list = [self.result, bd_list.result]
        else:  # list as tuple or array
            n = len(bd_list) + 1
            _result_list = [self.result]
            for bd in bd_list:
                if not isinstance(bd, BreakDown):
                    raise TypeError("Some explanations aren't of Break Down class")
                _result_list += [bd.result]

        for i in range(n):
            result = _result_list[i]

            if len(result['label'].unique()) > 1:
                n += len(result['label'].unique()) - 1
                # add new data frames to list
                _result_list += [v for k, v in result.groupby('label', sort=False)]

                deleted_indexes += [i]

        _result_list = [j for i, j in enumerate(_result_list) if i not in deleted_indexes]
        model_names = [result.iloc[0, result.columns.get_loc("label")] for result in _result_list]

        fig = make_subplots(rows=n, cols=1,
                            shared_xaxes=True, vertical_spacing=0.2/n,
                            x_title='contribution', subplot_titles=model_names)
        plot_height = 78 + 71

        if vcolors is None:
            vcolors = get_break_down_colors()

        if min_max is None:
            temp_min_max = [np.Inf, -np.Inf]
        else:
            temp_min_max = min_max

        for i in range(n):
            result = _result_list[i]

            if result.shape[0] - 2 <= max_vars:
                m = result.shape[0]
            else:
                m = max_vars + 3

            if baseline is None:
                baseline = result.iloc[0, result.columns.get_loc("cumulative")]

            df = prepare_data_for_break_down_plot(result, baseline, max_vars, rounding_function, digits)

            measure = ["relative"]*m
            measure[m-1] = "total"

            fig.add_shape(
                type='line',
                x0=baseline,
                x1=baseline,
                y0=0,
                y1=m-1,
                yref="paper",
                xref="x",
                line={'color': "#371ea3", 'width': 1.5, 'dash': 'dot'},
                row=i+1, col=1
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
                row=i+1, col=1
            )

            fig.update_yaxes({'type': 'category', 'autorange': 'reversed', 'gridwidth': 2, 'automargin': True,
                              'ticks': "outside", 'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True},
                             row=i+1, col=1)

            if min_max is None:
                cum = df.cumulative.values
                min_max_margin = cum.ptp() * 0.15
                temp_min_max[0] = np.min([temp_min_max[0], cum.min() - min_max_margin])
                temp_min_max[1] = np.max([temp_min_max[1], cum.max() + min_max_margin])

            fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                              'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                             row=i+1, col=1)

            plot_height += m*bar_width + (m+1)*bar_width/4

        plot_height += (n-1)*70

        fig.update_xaxes({'range': temp_min_max})
        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          height=plot_height, margin={'t': 78, 'b': 71, 'r': 30})

        fig.show(config={
            'displaylogo': False,
            'staticPlot': False,
            'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d', 'zoom2d', 'pan2d',
                                       'zoomIn2d', 'zoomOut2d', 'resetScale2d', 'toggleSpikelines',
                                       'hoverCompareCartesian', 'hoverClosestCartesian']
        })


def prepare_data_for_break_down_plot(x, baseline, max_vars, rounding_function, digits):

    x.loc[x["variable_name"] == "", "variable_name"] = "prediction"

    temp = x.iloc[[0, x.shape[0] - 1], :].copy()
    x = x.drop([0, x.shape[0] - 1])

    variable_count = x.shape[0]

    if variable_count > max_vars:
        new_x = x.iloc[0:(max_vars+1), :].copy()
        new_x.iloc[max_vars, new_x.columns.get_loc('variable')] = "+ all other factors"
        new_x.iloc[max_vars, new_x.columns.get_loc('contribution')] =\
            np.sum(x.iloc[max_vars:(variable_count - 1), x.columns.get_loc('contribution')])
        new_x.iloc[max_vars, new_x.columns.get_loc('cumulative')] =\
            x.iloc[variable_count - 1, x.columns.get_loc('cumulative')]

        x = new_x

    x = pd.concat((temp.iloc[[0]], x, temp.iloc[[1]]))

    # fix contribution and sign
    x.iloc[[0, x.shape[0] - 1], x.columns.get_loc("contribution")] -= baseline

    # use for text label and tooltip
    x.loc[:, 'contribution'] = rounding_function(x.loc[:, 'contribution'], digits)
    x.loc[:, 'cumulative'] = rounding_function(x.loc[:, 'cumulative'], digits)

    x['tooltip_text'] = x.apply(lambda row: tooltip_text(row), axis=1)
    x.iloc[[0, x.shape[0] - 1], x.columns.get_loc('tooltip_text')] = "Average response: " + str(
        x.iloc[0, x.columns.get_loc('cumulative')]) + "<br>Prediction: " + str(
        x.iloc[x.shape[0] - 1, x.columns.get_loc('cumulative')])

    x['label_text'] = label_text(x.iloc[:, x.columns.get_loc("contribution")].tolist())
    x.iloc[0, x.columns.get_loc("label_text")] = x.iloc[0, x.columns.get_loc('cumulative')]
    x.iloc[x.shape[0] - 1, x.columns.get_loc("label_text")] = x.iloc[x.shape[0]-1, x.columns.get_loc('cumulative')]

    return x


def tooltip_text(row):
    if row.contribution > 0:
        key_word = "increases"
    else:
        key_word = "decreases"
    return row.variable + "<br>" + key_word + " average response <br>by"


def label_text(contribution):
    def to_text(x):
        if x > 0:
            return "+" + str(x)
        else:
            return str(x)

    return [to_text(c) for c in contribution]

