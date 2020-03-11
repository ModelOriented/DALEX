import numpy as np
from plotly.subplots import make_subplots

from .checks import *
from .utils import shap


class Shap:
    def __init__(self,
                 path=None,
                 keep_distributions=True,
                 B=25):
        # TODO interactions / interactions preference
        # TODO checks

        self.path = path
        self.keep_distributions = keep_distributions
        self.B = B
        self.result = None
        self.yhats_distributions = None
        self.prediction = None
        self.intercept = None

    def fit(self,
            explainer,
            new_observation):

        new_observation = check_new_observation(new_observation)
        check_columns_in_new_observation(new_observation, explainer)
        self.result, self.prediction, self.intercept, self.yhats_distributions = shap(explainer,
                                                                                      new_observation,
                                                                                      self.path,
                                                                                      self.keep_distributions,
                                                                                      self.B)

    def plot(self,
             sh_list=None,
             baseline=None,
             max_vars=10,
             digits=3,
             rounding_function=np.around,
             bar_width=16,
             min_max=None,
             vcolors=None,
             title="Shapley Values"):

        """
        Plot function for Shap class.

        :param sh_list: object of Shap class or list or tuple containing such objects
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
        if sh_list is None:
            n = 1
            _result_list = [self.result.loc[self.result['B'] == 0, ]]
            _intercept_list = [self.intercept]
            _prediction_list = [self.prediction[0]]
        elif isinstance(sh_list, Shap):  # allow for list to be a single element
            n = 2
            _result_list = [self.result.loc[self.result['B'] == 0, ], sh_list.result.loc[sh_list.result['B'] == 0, ]]
            _intercept_list = [self.intercept, sh_list.intercept]
            _prediction_list = [self.prediction[0], sh_list.prediction[0]]
        else:  # list as tuple or array
            n = len(sh_list) + 1
            _result_list = [self.result.loc[self.result['B'] == 0, ]]
            _intercept_list = [self.intercept]
            _prediction_list = [self.prediction[0]]
            for sh in sh_list:
                if not isinstance(sh, Shap):
                    raise TypeError("Some explanations aren't of Shap class")
                _result_list += [sh.result.loc[sh.result['B'] == 0, ]]
                _intercept_list += [sh.intercept]
                _prediction_list += [sh.prediction[0]]

        # TODO add intercept and prediction list update for multiclass
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

        fig = make_subplots(rows=n, cols=1,
                            shared_xaxes=True, vertical_spacing=0.2/n,
                            x_title='contribution', subplot_titles=model_names)
        plot_height = 78 + 71

        if vcolors is None:
            vcolors = ["#371ea3", "#8bdcbe", "#f05a71"]

        if min_max is None:
            temp_min_max = [np.Inf, -np.Inf]
        else:
            temp_min_max = min_max

        for i in range(n):
            result = _result_list[i]

            if result.shape[0] <= max_vars:
                m = result.shape[0]
            else:
                m = max_vars + 1

            if baseline is None:
                baseline = _intercept_list[i]
            prediction = _prediction_list[i]

            df = prepare_data_for_shap_plot(result, baseline, prediction, max_vars, rounding_function, digits)

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
                row=i+1, col=1
            )

            fig.update_yaxes({'type': 'category', 'autorange': 'reversed', 'gridwidth': 2, 'automargin': True,
                              'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True},
                             row=i+1, col=1)

            fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                              'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                             row=i+1, col=1)

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

        fig.show(config={'displaylogo': False, 'staticPlot': False,
            'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d', 'zoom2d', 'pan2d',
                                       'zoomIn2d', 'zoomOut2d', 'resetScale2d', 'toggleSpikelines', 'hoverCompareCartesian',
                                       'hoverClosestCartesian']})


def prepare_data_for_shap_plot(x, baseline, prediction, max_vars, rounding_function, digits):

    variable_count = x.shape[0]

    if variable_count > max_vars:
        last_row = max_vars - 1
        new_x = x.iloc[0:(last_row + 1), :].copy()
        new_x.iloc[last_row, new_x.columns.get_loc('variable')] = "+ all other factors"
        new_x.iloc[last_row, new_x.columns.get_loc('contribution')] = np.sum(
            x.iloc[last_row:(variable_count - 1), x.columns.get_loc('contribution')])

        x = new_x

    # use for text label and tooltip
    x.loc[:, 'contribution'] = rounding_function(x.loc[:, 'contribution'], digits)
    baseline = rounding_function(baseline, digits)
    prediction = rounding_function(prediction, digits)

    tt = x.apply(lambda row: tooltip_text(row, baseline, prediction), axis=1)
    x = x.assign(tooltip_text=tt.values)

    lt = label_text(x.iloc[:, x.columns.get_loc("contribution")].tolist())
    x = x.assign(label_text=lt)

    return x


def tooltip_text(row, baseline, prediction):
    if row.contribution > 0:
        key_word = "increases"
    else:
        key_word = "decreases"
    return "Average response: " + str(baseline) + "<br>Prediction: " + str(prediction) + "<br>" +\
           row.variable + "<br>" + key_word + " average response <br>by " + str(np.abs(row.contribution))


def label_text(contribution):
    def to_text(x):
        if x > 0:
            return "+" + str(x)
        else:
            return str(x)

    return [to_text(c) for c in contribution]
