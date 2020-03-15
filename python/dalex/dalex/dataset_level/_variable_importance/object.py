import pandas as pd
from plotly.subplots import make_subplots

from .checks import *
from .utils import calculate_variable_importance
from ..._explainer.theme import get_default_colors


class VariableImportance:
    def __init__(self,
                 loss_function='loss_root_mean_square',
                 type=("variable_importance","ratio","difference"),
                 n_sample=None,
                 B=10,
                 variables=None,
                 variable_groups=None,
                 random_state=None,
                 keep_raw_permutations=None):
        """
        Calculate feature importance of the model

        :param loss_function: a function thet will be used to assess variable importance
        :param type: type of transformation that should be applied for dropout loss
        :param n_sample: number of observations that should be sampled for calculation of variable importance
        :param B: number of permutation rounds to perform on each variable
        :param variables: vector of variables. If None then variable importance will be tested for each variable from the data separately
        :param variable_groups: list of variables names vectors. This is for testing joint variable importance
        :param random_state: random state for the permutations
        :param keep_raw_permutations: TODO
        :return: None
        """

        loss_function = check_loss_function(loss_function)
        variable_groups = check_variable_groups(variable_groups)
        B = check_B(B)
        type = check_type(type)
        random_state = check_random_state(random_state)
        keep_raw_permutations = check_keep_raw_permutations(keep_raw_permutations, B)

        self.loss_function = loss_function
        self.type = type
        self.n_sample = n_sample
        self.B = B
        self.variables = variables
        self.variable_groups = variable_groups
        self.random_state = random_state
        self.keep_raw_permutations = keep_raw_permutations
        self.result= None
        self.permutation = None

    def fit(self, explainer):
        # if `variable_groups` are not specified, then extract from `variables`
        self.variables = check_variables(self.variables, self.variable_groups, explainer)
        self.result, self.permutation = calculate_variable_importance(explainer,
                                                                      self.type,
                                                                      self.loss_function,
                                                                      self.variables,
                                                                      self.n_sample,
                                                                      self.B,
                                                                      explainer.label,
                                                                      self.keep_raw_permutations)

    def plot(self,
             objects=None,
             max_vars=10,
             digits=3,
             rounding_function=np.around,
             bar_width=16,
             split=("model", "variable"),
             title="Variable Importance"):
        """
        Plot function for VariableImportance class.

        :param objects: object of VariableImportance class or list or tuple containing such objects
        :param max_vars: int, maximum number of variables that shall be presented for for each model
        :param digits: int, number of columns in the plot grid
        :param rounding_function: a function to be used for rounding numbers
        :param bar_width: float, width of bars
        :param split: either "model" or "variable", determines the plot layout
        :param title: str, the plot's title
        """

        if isinstance(split, tuple):
            split = split[0]

        if split not in ("model", "variable"):
            raise TypeError("split should be 'model' or 'variable'")

        # are there any other objects to plot?
        if objects is None:
            n = 1
            _result_df = self.result.copy()
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            n = 2
            _result_df = pd.concat([self.result.copy(), objects.result.copy()])
        else:  # objects as tuple or array
            n = len(objects) + 1
            _result_df = self.result.copy()
            for ob in objects:
                if not isinstance(ob, self.__class__):
                    raise TypeError("Some explanations aren't of VariableImportance class")
                _result_df = pd.concat([_result_df, ob.result].copy())

        dl = _result_df.loc[_result_df.variable != '_baseline_', 'dropout_loss'].to_numpy()
        min_max_margin = dl.ptp() * 0.15
        min_max = [dl.min() - min_max_margin, dl.max() + min_max_margin]

        # take out full model
        best_fits = _result_df[_result_df.variable == '_full_model_']

        # this produces dropout_loss_x and dropout_loss_y columns
        _result_df = _result_df.merge(best_fits[['label', 'dropout_loss']], how="left", on="label")
        _result_df = _result_df[['label', 'variable', 'dropout_loss_x', 'dropout_loss_y']].rename(
            columns={'dropout_loss_x': 'dropout_loss', 'dropout_loss_y': 'full_model'})

        # remove full_model and baseline
        _result_df = _result_df[(_result_df.variable != '_full_model_') & (_result_df.variable != '_baseline_')]

        # calculate order of bars or variable plots (split = 'variable')
        # get variable permutation
        perm = _result_df[['variable', 'dropout_loss']].groupby('variable').mean().reset_index().\
            sort_values('dropout_loss', ascending=False).variable.values

        plot_height = 78 + 71

        colors = get_default_colors(n, 'bar')

        if split == "model":

            # init plot
            model_names = _result_df['label'].unique().tolist()
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.2/n, x_title='drop-out loss',
                                subplot_titles=model_names)

            # split df by model
            df_list = [v for k, v in _result_df.groupby('label', sort=False)]

            for i in range(n):
                m = df_list[i].shape[0]
                if max_vars is not None and max_vars < m:
                    m = max_vars

                # take only m variables (for max_vars)
                # sort rows of df by variable permutation and drop unused variables
                df = df_list[i].sort_values('dropout_loss').tail(m)\
                    .set_index('variable').reindex(perm).dropna().reset_index()

                baseline = df.iloc[0, df.columns.get_loc('full_model')]

                df = df.assign(difference=lambda x: x['dropout_loss'] - baseline)

                lt = df.apply(lambda row: label_text(row, rounding_function, digits), axis=1)
                df = df.assign(label_text=lt.values)

                tt = df.apply(lambda row: tooltip_text(row, rounding_function, digits), axis=1)
                df = df.assign(tooltip_text=tt.values)

                fig.add_shape(type='line', x0=baseline, x1=baseline, y0=0, y1=m - 1, yref="paper", xref="x",
                    line={'color': "#371ea3", 'width': 1.5, 'dash': 'dot'}, row=i + 1, col=1)

                fig.add_bar(
                    orientation="h",
                    y=df['variable'].tolist(),
                    x=df['difference'].tolist(),
                    textposition="outside",
                    text=df['label_text'].tolist(),
                    marker_color=colors[i],
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

                fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                                  'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                                 row=i + 1, col=1)

                plot_height += m * bar_width + (m + 1) * bar_width / 4 + 30

        elif split == "variable":

            # split df by variable
            df_list = [v for k, v in _result_df.groupby('variable', sort=False)]

            n = len(df_list)
            if max_vars is not None and max_vars < n:
                n = max_vars

            # init plot
            variable_names = perm[0:n]
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.1, x_title='drop-out loss',
                                subplot_titles=variable_names)

            df_dict = {e.variable.array[0]: e for e in df_list}

            # take only n=max_vars elements from df_dict
            for i in range(n):
                df = df_dict[perm[i]]
                m = df.shape[0]

                baseline = 0

                df = df.assign(difference=lambda x: x['dropout_loss'] - x['full_model'])

                lt = df.apply(lambda row: label_text(row, rounding_function, digits), axis=1)
                df = df.assign(label_text=lt.values)

                tt = df.apply(lambda row: tooltip_text(row, rounding_function, digits), axis=1)
                df = df.assign(tooltip_text=tt.values)

                fig.add_shape(type='line', x0=baseline, x1=baseline, y0=0, y1=m - 1, yref="paper", xref="x",
                    line={'color': "#371ea3", 'width': 1.5, 'dash': 'dot'}, row=i + 1, col=1)

                fig.add_bar(
                    orientation="h",
                    y=df['label'].tolist(),
                    x=df['dropout_loss'].tolist(),
                    # textposition="outside",
                    # text=df['label_text'].tolist(),
                    marker_color=colors,
                    base=baseline,
                    hovertext=df['tooltip_text'].tolist(),
                    hoverinfo='text',
                    hoverlabel={'bgcolor': 'rgba(0,0,0,0.8)'},
                    showlegend=False,
                    row=i + 1, col=1)

                fig.update_yaxes({'type': 'category', 'autorange': 'reversed', 'gridwidth': 2, 'automargin': True,
                                  'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True},
                                 row=i + 1, col=1)

                fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                                  'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                                 row=i + 1, col=1)

                plot_height += m * bar_width + (m + 1) * bar_width / 4

        plot_height += (n - 1) * 70

        fig.update_xaxes({'range': min_max})
        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          height=plot_height, margin={'t': 78, 'b': 71, 'r': 30})

        fig.show(config={'displaylogo': False, 'staticPlot': False,
                         'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d', 'zoom2d',
                                                    'pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d',
                                                    'toggleSpikelines', 'hoverCompareCartesian',
                                                    'hoverClosestCartesian']})


def label_text(row, rounding_function, digits):
    if row.difference > 0:
        key_word = "+"
    else:
        key_word = ""
    return key_word + str(rounding_function(np.abs(row.difference), digits))


def tooltip_text(row, rounding_function, digits):
    if row.difference > 0:
        key_word = "+"
    else:
        key_word = ""
    return "Model: " + row.label + " loss after<br>variable: " + row.variable + " is permuted: " +\
           str(rounding_function(row.dropout_loss, digits)) + "<br>" +\
           "Drop-out loss change: " + key_word + str(rounding_function(np.abs(row.difference), digits))

