from .checks import *
from .utils import aggregate_profiles

from plotly.subplots import make_subplots
from ...dataset_level.variable_importance.variable_importance import get_colors

class AggregatedProfiles:
    """
    The class for calculating an aggregation of ceteris paribus profiles.
    It can be: Partial Dependency Profile (average across Ceteris Paribus Profiles),
    Conditional Dependency Profile (local weighted average across Ceteris Paribus Profiles) or
    Accumulated Local Dependency Profile (cummulated average local changes in Ceteris Paribus Profiles).
    """

    def __init__(self,
                 type='partial',
                 variables=None,
                 variable_type='numerical',
                 groups=None,
                 span=0.25,
                 intercept=True):
        """
        Constructor for AggregatedProfiles.

        :param variables: list, variables if not None then aggregate only for selected variables will be calculated
        :param groups: str, a variable name that will be used for grouping
        :param type: str, either partial/conditional/accumulated for partial dependence, conditional profiles of accumulated local effects
        :param span: float, smoothing coeffcient, by default 0.25. It's the sd for gaussian kernel
        :param variable_type: str, If numerical then only numerical variables will be calculated. If categorical then only categorical variables will be calculated.
        """

        check_variable_type(variable_type)
        groups = check_groups(groups)

        self.variable_type = variable_type
        self.groups = groups
        self.type = type
        self.variables = variables
        self.span = span
        self.intercept = intercept
        self.result = None
        self.mean_prediction = None

    def fit(self,
            ceteris_paribus):

        # are there any other cp?
        from dalex.instance_level import CeterisParibus
        if isinstance(ceteris_paribus, CeterisParibus):  # allow for ceteris_paribus to be a single element
            all_profiles = ceteris_paribus.result.copy()
        elif isinstance(ceteris_paribus, list) or isinstance(ceteris_paribus,
                                                             tuple):  # ceteris_paribus as tuple or array
            all_profiles = None
            for cp in ceteris_paribus:
                if not isinstance(cp, CeterisParibus):
                    raise TypeError("Some explanations aren't of CeterisParibus class")
                all_profiles = pd.concat([all_profiles, cp.result.copy()])
        else:
            raise TypeError(
                "'ceteris_paribus' should be either Ceteris Paribus object or list/tuple of CeterisParbus objects")

        all_variables = prepare_all_variables(all_profiles, self.variables)

        all_profiles, vnames = prepare_numerical_categorical(all_variables, all_profiles, self.variable_type)

        # select only suitable variables
        all_profiles = all_profiles.loc[all_profiles['_vname_'].isin(vnames), :]

        all_profiles = create_x(all_profiles, self.variable_type)

        self.result = aggregate_profiles(all_profiles, ceteris_paribus, self.type, self.groups, self.intercept, self.span)
        self.mean_prediction = 0

    def plot(self, ap_list=None, size=2, facet_ncol=2, variables=None,
             chart_title="Aggregated Profiles"):

        # are there any other explanation to plot?
        if ap_list is None:
            m = 1
            _result_df = self.result
        elif isinstance(ap_list, AggregatedProfiles):  # allow for list to be a single element
            m = 2
            _result_df = pd.concat([self.result, ap_list.result])
        else:  # list as tuple or array
            m = len(ap_list) + 1
            _result_df = self.result
            for ap in ap_list:
                if not isinstance(ap, AggregatedProfiles):
                    raise TypeError("Some explanations aren't of AggregatedProfiles class")
                _result_df = pd.concat([_result_df, ap.result])

        # variables to use
        all_variables = _result_df['_vname_'].dropna().unique().tolist()

        if variables is not None:
            all_variables = np.intersect1d(all_variables, variables).tolist()
            if len(all_variables) == 0:
                raise TypeError("variables do not overlap with " + ''.join(variables))

            _result_df = _result_df.loc[_result_df['_vname_'].isin(all_variables), :]

        variable_names = all_variables
        n = len(variable_names)
        is_x_numeric = not pd.api.types.is_string_dtype(_result_df['_x_'])

        dl = _result_df['_yhat_'].to_numpy()
        min_max = [np.Inf, -np.Inf]
        min_max_margin = dl.ptp() * 0.15
        min_max[0] = dl.min() - min_max_margin
        min_max[1] = dl.max() + min_max_margin

        # split var by variable
        var_df_list = [v for k, v in _result_df.groupby('_vname_', sort=False)]
        var_df_dict = {e['_vname_'].array[0]: e for e in var_df_list}

        facet_nrow = int(np.ceil(n / facet_ncol))
        fig = make_subplots(rows=facet_nrow, cols=facet_ncol, horizontal_spacing=0.1,
                            vertical_spacing=0.3/n, x_title='prediction', subplot_titles=variable_names)

        colors = get_colors(m, 'line')

        for i in range(n):
            name = variable_names[i]
            var_df = var_df_dict[name]

            row = int(np.floor(i/facet_ncol) + 1)
            col = int(np.mod(i, facet_ncol) + 1)

            # line plot or bar plot? TODO: add is_numeric and implement 'both'
            if is_x_numeric:
                ret = var_df[['_x_', "_yhat_", "_vname_", "_label_"]].rename(
                    columns={'_x_': "xhat", "_yhat_": "yhat", "_vname_": "vname", "_label_": "label"})
                ret["xhat"] = pd.to_numeric(ret["xhat"])
                ret["yhat"] = pd.to_numeric(ret["yhat"])
                ret = ret.sort_values('xhat')

                df_list = [v for k, v in ret.groupby('label', sort=False)]

                for j in range(len(df_list)):
                    df = df_list[j]

                    tt = df.apply(lambda r: tooltip_text(r), axis=1)
                    df = df.assign(tooltip_text=tt.values)

                    fig.add_scatter(
                        mode='lines',
                        y=df['yhat'].tolist(),
                        x=df['xhat'].tolist(),
                        line={'color': colors[j], 'width': size, 'shape': 'spline'},
                        hovertext=df['tooltip_text'].tolist(),
                        hoverinfo='text',
                        hoverlabel={'bgcolor': 'rgba(0,0,0,0.8)'},
                        showlegend=False,
                        row=row, col=col
                    )

                fig.update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                  'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                                 row=row, col=col)

                fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                  'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                                 row=row, col=col)

                fig.update_yaxes({'range': min_max})

            else:

                ret = var_df[['_x_', "_yhat_", "_vname_", "_label_"]].rename(
                    columns={'_x_': "xhat", "_yhat_": "yhat", "_vname_": "vname", "_label_": "label"})
                ret["yhat"] = pd.to_numeric(ret["yhat"])
                ret = ret.sort_values('xhat')

                baseline = 0

                df_list = [v for k, v in ret.groupby('label', sort=False)]

                for j in range(len(df_list)):
                    df = df_list[j]

                    difference = df['yhat'] - baseline
                    df = df.assign(difference=difference.values)

                    # lt = df.apply(lambda r: label_text(r), axis=1)
                    # df = df.assign(label_text=lt.values)

                    tt = df.apply(lambda r: tooltip_text(r), axis=1)
                    df = df.assign(tooltip_text=tt.values)

                    fig.add_shape(type='line', x0=baseline, x1=baseline, y0=0, y1=len(df['xhat'].unique()) - 1, yref="paper", xref="x",
                                  line={'color': "#371ea3", 'width': 1.5, 'dash': 'dot'}, row=row, col=col)

                    fig.add_bar(
                        orientation="h",
                        y=df['xhat'].tolist(),
                        x=df['difference'].tolist(),
                        # textposition="outside",
                        # text=df['label_text'].tolist(),
                        marker_color=colors[j],
                        base=baseline,
                        hovertext=df['tooltip_text'].tolist(),
                        hoverinfo='text',
                        hoverlabel={'bgcolor': 'rgba(0,0,0,0.8)'},
                        showlegend=False,
                        row=row, col=col)

                fig.update_yaxes({'type': 'category', 'autorange': 'reversed', 'gridwidth': 2, 'automargin': True,
                                  'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True},
                                 row=row, col=col)

                fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                  'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                                 row=row, col=col)

                fig.update_xaxes({'range': min_max})

        plot_height = 78 + 71 + facet_nrow*(280+60)
        fig.update_layout(title_text=chart_title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          height=plot_height, margin={'t': 78, 'b': 71, 'r': 30}, hovermode='closest')

        fig.show(config={'displaylogo': False, 'staticPlot': False,
            'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d', 'zoom2d', 'pan2d',
                                       'zoomIn2d', 'zoomOut2d', 'resetScale2d', 'toggleSpikelines', 'hoverCompareCartesian',
                                       'hoverClosestCartesian']})


def tooltip_text(r):
    return "todo"