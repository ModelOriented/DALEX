import plotly.express as px

from .checks import *
from .utils import aggregate_profiles
from ..._explainer.theme import get_default_colors, fig_update_line_plot


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
                 intercept=True,
                 random_state=None):
        """
        Constructor for AggregatedProfiles.

        :param variables: str or list or numpy.ndarray or pandas.Series, if not None then aggregate only for selected variables will be calculated, if None all will be selected
        :param groups: str or list or numpy.ndarray or pandas.Series, a variable names that will be used for grouping
        :param type: str, either partial/conditional/accumulated for partial dependence, conditional profiles of accumulated local effects
        :param span: float, smoothing coeffcient, by default 0.25. It's the sd for gaussian kernel
        :param variable_type: str, If numerical then only numerical variables will be calculated. If categorical then only categorical variables will be calculated.

        :return None
        """

        check_variable_type(variable_type)
        variables_ = check_variables(variables)
        groups_ = check_groups(groups)

        self.variable_type = variable_type
        self.groups = groups_
        self.type = type
        self.variables = variables_
        self.span = span
        self.intercept = intercept
        self.result = None
        self.mean_prediction = None
        self.random_state = random_state

    def fit(self,
            ceteris_paribus,
            verbose=True):

        # are there any other cp?
        from dalex.instance_level import CeterisParibus
        if isinstance(ceteris_paribus, CeterisParibus):  # allow for ceteris_paribus to be a single element
            all_profiles = ceteris_paribus.result.copy()
            all_observations = ceteris_paribus.new_observation.copy()
        elif isinstance(ceteris_paribus, list) or isinstance(ceteris_paribus,
                                                             tuple):  # ceteris_paribus as tuple or array
            all_profiles = None
            all_observations = None
            for cp in ceteris_paribus:
                if not isinstance(cp, CeterisParibus):
                    raise TypeError("Some explanations aren't of CeterisParibus class")
                all_profiles = pd.concat([all_profiles, cp.result.copy()])
                all_observations = pd.concat([all_observations, cp.new_observation.copy()])
        else:
            raise TypeError(
                "'ceteris_paribus' should be either Ceteris Paribus object or list/tuple of CeterisParbus objects")

        all_variables = prepare_all_variables(all_profiles, self.variables)

        all_profiles, vnames = prepare_numerical_categorical(all_variables, all_profiles, self.variable_type)

        # select only suitable variables
        all_profiles = all_profiles.loc[all_profiles['_vname_'].isin(vnames), :]

        all_profiles = create_x(all_profiles, self.variable_type)

        self.result = aggregate_profiles(all_profiles, self.type, self.groups, self.intercept,
                                         self.span, verbose)

        self.mean_prediction = all_observations['_yhat_'].mean()

    def plot(self, objects=None, variables=None, size=2, alpha=1,
             facet_ncol=2, title="Aggregated Profiles", title_x='prediction',
             horizontal_spacing=0.05, vertical_spacing=None, show=True):
        """
        Plot function for AggregatedProfiles class.

        :param objects: object of AggregatedProfiles class or list or tuple containing such objects
        :param variables: str list, if not None then only variables will be presented
        :param size: float, width of lines
        :param alpha: float, opacity of lines
        :param facet_ncol: int, number of columns on the plot grid
        :param title: str, the plot's title
        :param title_x: str, x axis title
        :param horizontal_spacing: ratio of horizontal space between the plots, by default it's 0.1
        :param vertical_spacing: ratio of vertical space between the plots, by default it's 0.3/`number of plots`
        :param show: True shows the plot, False returns the plotly Figure object that can be edited or saved using `write_image()` method

        :return None or plotly Figure (see :param show)
        """
        # TODO: numerical+categorical in one plot https://github.com/plotly/plotly.py/issues/2647

        if isinstance(variables, str):
            variables = (variables,)

        # are there any other objects to plot?
        if objects is None:
            _result_df = self.result.assign(_mp_=self.mean_prediction)
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            _result_df = pd.concat([self.result.assign(_mp_=self.mean_prediction),
                                    objects.result.assign(_mp_=objects.mean_prediction)])
        else:  # objects as tuple or array
            _result_df = self.result.assign(_mp_=self.mean_prediction)
            for ob in objects:
                if not isinstance(ob, self.__class__):
                    raise TypeError("Some explanations aren't of AggregatedProfiles class")
                _result_df = pd.concat([_result_df, ob.result.assign(_mp_=ob.mean_prediction)])

        # variables to use
        all_variables = _result_df['_vname_'].dropna().unique().tolist()

        if variables is not None:
            all_variables = np.intersect1d(all_variables, variables).tolist()
            if len(all_variables) == 0:
                raise TypeError("variables do not overlap with " + ''.join(variables))

            _result_df = _result_df.loc[_result_df['_vname_'].isin(all_variables), :]

        is_x_numeric = pd.api.types.is_numeric_dtype(_result_df['_x_'])
        n = len(all_variables)

        facet_nrow = int(np.ceil(n / facet_ncol))
        if vertical_spacing is None:
            vertical_spacing = 0.2 / facet_nrow
        plot_height = 78 + 71 + facet_nrow * (280 + 60)

        color = '_label_'  # _groups_ doesnt make much sense for multiple AP objects
        m = len(_result_df[color].dropna().unique())

        if is_x_numeric:
            fig = px.line(_result_df,
                          x="_x_", y="_yhat_", color=color, facet_col="_vname_",
                          labels={'_yhat_': 'prediction', '_mp_': 'mean_prediction'},  # , color: 'group'},
                          hover_name=color,
                          hover_data={'_yhat_': ':.3f', '_mp_': ':.3f',
                                      color: False, '_vname_': False, '_x_': False},
                          facet_col_wrap=facet_ncol,
                          facet_row_spacing=vertical_spacing,
                          facet_col_spacing=horizontal_spacing,
                          template="none",
                          color_discrete_sequence=get_default_colors(m, 'line')) \
                    .update_traces(dict(line_width=size, opacity=alpha)) \
                    .update_xaxes({'matches': None, 'showticklabels': True,
                                   'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3}) \
                    .update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3})
        else:
            fig = px.bar(_result_df,
                         x="_x_", y="_yhat_", color="_label_", facet_col="_vname_",
                         labels={'_yhat_': 'prediction', '_mp_': 'mean_prediction'},  # , color: 'group'},
                         hover_name=color,
                         hover_data={'_yhat_': ':.3f', '_mp_': ':.3f',
                                     color: False, '_vname_': False, '_x_': False},
                         facet_col_wrap=facet_ncol,
                         facet_row_spacing=vertical_spacing,
                         facet_col_spacing=horizontal_spacing,
                         template="none",
                         color_discrete_sequence=get_default_colors(m, 'line'),  # bar was forgotten
                         barmode='group')  \
                    .update_xaxes({'matches': None, 'showticklabels': True,
                                   'type': 'category', 'gridwidth': 2, 'autorange': 'reversed', 'automargin': True,
                                   'ticks': "outside", 'tickcolor': 'white', 'ticklen': 10}) \
                    .update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3})

        fig = fig_update_line_plot(fig, title, title_x, plot_height, 'x unified')

        if show:
            fig.show(config={'displaylogo': False, 'staticPlot': False,
                             'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d',
                                                        'zoom2d', 'pan2d',
                                                        'zoomIn2d', 'zoomOut2d', 'resetScale2d', 'toggleSpikelines',
                                                        'hoverCompareCartesian',
                                                        'hoverClosestCartesian']})
        else:
            return fig
