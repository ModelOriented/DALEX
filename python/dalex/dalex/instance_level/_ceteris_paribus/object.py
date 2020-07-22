from warnings import warn

import plotly.express as px

from .checks import *
from .utils import calculate_ceteris_paribus
from .plot import tooltip_text
from ..._explainer.theme import get_default_colors, fig_update_line_plot


class CeterisParibus:
    def __init__(self,
                 variables=None,
                 grid_points=101,
                 variable_splits=None,
                 processes=1):
        """
        Creates Ceteris Paribus object

        :param variables: variables for which the profiles are calculated
        :param grid_points: number of points in a single variable split if calculated automatically
        :param variable_splits: mapping of variables into points the profile will be calculated, if None then calculate with the function `_calculate_variable_splits`
        :param processes: integer, number of parallel processes, iterated over variables

        :return None
        """

        processes_ = check_processes(processes)

        self.variables = variables
        self.grid_points = grid_points
        self.variable_splits = variable_splits
        self.result = None
        self.new_observation = None
        self.processes = processes_

    def fit(self,
            explainer,
            new_observation,
            y=None,
            verbose=True):

        self.variables = check_variables(self.variables, explainer, self.variable_splits)

        check_data(explainer.data, self.variables)

        self.new_observation = check_new_observation(new_observation, explainer)

        self.variable_splits = check_variable_splits(self.variable_splits, self.variables, explainer, self.grid_points)

        y = check_y(y)

        self.result, self.new_observation = calculate_ceteris_paribus(explainer,
                                                                      self.new_observation,
                                                                      self.variable_splits,
                                                                      y,
                                                                      self.processes,
                                                                      verbose)

    def plot(self, objects=None, variable_type="numerical", variables=None, size=2, alpha=1, color="_label_", facet_ncol=2,
             show_observations=True, show_rugs=True, title="Ceteris Paribus Profiles", title_x='prediction',
             horizontal_spacing=0.05, vertical_spacing=None, show=True):
        """
        Plot function for CeterisParibus class.

        :param objects: object of CeterisParibus class or list or tuple containing such objects
        :param variable_type: either "numerical" or "categorical", determines type of variables to plot
        :param variables: str list, if not None then only variables will be presented
        :param size: int, width of lines
        :param alpha: float, opacity of lines
        :param color: string, variable name for groups, by default `_label_` which groups by models
        :param facet_ncol: int, number of columns on the plot grid
        :param show_observations show observation points
        :param show_rugs show observation points rugs
        :param title: str, the plot's title
        :param title_x: str, x axis title
        :param horizontal_spacing: ratio of horizontal space between the plots, by default it's 0.1
        :param vertical_spacing: ratio of vertical space between the plots, by default it's 0.3/`number of plots`
        :param show: True shows the plot, False returns the plotly Figure object that can be edited or saved using `write_image()` method

        :return None or plotly Figure (see :param show)
        """

        # TODO: numerical+categorical in one plot https://github.com/plotly/plotly.py/issues/2647
        # TODO: show_observations and show_rugs (when _original_ is fixed) + tooltip data

        if variable_type not in ("numerical", "categorical"):
            raise TypeError("variable_type should be 'numerical' or 'categorical'")
        if isinstance(variables, str):
            variables = (variables,)

        # are there any other objects to plot?
        if objects is None:
            _result_df = self.result.copy()
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            _result_df = pd.concat([self.result.copy(), objects.result.copy()])
        else:  # objects as tuple or array
            _result_df = self.result.copy()
            for ob in objects:
                if not isinstance(ob, self.__class__):
                    raise TypeError("Some explanations aren't of CeterisParibus class")
                _result_df = pd.concat([_result_df, ob.result.copy()])

        # variables to use
        all_variables = list(_result_df['_vname_'].dropna().unique())

        if variables is not None:
            all_variables = np.intersect1d(all_variables, variables).tolist()
            if len(all_variables) == 0:
                raise TypeError("variables do not overlap with " + ''.join(variables))

        # names of numeric variables
        numeric_variables = _result_df[all_variables].select_dtypes(include=np.number).columns.tolist()

        if variable_type == "numerical":
            variable_names = numeric_variables

            if len(variable_names) == 0:
                # change to categorical
                variable_type = "categorical"
                # send message
                warn("'variable_type' changed to 'categorical' due to lack of numerical variables.")
                # take all
                variable_names = all_variables
            elif variables is not None and len(variable_names) != len(variables):
                raise TypeError("There are no numerical variables")
        else:
            variable_names = np.setdiff1d(all_variables, numeric_variables).tolist()

            # there are variables selected
            if variables is not None:
                # take all
                variable_names = all_variables
            elif len(variable_names) == 0:
                # there were no variables selected and there are no categorical variables
                raise TypeError("There are no non-numerical variables.")

        # prepare profiles data
        _result_df = _result_df.loc[_result_df['_vname_'].apply(lambda x: x in variable_names), ].reset_index(drop=True)

        # create _x_
        for variable in _result_df['_vname_'].unique():
            where_variable = _result_df['_vname_'] == variable
            _result_df.loc[where_variable, '_x_'] = _result_df.loc[where_variable, variable]

        # change x column to proper character values
        if variable_type == 'categorical':
            _result_df.loc[:, '_x_'] = _result_df.apply(lambda row: str(row[row['_vname_']]), axis=1)

        n = len(variable_names)
        if vertical_spacing is None:
            vertical_spacing = 0.3 / n
        facet_nrow = int(np.ceil(n / facet_ncol))

        plot_height = 78 + 71 + facet_nrow * (280 + 60)

        m = len(_result_df[color].dropna().unique())

        _result_df[color] = _result_df[color].astype(object)  # prevent error when using pd.StringDtype
        _result_df = _result_df.assign(_text_=_result_df.apply(lambda obs: tooltip_text(obs), axis=1))

        if variable_type == "numerical":

            fig = px.line(_result_df,
                          x="_x_", y="_yhat_", color=color, facet_col="_vname_", line_group='_ids_',
                          labels={'_yhat_': 'prediction', '_label_': 'label', '_ids_': 'id'},  # , color: 'group'},
                          # hover_data={'_text_': True, '_yhat_': ':.3f', '_vname_': False, '_x_': False, color: False},
                          custom_data=['_text_'],
                          facet_col_wrap=facet_ncol,
                          facet_row_spacing=vertical_spacing,
                          facet_col_spacing=horizontal_spacing,
                          template="none",
                          color_discrete_sequence=get_default_colors(m, 'line')) \
                    .update_traces(dict(line_width=size, opacity=alpha),
                                   hovertemplate="%{customdata[0]}<extra></extra>") \
                    .update_xaxes({'matches': None, 'showticklabels': True,
                                   'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3}) \
                    .update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3})

            if show_observations:
                pass

            if show_rugs:
                pass
        else:
            if len(_result_df['_ids_'].unique()) > 1:  # https://github.com/plotly/plotly.py/issues/2657
                raise TypeError("Please pick one observation per label.")

            fig = px.bar(_result_df,
                         x="_x_", y="_yhat_", color="_label_", facet_col="_vname_",
                         labels={'_yhat_': 'prediction', '_label_': 'label', '_ids_': 'id'},  # , color: 'group'},
                         # hover_data={'_yhat_': ':.3f', '_ids_': True, '_vname_': False, color: False},
                         custom_data=['_text_'],
                         facet_col_wrap=facet_ncol,
                         facet_row_spacing=vertical_spacing,
                         facet_col_spacing=horizontal_spacing,
                         template="none",
                         color_discrete_sequence=get_default_colors(m, 'line'),  # bar was forgotten
                         barmode='group')  \
                    .update_traces(dict(opacity=alpha),
                                   hovertemplate="%{customdata[0]}<extra></extra>") \
                    .update_xaxes({'matches': None, 'showticklabels': True,
                                   'type': 'category', 'gridwidth': 2, 'autorange': 'reversed', 'automargin': True,
                                   'ticks': "outside", 'tickcolor': 'white', 'ticklen': 10}) \
                    .update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3})

        fig = fig_update_line_plot(fig, title, title_x, plot_height, 'closest')

        fig.update_layout(
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)')
        )

        if show:
            fig.show(config={'displaylogo': False, 'staticPlot': False,
                             'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d',
                                                        'zoom2d', 'pan2d',
                                                        'zoomIn2d', 'zoomOut2d', 'resetScale2d', 'toggleSpikelines',
                                                        'hoverCompareCartesian',
                                                        'hoverClosestCartesian']})
        else:
            return fig
