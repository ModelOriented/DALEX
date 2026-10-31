import numpy as np
import pandas as pd
import plotly.express as px
import warnings

from . import checks, utils
from ... import _theme, _global_checks, _global_utils
from ..._explanation import Explanation, CompatibilityTable, EmptyTable
import narwhals.stable.v2 as nw


class CeterisParibus(Explanation):
    """Calculate predict-level variable profiles as Ceteris Paribus

    Parameters
    -----------
    variables : array_like of str, optional
        Variables for which the profiles will be calculated
        (default is `None`, which means all of the variables).
    grid_points : int, optional
        Maximum number of points for profile calculations (default is `101`).
        NOTE: The final number of points may be lower than `grid_points`,
        eg. if there is not enough unique values for a given variable.
    variable_splits : dict of lists, optional
        Split points for variables e.g. `{'x': [0, 0.2, 0.5, 0.8, 1], 'y': ['a', 'b']}`
        (default is `None`, which means that they will be calculated using one of
        `variable_splits_type` and the `data` attribute).
    variable_splits_type : {'uniform', 'quantiles'}, optional
        Way of calculating `variable_splits`. Set `'quantiles'` for percentiles.
        (default is `'uniform'`, which means uniform grid of points).
    variable_splits_with_obs: bool, optional
        Add variable values of `new_observation` data to the `variable_splits`
        (default is `True`).
    processes : int, optional
        Number of parallel processes to use in calculations. Iterated over `variables`
        (default is `1`, which means no parallel computation).

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    new_observation : pd.DataFrame
        Observations for which predictions need to be explained.
    variables : array_like of str or None
        Variables for which the profiles will be calculated.
    grid_points : int
        Maximum number of points for profile calculations.
    variable_splits : dict of lists or None
        Split points for variables.
    variable_splits_type : {'uniform', 'quantiles'}
        Way of calculating `variable_splits`.
    variable_splits_with_obs: bool
        Add variable values of `new_observation` data to the `variable_splits`.
    processes : int
        Number of parallel processes to use in calculations. Iterated over `B`.

    Notes
    --------
    - https://pbiecek.github.io/ema/ceterisParibus.html
    """

    def __init__(self,
                 variables=None,
                 grid_points=101,
                 variable_splits=None,
                 variable_splits_type='uniform',
                 variable_splits_with_obs=False,
                 processes=1):

        _processes = checks.check_processes(processes)
        _variable_splits_type = checks.check_variable_splits_type(variable_splits_type)

        self.variables = variables
        self.grid_points = grid_points
        self.variable_splits = variable_splits
        self.variable_splits_type = _variable_splits_type
        self.variable_splits_with_obs = variable_splits_with_obs
        self.result = EmptyTable()
        self.new_observation = None
        self.processes = _processes

    def _repr_html_(self):
        return self.result._repr_html_()

    new_observation = CompatibilityTable()

    def _native_profile_frames(self):
        """Use native partitions until a caller edits the legacy pandas view."""
        from ._table import ProfileTables
        stored = self.__dict__.get('result')
        cached = self.__dict__.get('_compatibility_result')
        if isinstance(stored, ProfileTables) and cached is None:
            return list(zip(stored.variables, stored.frames))
        frame = nw.from_native(cached if cached is not None else stored, eager_only=True)
        variables = frame['_vname_'].unique(maintain_order=True).to_list()
        return [(variable, frame.filter(nw.col('_vname_') == nw.lit(variable)).to_native()) for variable in variables]

    def _native_observation_frame(self):
        from ._table import ObservationTable
        stored = self.__dict__.get('new_observation')
        cached = self.__dict__.get('_compatibility_new_observation')
        if isinstance(stored, ObservationTable) and cached is None:
            return stored.data
        return cached if cached is not None else stored

    def fit(self,
            explainer,
            new_observation,
            y=None,
            verbose=True):
        """Calculate the result of explanation

        Fit method makes calculations in place and changes the attributes.

        Parameters
        -----------
        explainer : Explainer object
            Model wrapper created using the Explainer class.
        new_observation : pd.DataFrame or np.ndarray
            Observations for which predictions need to be explained.
        y : pd.Series or np.ndarray (1d), optional
            Target variable with the same length as `new_observation`.
        verbose : bool, optional
            Print tqdm progress bar (default is `True`).

        Returns
        -----------
        None
        """

        self.variables = checks.check_variables(self.variables, explainer, self.variable_splits)

        checks.check_data(explainer.data, self.variables)

        observation = checks.check_new_observation(new_observation, explainer)
        self._native_features = not isinstance(explainer.data, pd.DataFrame)

        self.variable_splits = checks.check_variable_splits(self.variable_splits,
                                                     self.variables,
                                                     self.grid_points,
                                                     explainer.data,
                                                     self.variable_splits_type,
                                                     self.variable_splits_with_obs,
                                                     observation)

        y = checks.check_y(y)

        self._native_missing_states = utils.native_numeric_missing_states(observation, self.variable_splits)

        from ._native import calculate
        self.result, self.new_observation = calculate(
            explainer,
            observation,
            self.variable_splits,
            y,
            self.processes,
            verbose
        )

    def plot(self,
             objects=None,
             variable_type="numerical",
             variables=None,
             size=2,
             alpha=1,
             color="_label_",
             facet_ncol=2,
             facet_scales="free_x",
             show_observations=True,
             title="Ceteris Paribus Profiles",
             y_title='prediction',
             horizontal_spacing=None,
             vertical_spacing=None,
             show=True):
        """Plot the Ceteris Paribus explanation

        Parameters
        -----------
        objects : CeterisParibus object or array_like of CeterisParibus objects
            Additional objects to plot in subplots (default is `None`).
        variable_type : {'numerical', 'categorical'}
            Plot the profiles for numerical or categorical variables 
            (default is `'numerical'`).
        variables : str or array_like of str, optional
            Variables for which the profiles will be calculated
            (default is `None`, which means all of the variables).
        size : float, optional
            Width of lines in px (default is `2`).
        alpha : float <0, 1>, optional
            Opacity of lines (default is `1`).
        color : str, optional
            Variable name used for grouping
            (default is `'_label_'`, which groups by models).
        facet_ncol : int, optional
            Number of columns on the plot grid (default is `2`).
        facet_scales : {'free_x', 'free'}, optional
            Set to `free` for free y-axis range (default is `'free_x'`).
        show_observations : bool, optional
            Show observation points (default is `True`).
        title : str, optional
            Title of the plot (default is `"Ceteris Paribus Profiles"`).
        y_title : str, optional
            Title of the y/x axis (default is `"prediction"`).
        horizontal_spacing : float <0, 1>, optional
            Ratio of horizontal space between the plots
            (default depends on `variable_type`).
        vertical_spacing : float <0, 1>, optional
            Ratio of vertical space between the plots (default is `0.3/number of rows`).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object that can 
            be edited or saved using the `write_image()` method (default is `True`).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """

        if variable_type not in ("numerical", "categorical"):
            raise TypeError("variable_type should be 'numerical' or 'categorical'")
        if isinstance(variables, str):
            variables = (variables,)

        from ._plotting import profile_frame, plot_frame, categorical_color
        if objects is None:
            profiles = [self]
            _include = self.variable_splits_with_obs
        elif isinstance(objects, self.__class__):
            profiles = [self, objects]
            _include = np.all([ob.variable_splits_with_obs for ob in profiles])
        elif isinstance(objects, (list, tuple)):
            profiles = [self, *objects]
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
            _include = np.all([ob.variable_splits_with_obs for ob in profiles])
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)
        _result_df, _originals, _converted, _absent = profile_frame(profiles)

        if _include is False and show_observations:
                warnings.warn("'show_observations' parameter changed to False,"
                              "because the 'variable_splits_with_obs' attribute is False"
                              "See `variable_splits_with_obs` parameter in `predict_profile`.")
                show_observations = False

        # variables to use
        all_variables = _result_df['_vname_'].drop_nulls().unique(maintain_order=True).to_list()

        if variables is not None:
            all_variables = _global_utils.intersect_unsorted(variables, all_variables)
            if len(all_variables) == 0:
                raise TypeError("variables do not overlap with " + ''.join(variables))

        # names of numeric variables
        numeric_variables = [name for name in all_variables if (
            _result_df.schema[name].is_numeric() and not isinstance(_result_df.schema[name], nw.Decimal)
        ) or isinstance(_result_df.schema[name], nw.Duration)]

        if variable_type == "numerical":
            variable_names = numeric_variables

            if len(variable_names) == 0:
                # change to categorical
                variable_type = "categorical"
                # send message
                warnings.warn("'variable_type' parameter changed to 'categorical' due to lack of numerical variables.")
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
        _result_df = plot_frame(_result_df, _originals, _converted, _absent, variable_names,
                               variable_type == 'categorical')

        #  calculate y axis range to allow for fixedrange True
        dl = _result_df['_yhat_'].to_numpy()
        min_max_margin = np.ptp(dl) * 0.10
        min_max = [dl.min() - min_max_margin, dl.max() + min_max_margin]

        n = len(variable_names)
        facet_nrow = int(np.ceil(n / facet_ncol))
        if vertical_spacing is None:
            vertical_spacing = 0.3 / facet_nrow if variable_type == 'numerical' else 0.05
        if horizontal_spacing is None:
            horizontal_spacing = 0.05 if variable_type == 'numerical' else 0.1

        plot_height = 78 + 71 + facet_nrow * (280 + 60)


        if variable_type == "numerical":    
            m = len(_result_df[color].drop_nulls().unique())
            _result_df = categorical_color(_result_df, color)
        
            fig = px.line(_result_df.to_native(),
                          x="_x_", y="_yhat_", color=color, facet_col="_vname_", line_group='_ids_',
                          category_orders={"_vname_": list(variable_names)},
                          labels={'_yhat_': 'prediction', '_label_': 'label', '_ids_': 'id'},  # , color: 'group'},
                          # hover_data={'_text_': True, '_yhat_': ':.3f', '_vname_': False, '_x_': False, color: False},
                          custom_data=['_text_'],
                          facet_col_wrap=facet_ncol,
                          facet_row_spacing=vertical_spacing,
                          facet_col_spacing=horizontal_spacing,
                          template="none",
                          color_discrete_sequence=_theme.get_default_colors(m, 'line')) \
                    .update_traces(dict(line_width=size, opacity=alpha,
                                        hovertemplate="%{customdata[0]}<extra></extra>")) \
                    .update_xaxes({'matches': None, 'showticklabels': True,
                                   'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True}) \
                    .update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True,
                                   'range': min_max})

            if show_observations:
                _points_df = _result_df.filter(nw.col('_is_observation_'))

                fig_points = px.scatter(_points_df.to_native(),
                                        x='_x_', y='_yhat_', facet_col='_vname_',
                                        category_orders={"_vname_": list(variable_names)},
                                        labels={'_yhat_': 'prediction', '_label_': 'label', '_ids_': 'id'},
                                        custom_data=['_text_'],
                                        facet_col_wrap=facet_ncol,
                                        facet_row_spacing=vertical_spacing,
                                        facet_col_spacing=horizontal_spacing,
                                        color_discrete_sequence=["#371ea3"]) \
                               .update_traces(dict(marker_size=5*size, opacity=alpha),
                                              hovertemplate="%{customdata[0]}<extra></extra>")

                for _, value in enumerate(fig_points.data):
                    fig.add_trace(value)
                    
            fig = _theme.fig_update_line_plot(fig, title, y_title, plot_height, 'closest')

        else:
            if color=="_label_" and len(_result_df['_ids_'].unique()) > 1 and len(_result_df['_label_'].unique()) == 1:
                warnings.warn("'color' parameter changed to '_ids_' because there are multiple observations for one model.")
                color = '_ids_'
            elif color=="_label_" and len(_result_df['_ids_'].unique()) > len(_result_df['_label_'].unique()): 
                # https://github.com/plotly/plotly.py/issues/2657
                raise TypeError("Please pick one observation per label or change the `color` parameter.")

            m = len(_result_df[color].drop_nulls().unique())
            _result_df = categorical_color(_result_df, color)
            
            _result_df = _result_df.with_columns((nw.col('_yhat_') - nw.col('_original_yhat_')).alias('_diff_'))
            fig = px.bar(_result_df.to_native(),
                         x="_diff_", y="_x_", color=color, facet_col="_vname_",
                         category_orders={"_vname_": list(variable_names)},
                         labels={'_yhat_': 'prediction', '_label_': 'label', '_ids_': 'id'},  # , color: 'group'},
                         # hover_data={'_yhat_': ':.3f', '_ids_': True, '_vname_': False, color: False},
                         custom_data=['_text_'],
                         base="_original_yhat_",
                         facet_col_wrap=facet_ncol,
                         facet_row_spacing=vertical_spacing,
                         facet_col_spacing=horizontal_spacing,
                         template="none",
                         color_discrete_sequence=_theme.get_default_colors(m, 'line'),
                         barmode='group',
                         orientation='h')  \
                    .update_traces(dict(opacity=alpha),
                                   hovertemplate="%{customdata[0]}<extra></extra>") \
                    .update_yaxes({'matches': None, 'showticklabels': True,
                                   'type': 'category', 'gridwidth': 2, 'automargin': True, 
                                   'ticks': "outside", 'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True}) \
                    .update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True,
                                   'range': min_max})
                    
            for _, bar in enumerate(fig.data):
                fig.add_vline(x=bar.base[0], layer='below',
                              line={'color': "#371ea3", 'width': 1.5, 'dash': 'dot'})
                
            if show_observations:
                _points_df = _result_df.filter(nw.col('_is_observation_'))

                fig_points = px.scatter(_points_df.to_native(),
                                        x='_yhat_', y='_x_', facet_col='_vname_',
                                        category_orders={"_vname_": list(variable_names)},
                                        labels={'_yhat_': 'prediction', '_label_': 'label', '_ids_': 'id'},
                                        custom_data=['_text_'],
                                        facet_col_wrap=facet_ncol,
                                        facet_row_spacing=vertical_spacing,
                                        facet_col_spacing=horizontal_spacing,
                                        color_discrete_sequence=["#371ea3"]) \
                               .update_traces(dict(marker_size=5*size, opacity=alpha),
                                              hovertemplate="%{customdata[0]}<extra></extra>")

                for _, value in enumerate(fig_points.data):
                    fig.add_trace(value)

            fig = _theme.fig_update_bar_plot(fig, title, y_title, plot_height, 'closest')
            
        fig.update_layout(hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)'))

        if facet_scales == "free":
            fig.update_yaxes(matches=None, range=None, showticklabels=True)

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
