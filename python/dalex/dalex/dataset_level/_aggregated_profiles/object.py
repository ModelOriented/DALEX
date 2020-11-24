import plotly.express as px
from copy import deepcopy

from .checks import *
from .utils import aggregate_profiles
from ... import _theme, _global_checks, _global_utils


class AggregatedProfiles:
    """Calculate dataset level variable profiles as Partial or Accumulated Dependence

    Partial Dependence Profile (average across CP Profiles),
    Individual Conditional Expectation (local weighted average across CP Profiles),
    Accumulated Local Effects (cummulated average local changes in CP Profiles).

    Parameters
    -----------
    type : {'partial', 'accumulated', 'conditional'}
        Type of model profiles (default is 'partial' for Partial Dependence Profiles).
    variables : str or array_like of str, optional
        Variables for which the profiles will be calculated
        (default is None, which means all of the variables).
    variable_type : {'numerical', 'categorical'}
        Calculate the profiles for numerical or categorical variables
        (default is 'numerical').
    groups : str or array_like of str, optional
        Names of categorical variables that will be used for profile grouping
        (default is None, which means no grouping).
    span : float, optional
        Smoothing coefficient used as sd for gaussian kernel (default is 0.25).
    center : bool, optional
        Theoretically Accumulated Profiles start at 0, but are centered to compare
        them with Partial Dependence Profiles (default is True, which means center
        around the average y_hat calculated on the data sample).
    random_state : int, optional
        Set seed for random number generator (default is random seed).

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    mean_prediction : float
        Average prediction for sampled `data` (using `N`).
    raw_profiles : pd.DataFrame or None
        Saved CeterisParibus object.
        NOTE: None if more objects were passed to the `fit` method.
    type : {'partial', 'accumulated', 'conditional'}
        Type of model profiles.
    variables : array_like of str or None
        Variables for which the profiles will be calculated
    variable_type : {'numerical', 'categorical'}
        Calculate the profiles for numerical or categorical variables.
    groups : str or array_like of str or None
        Names of categorical variables that will be used for profile grouping.
    span : float
        Smoothing coefficient used as sd for gaussian kernel.
    center : bool
        Theoretically Accumulated Profiles start at 0, but are centered to compare
        them with Partial Dependence Profiles (default is True, which means center
        around the average y_hat calculated on the data sample).
    random_state : int or None
        Set seed for random number generator.

    Notes
    --------
    https://pbiecek.github.io/ema/partialDependenceProfiles.html
    https://pbiecek.github.io/ema/accumulatedLocalProfiles.html
    """

    def __init__(self,
                 type='partial',
                 variables=None,
                 variable_type='numerical',
                 groups=None,
                 span=0.25,
                 center=True,
                 random_state=None):

        check_variable_type(variable_type)
        variables_ = check_variables(variables)
        groups_ = check_groups(groups)

        self.variable_type = variable_type
        self.groups = groups_
        self.type = type
        self.variables = variables_
        self.span = span
        self.center = center
        self.result = None
        self.mean_prediction = None
        self.raw_profiles = None
        self.random_state = random_state

    def _repr_html_(self):
        return self.result._repr_html_()

    def fit(self,
            ceteris_paribus,
            verbose=True):
        """Calculate the result of explanation

        Fit method makes calculations in place and changes the attributes.

        Parameters
        -----------
        ceteris_paribus : CeterisParibus object or array_like of CeterisParibus objects
            Profile objects to aggregate.
        verbose : bool, optional
            Print tqdm progress bar (default is True).

        Returns
        -----------
        None
        """
        # are there any other cp?
        from dalex.instance_level import CeterisParibus
        if isinstance(ceteris_paribus, CeterisParibus):  # allow for ceteris_paribus to be a single element
            all_profiles = ceteris_paribus.result.copy()
            all_observations = ceteris_paribus.new_observation.copy()
            self.raw_profiles = deepcopy(ceteris_paribus)
        elif isinstance(ceteris_paribus, (list, tuple)):  # ceteris_paribus as tuple or array
            all_profiles = None
            all_observations = None
            for cp in ceteris_paribus:
                _global_checks.global_check_object_class(cp, CeterisParibus)
                all_profiles = pd.concat([all_profiles, cp.result.copy()])
                all_observations = pd.concat([all_observations, cp.new_observation.copy()])
        else:
            _global_checks.global_raise_objects_class(ceteris_paribus, CeterisParibus)

        all_profiles, vnames = prepare_numerical_categorical(all_profiles, self.variables, self.variable_type)

        # select only suitable variables
        all_profiles = all_profiles.loc[all_profiles['_vname_'].isin(vnames), :]

        all_profiles = create_x(all_profiles, self.variable_type)

        self.mean_prediction = all_observations['_yhat_'].mean()

        self.result = aggregate_profiles(all_profiles, self.mean_prediction, self.type, self.groups, self.center,
                                         self.span, verbose)

    def plot(self,
             objects=None,
             geom='aggregates',
             variables=None,
             center=True,
             size=2,
             alpha=1,
             facet_ncol=2,
             title="Aggregated Profiles",
             y_title='prediction',
             horizontal_spacing=0.05,
             vertical_spacing=None,
             show=True):
        """Plot the Aggregated Profiles explanation

        Parameters
        -----------
        objects : AggregatedProfiles object or array_like of AggregatedProfiles objects
            Additional objects to plot in subplots (default is None).
        geom : {'aggregates', 'profiles'}
            If 'profiles' then raw profiles will be plotted in the background
            (default is 'aggregates', which means plot only aggregated profiles).
            NOTE: It is useful to use small values of the `N` parameter in object creation
            before using `profiles`, because of plot performance and clarity (e.g. 100).
        variables : str or array_like of str, optional
            Variables for which the profiles will be calculated
            (default is None, which means all of the variables).
        center : bool, optional
            Theoretically Accumulated Profiles start at 0, but are centered to compare
            them with Partial Dependence Profiles (default is True, which means center
            around the average y_hat calculated on the data sample).
        size : float, optional
            Width of lines in px (default is 2).
        alpha : float <0, 1>, optional
            Opacity of lines (default is 1).
        color : str, optional
            Variable name used for grouping (default is '_label_', which groups by models).
        facet_ncol : int, optional
            Number of columns on the plot grid (default is 2).
        title : str, optional
            Title of the plot (default is "Aggregated Profiles").
        y_title : str, optional
            Title of the x axis (default is "prediction").
        horizontal_spacing : float <0, 1>, optional
            Ratio of horizontal space between the plots (default is 0.05).
        vertical_spacing : float <0, 1>, optional
            Ratio of vertical space between the plots (default is 0.3/number of rows).
        show : bool, optional
            True shows the plot; False returns the plotly Figure object that can be
            edited or saved using the `write_image()` method (default is True).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """
        # TODO: numerical+categorical in one plot https://github.com/plotly/plotly.py/issues/2647

        if geom not in ("aggregates", "profiles"):
            raise TypeError("geom should be 'aggregates' or 'profiles'")
        if isinstance(variables, str):
            variables = (variables,)

        # are there any other objects to plot?
        if objects is None:
            _result_df = self.result.assign(_mp_=self.mean_prediction if center else 0)
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            _result_df = pd.concat([self.result.assign(_mp_=self.mean_prediction if center else 0),
                                    objects.result.assign(_mp_=objects.mean_prediction if center else 0)])
        elif isinstance(objects, (list, tuple)):  # objects as tuple or array
            _result_df = self.result.assign(_mp_=self.mean_prediction if center else 0)
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _result_df = pd.concat([_result_df, ob.result.assign(_mp_=ob.mean_prediction if center else 0)])
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)

        # variables to use
        all_variables = _result_df['_vname_'].dropna().unique().tolist()

        if variables is not None:
            all_variables = _global_utils.intersect_unsorted(variables, all_variables)
            if len(all_variables) == 0:
                raise TypeError("variables do not overlap with " + ''.join(variables))

            _result_df = _result_df.loc[_result_df['_vname_'].isin(all_variables), :]

        #  calculate y axis range to allow for fixedrange True
        dl = _result_df['_yhat_'].to_numpy()
        min_max_margin = dl.ptp() * 0.10
        min_max = [dl.min() - min_max_margin, dl.max() + min_max_margin]

        is_x_numeric = pd.api.types.is_numeric_dtype(_result_df['_x_'])
        n = len(all_variables)

        facet_nrow = int(np.ceil(n / facet_ncol))
        if vertical_spacing is None:
            vertical_spacing = 0.3 / facet_nrow
        plot_height = 78 + 71 + facet_nrow * (280 + 60)
        hovermode, render_mode = 'x unified', 'svg'

        color = '_label_'  # _groups_ doesnt make much sense for multiple AP objects
        m = len(_result_df[color].dropna().unique())

        if is_x_numeric:
            if geom == 'profiles' and self.raw_profiles is not None:
                render_mode = 'webgl'

            fig = px.line(_result_df,
                          x="_x_", y="_yhat_", color=color, facet_col="_vname_",
                          category_orders={"_vname_": list(all_variables)},
                          labels={'_yhat_': 'prediction', '_mp_': 'mean_prediction'},  # , color: 'group'},
                          hover_name=color,
                          hover_data={'_yhat_': ':.3f', '_mp_': ':.3f',
                                      color: False, '_vname_': False, '_x_': False},
                          facet_col_wrap=facet_ncol,
                          facet_row_spacing=vertical_spacing,
                          facet_col_spacing=horizontal_spacing,
                          template="none",
                          render_mode=render_mode,
                          color_discrete_sequence=_theme.get_default_colors(m, 'line')) \
                    .update_traces(dict(line_width=size, opacity=alpha)) \
                    .update_xaxes({'matches': None, 'showticklabels': True,
                                   'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': "outside", 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True}) \
                    .update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True,
                                   'range': min_max})

            if geom == 'profiles' and self.raw_profiles is not None:
                fig.update_traces(dict(line_width=2*size, opacity=1))
                fig_cp = self.raw_profiles.plot(variables=list(all_variables),
                                                facet_ncol=facet_ncol,
                                                show_observations=False, show=False) \
                    .update_traces(dict(line_width=1, opacity=0.5, line_color='#ceced9'))

                for _, value in enumerate(fig.data):
                    fig_cp.add_trace(value)
                hovermode = False
                fig = fig_cp
        else:
            _result_df = _result_df.assign(_diff_=lambda x: x['_yhat_'] - x['_mp_'])
            mp_format = ':.3f'
            if not center:
                min_max = [np.min([min_max[0], 0]), np.max([min_max[1], 0])]
                mp_format = False

            fig = px.bar(_result_df,
                         x="_x_", y="_diff_", color="_label_", facet_col="_vname_",
                         category_orders={"_vname_": list(all_variables)},
                         labels={'_yhat_': 'prediction', '_mp_': 'mean_prediction'},  # , color: 'group'},
                         hover_name=color,
                         base="_mp_",
                         hover_data={'_yhat_': ':.3f', '_mp_': mp_format, '_diff_': False,
                                     color: False, '_vname_': False, '_x_': False},
                         facet_col_wrap=facet_ncol,
                         facet_row_spacing=vertical_spacing,
                         facet_col_spacing=horizontal_spacing,
                         template="none",
                         color_discrete_sequence=_theme.get_default_colors(m, 'line'),  # bar was forgotten
                         barmode='group')  \
                    .update_xaxes({'matches': None, 'showticklabels': True,
                                   'type': 'category', 'gridwidth': 2, 'automargin': True,  # autorange="reversed"
                                   'ticks': "outside", 'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True}) \
                    .update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True,
                                   'ticks': 'outside', 'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True,
                                   'range': min_max})

            # add hline https://github.com/plotly/plotly.py/issues/2141
            for i, bar in enumerate(fig.data):
                fig.add_shape(type='line', y0=bar.base[0], y1=bar.base[0], x0=-1, x1=len(bar.x),
                              xref=bar.xaxis, yref=bar.yaxis, layer='below',
                              line={'color': "#371ea3", 'width': 1.5, 'dash': 'dot'})

        fig = _theme.fig_update_line_plot(fig, title, y_title, plot_height, hovermode)

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
