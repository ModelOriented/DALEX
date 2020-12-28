import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from . import checks, plot, utils
from ... import _theme, _global_checks
from ..._explanation import Explanation


class VariableImportance(Explanation):
    """Calculate model-level variable importance

    Parameters
    -----------
    loss_function : {'rmse', '1-auc', 'mse', 'mae', 'mad'} or function, optional
        If string, then such loss function will be used to assess variable importance
        (default is `'rmse'` or `'1-auc', depends on `model_type` attribute).
    type : {'variable_importance', 'ratio', 'difference'}, optional
        Type of transformation that will be applied to dropout loss.
    N : int, optional
        Number of observations that will be sampled from the `data` attribute before
        the calculation of variable importance. 
        `None` means all `data` (default is `1000`).
    B : int, optional
        Number of permutation rounds to perform on each variable (default is `10`).
    variables : array_like of str, optional
        Variables for which the importance will be calculated
        (default is `None`, which means all of the variables).
        NOTE: Ignored if `variable_groups` is not None.
    variable_groups : dict of lists, optional
        Group the variables to calculate their joint variable importance
        e.g. `{'X': ['x1', 'x2'], 'Y': ['y1', 'y2']}` (default is `None`).
    keep_raw_permutations: bool, optional
        Save results for all permutation rounds (default is `True`).
    processes : int, optional
        Number of parallel processes to use in calculations. Iterated over `B`
        (default is `1`, which means no parallel computation).
    random_state : int, optional
        Set seed for random number generator (default is random seed).

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    loss_function : function
        Loss function used to assess the variable importance.
    type : {'variable_importance', 'ratio', 'difference'}
        Type of transformation that will be applied to dropout loss.
    N : int
        Number of observations that will be sampled from the `data` attribute 
        before the calculation of variable importance.
    B : int
        Number of permutation rounds to perform on each variable.
    variables : array_like of str or None
        Variables for which the importance will be calculated
    variable_groups : dict of lists or None
        Grouped variables to calculate their joint variable importance.
    keep_raw_permutations: bool
        Save the results for all permutation rounds.
    permutation : pd.DataFrame or None
        The results for all permutation rounds.
    processes : int
        Number of parallel processes to use in calculations. Iterated over `B`.
    random_state : int or None
        Set seed for random number generator.

    Notes
    --------
    - https://pbiecek.github.io/ema/featureImportance.html
    """

    def __init__(self,
                 loss_function='rmse',
                 type='variable_importance',
                 N=1000,
                 B=10,
                 variables=None,
                 variable_groups=None,
                 keep_raw_permutations=True,
                 processes=1,
                 random_state=None):

        _loss_function = checks.check_loss_function(loss_function)
        _B = checks.check_B(B)
        _type = checks.check_type(type)
        _random_state = checks.check_random_state(random_state)
        _keep_raw_permutations = checks.check_keep_raw_permutations(keep_raw_permutations, B)
        _processes = checks.check_processes(processes)

        self.loss_function = _loss_function
        self.type = _type
        self.N = N
        self.B = _B
        self.variables = variables
        self.variable_groups = variable_groups
        self.random_state = _random_state
        self.keep_raw_permutations = _keep_raw_permutations
        self.result = None
        self.permutation = None
        self.processes = _processes

    def _repr_html_(self):
        return self.result._repr_html_()

    def fit(self, explainer):
        """Calculate the result of explanation

        Fit method makes calculations in place and changes the attributes.

        Parameters
        -----------
        explainer : Explainer object
            Model wrapper created using the Explainer class.

        Returns
        -----------
        None
        """

        # if `variable_groups` are not specified, then extract from `variables`
        self.variable_groups = checks.check_variable_groups(self.variable_groups, explainer)
        self.variables = checks.check_variables(self.variables, self.variable_groups, explainer)
        self.result, self.permutation = utils.calculate_variable_importance(explainer,
                                                                            self.type,
                                                                            self.loss_function,
                                                                            self.variables,
                                                                            self.N,
                                                                            self.B,
                                                                            explainer.label,
                                                                            self.processes,
                                                                            self.keep_raw_permutations)

    def plot(self,
             objects=None,
             max_vars=10,
             digits=3,
             rounding_function=np.around,
             bar_width=16,
             split=("model", "variable"),
             title="Variable Importance",
             vertical_spacing=None,
             show=True):
        """Plot the Variable Importance explanation

        Parameters
        -----------
        objects : VariableImportance object or array_like of VariableImportance objects
            Additional objects to plot in subplots (default is `None`).
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
        split : {'model', 'variable'}, optional
            Split the subplots by model or variable (default is `'model'`).
        title : str, optional
            Title of the plot (default is `"Variable Importance"`).
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

        if isinstance(split, tuple):
            split = split[0]

        if split not in ("model", "variable"):
            raise TypeError("split should be 'model' or 'variable'")

        # are there any other objects to plot?
        if objects is None:
            n = 1
            _result_df = self.result.copy()
            if split == 'variable':  # force split by model if only one explainer
                split = 'model'
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            n = 2
            _result_df = pd.concat([self.result.copy(), objects.result.copy()])
        elif isinstance(objects, (list, tuple)):  # objects as tuple or array
            n = len(objects) + 1
            _result_df = self.result.copy()
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _result_df = pd.concat([_result_df, ob.result.copy()])
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)

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
        perm = _result_df[['variable', 'dropout_loss']].groupby('variable').mean().reset_index(). \
            sort_values('dropout_loss', ascending=False).variable.values

        plot_height = 78 + 71

        colors = _theme.get_default_colors(n, 'bar')

        if vertical_spacing is None:
            vertical_spacing = 0.2 / n

        model_names = _result_df['label'].unique().tolist()

        if len(model_names) != n:
            raise ValueError('label must be unique for each model')

        if split == "model":
            # init plot
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=vertical_spacing,
                                x_title='drop-out loss',
                                subplot_titles=model_names)

            # split df by model
            df_list = [v for k, v in _result_df.groupby('label', sort=False)]

            for i, df in enumerate(df_list):
                m = df.shape[0]
                if max_vars is not None and max_vars < m:
                    m = max_vars

                # take only m variables (for max_vars)
                # sort rows of df by variable permutation and drop unused variables
                df = df.sort_values('dropout_loss').tail(m) \
                    .set_index('variable').reindex(perm).dropna().reset_index()

                baseline = df.iloc[0, df.columns.get_loc('full_model')]

                df = df.assign(difference=lambda x: x['dropout_loss'] - baseline)

                lt = df.difference.apply(lambda val:
                                         "+"+str(rounding_function(np.abs(val), digits)) if val > 0
                                         else str(rounding_function(np.abs(val), digits)))
                tt = df.apply(lambda row: plot.tooltip_text(row, rounding_function, digits), axis=1)
                df = df.assign(label_text=lt,
                               tooltip_text=tt)

                fig.add_shape(type='line', x0=baseline, x1=baseline, y0=-1, y1=m, yref="paper", xref="x",
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

                fig.update_xaxes(
                    {'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                     'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                    row=i + 1, col=1)

                plot_height += m * bar_width + (m + 1) * bar_width / 4 + 30
        else:
            # split df by variable
            df_list = [v for k, v in _result_df.groupby('variable', sort=False)]

            n = len(df_list)
            if max_vars is not None and max_vars < n:
                n = max_vars
            
            if vertical_spacing is None:
                vertical_spacing = 0.2 / n
            
            # init plot
            variable_names = perm[0:n]
            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=vertical_spacing, x_title='drop-out loss',
                                subplot_titles=variable_names)

            df_dict = {e.variable.array[0]: e for e in df_list}

            # take only n=max_vars elements from df_dict
            for i in range(n):
                df = df_dict[perm[i]]
                m = df.shape[0]

                baseline = 0

                df = df.assign(difference=lambda x: x['dropout_loss'] - x['full_model'])

                lt = df.difference.apply(lambda val:
                                         "+"+str(rounding_function(np.abs(val), digits)) if val > 0
                                         else str(rounding_function(np.abs(val), digits)))
                tt = df.apply(lambda row: plot.tooltip_text(row, rounding_function, digits), axis=1)
                df = df.assign(label_text=lt,
                               tooltip_text=tt)

                fig.add_shape(type='line', x0=baseline, x1=baseline, y0=-1, y1=m, yref="paper", xref="x",
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

                fig.update_xaxes(
                    {'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                     'tickcolor': 'white', 'ticklen': 3, 'fixedrange': True},
                    row=i + 1, col=1)

                plot_height += m * bar_width + (m + 1) * bar_width / 4

        plot_height += (n - 1) * 70

        fig.update_xaxes({'range': min_max})
        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          height=plot_height, margin={'t': 78, 'b': 71, 'r': 30})

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
