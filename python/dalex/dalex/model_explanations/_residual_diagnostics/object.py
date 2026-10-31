import numpy as np
import plotly.express as px
import narwhals.stable.v2 as nw
from narwhals.dependencies import is_pandas_dataframe

from . import checks
from ... import _theme, _global_checks, _global_utils
from ..._explanation import Explanation, EmptyTable, IndexedTable, PandasFeatureTable
from ..._dataframe import as_eager_dataframe, copy_for_edit
from ..._plotting import result_frame, concat_frames


class ResidualDiagnostics(Explanation):
    """Calculate model-level residuals diagnostics

    Parameters
    -----------
    variables : str or array_like of str, optional
        Variables for which the profiles will be calculated
        (default is `None`, which means all of the variables).

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    variables : array_like of str or None
        Variables for which the profiles will be calculated.

    Notes
    --------
    - https://pbiecek.github.io/ema/residualDiagnostic.html
    """
    def __init__(self,
                 variables=None):

        _variables = checks.check_variables(variables)

        self.result = EmptyTable()
        self.variables = _variables

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
        source = copy_for_edit(explainer.data)
        legacy_features = None
        if is_pandas_dataframe(source) and not source.columns.is_unique:
            # Keep duplicate labels at the input/publication boundary while
            # computing the diagnostic columns through the shared native path.
            if self.variables is not None:
                selected = _global_utils.intersect_unsorted(self.variables, source.columns)
                source = source.loc[:, selected]
            legacy_features = source
            result = nw.from_dict({'__dalex_row__': np.arange(len(source))}, backend='pandas')
        else:
            result = as_eager_dataframe(source)
        namespace = nw.get_native_namespace(result)
        index = source.index.copy() if is_pandas_dataframe(source) else None

        # if variables = NULL then all variables are added
        # otherwise only selected
        if self.variables is not None and legacy_features is None:
            selected = _global_utils.intersect_unsorted(self.variables, result.columns)
            result = result.select(selected)
        # is there target
        if explainer.y is not None:
            result = result.with_columns(nw.new_series('y', np.array(explainer.y, copy=True), backend=namespace))
        # are there predictions - add y_hat to the Explainer for the future
        if explainer.y_hat is None:
            explainer.y_hat = explainer.predict(explainer.data)
        # are there residuals - add residuals to the Explainer for the future
        if explainer.residuals is None:
            explainer.residuals = explainer.residual(explainer.data, explainer.y)

        result = result.with_columns(
            nw.new_series('y_hat', np.array(explainer.y_hat, copy=True), backend=namespace),
            nw.new_series('residuals', np.array(explainer.residuals, copy=True), backend=namespace),
            nw.new_series('abs_residuals', np.abs(explainer.residuals), backend=namespace),
            nw.lit(explainer.label).alias('label'),
            nw.new_series('ids', np.arange(len(result)) + 1, backend=namespace),
        )
        if legacy_features is not None:
            self.result = PandasFeatureTable(legacy_features, result.drop('__dalex_row__').to_native(),
                                            scalar_columns={'label': explainer.label})
        else:
            self.result = IndexedTable(result.to_native(), index)

    def plot(self,
             objects=None,
             variable="y_hat",
             yvariable="residuals",
             smooth=True,
             line_width=2,
             marker_size=3,
             title="Residual Diagnostics",
             N=50000,
             show=True):
        """Plot the Residual Diagnostics explanation

        Parameters
        ----------
        objects : ResidualDiagnostics object or array_like of ResidualDiagnostics objects
            Additional objects to plot (default is `None`).
        variable : str, optional
            Name of the variable from the `result` attribute to appear on the OX axis
            (default is `'y_hat'`).
        yvariable : str, optional
            Name of the variable from the `result` attribute to appear on the OY axis
            (default is `'residuals'`).
        smooth : bool, optional
            Add the smooth line (default is `True`).
        line_width : float, optional
            Width of lines in px (default is `2`).
        marker_size : float, optional
            Size of points (default is `3`).
        title : str, optional
            Title of the plot (default depends on the `type` attribute).
        N : int, optional
            Number of observations that will be sampled from the `result` attribute before
            calculating the smooth line. This is for performance issues with large data.
            `None` means take all `result` (default is `50_000`).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object that can 
            be edited or saved using the `write_image()` method (default is `True`).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """

        if smooth:
            _global_checks.global_check_import('statsmodels', 'smoothing line')

        # are there any other objects to plot?
        if objects is None:
            _df_list = [result_frame(self)]
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            _df_list = [result_frame(self), result_frame(objects)]
        elif isinstance(objects, (list, tuple)):  # objects as tuple or array
            _df_list = [result_frame(self)]
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _df_list += [result_frame(ob)]
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)

        columns = list(dict.fromkeys([variable, yvariable, 'ids', 'label']))
        # Do not concatenate unused original features (which may have different
        # schemas across models) just to plot a few diagnostic columns.
        _df = concat_frames((frame.select([name for name in columns if name in frame.columns])
                             for frame in _df_list), how='diagonal').select(columns)

        if isinstance(N, int) and smooth:
            if N < _df.shape[0]:
                # Use pandas' historical positional sampling, independent of backend RNG.
                positions = np.random.RandomState(0).choice(len(_df), size=N, replace=False)
                _df = _df[positions.tolist(), :]

        fig = px.scatter(_df.to_native(),
                         x=variable,
                         y=yvariable,
                         hover_name='ids',
                         color="label",
                         trendline="lowess" if smooth else None,
                         color_discrete_sequence=_theme.get_default_colors(len(_df_list), 'line')) \
                .update_traces(dict(marker_size=marker_size, line_width=line_width))

        # wait for https://github.com/plotly/plotly.py/pull/2558 to add hline to the plot

        fig.update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': 'outside',
                          'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': yvariable})

        fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                          'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': variable})

        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          margin={'t': 78, 'b': 71, 'r': 30})

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
