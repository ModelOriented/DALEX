import numpy as np
import pandas as pd
import plotly.express as px

from . import checks
from ... import _theme, _global_checks, _global_utils
from ..._explanation import Explanation


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

        self.result = None
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
        result = explainer.data.copy()

        # if variables = NULL then all variables are added
        # otherwise only selected
        if self.variables is not None:
            result = result.loc[:, _global_utils.intersect_unsorted(self.variables, result.columns)]
        # is there target
        if explainer.y is not None:
            result = result.assign(y=explainer.y)
        # are there predictions - add y_hat to the Explainer for the future
        if explainer.y_hat is None:
            explainer.y_hat = explainer.predict(explainer.data)
        # are there residuals - add residuals to the Explainer for the future
        if explainer.residuals is None:
            explainer.residuals = explainer.residual(explainer.data, explainer.y)

        self.result = result.assign(
            y_hat=explainer.y_hat,
            residuals=explainer.residuals,
            abs_residuals=np.abs(explainer.residuals),
            label=explainer.label,
            ids=np.arange(result.shape[0])+1
        )

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

        _global_checks.global_check_import('statsmodels', 'smoothing line')

        # are there any other objects to plot?
        if objects is None:
            _df_list = [self.result.copy()]
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            _df_list = [self.result.copy(), objects.result.copy()]
        elif isinstance(objects, (list, tuple)):  # objects as tuple or array
            _df_list = [self.result.copy()]
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _df_list += [ob.result.copy()]
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)

        _df = pd.concat(_df_list)

        if isinstance(N, int) and smooth:
            if N < _df.shape[0]:
                _df = _df.sample(N, random_state=0, replace=False)

        fig = px.scatter(_df,
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
