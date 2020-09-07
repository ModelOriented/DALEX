import plotly.express as px

from .checks import *
from ..._explainer.theme import get_default_colors
from ..._explainer.utils import check_import


class ResidualDiagnostics:
    """Calculate dataset level residuals diagnostics

    Parameters
    -----------
    variables : str or array_like of str, optional
        Variables for which the profiles will be calculated
        (default is None, which means all of the variables).

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    variables : array_like of str or None
        Variables for which the profiles will be calculated

    Notes
    --------
    https://pbiecek.github.io/ema/residualDiagnostic.html
    """
    def __init__(self,
                 variables=None):

        variables_ = check_variables(variables)

        self.result = None
        self.variables = variables_

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
            result = result.loc[:, np.intersect1d(self.variables, result.columns)]
        # is there target
        if explainer.y is not None:
            result = result.assign(y=explainer.y)
        # are there predictions - add y_hat to the Explainer for the future
        if explainer.y_hat is None:
            explainer.y_hat = explainer.predict(explainer.data)
        # are there residuals - add residuals to the Explainer for the future
        if explainer.residuals is None:
            explainer.residuals = explainer.residual(explainer.data, explainer.y)

        self.result  = result.assign(
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
             show=True):
        """Plot the Residual Diagnostics explanation

        Parameters
        ----------
        objects : ResidualDiagnostics object or array_like of ResidualDiagnostics objects
            Additional objects to plot (default is None).
        variable : str, optional
            Name of the variable from the `result` attribute to appear on the OX axis
            (default is 'y_hat').
        yvariable : str, optional
            Name of the variable from the `result` attribute to appear on the OY axis
            (default is 'residuals').
        smooth : bool, optional
            Add the smooth line (default is True).
        line_width : float, optional
            Width of lines in px (default is 2).
        marker_size : float, optional
            Size of points (default is 3).
        title : str, optional
            Title of the plot (default depends on the `type` attribute).
        show : bool, optional
            True shows the plot; False returns the plotly Figure object that can be
            edited or saved using the `write_image()` method (default is True).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """

        check_import('statsmodels', msg='Install statsmodels>=0.11.1 for smoothing line.')

        # are there any other objects to plot?
        if objects is None:
            _df_list = [self.result.copy()]
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            _df_list = [self.result.copy(), objects.result.copy()]
        else:  # objects as tuple or array
            _df_list = [self.result.copy()]
            for ob in objects:
                if not isinstance(ob, self.__class__):
                    raise TypeError("Some explanations aren't of ResidualDiagnostics class: " +
                                    type(ob))
                _df_list += [ob.result.copy()]

        fig = px.scatter(pd.concat(_df_list),
                         x=variable,
                         y=yvariable,
                         color="label",
                         trendline="lowess" if smooth else None,
                         color_discrete_sequence=get_default_colors(len(_df_list), 'line')) \
               .update_traces(dict(marker_size=marker_size, line_width=line_width))

        # wait for https://github.com/plotly/plotly.py/pull/2558 to add hline to the plot

        fig.update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': 'outside',
                          'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': yvariable})

        fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                          'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': variable})

        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          margin={'t': 78, 'b': 71, 'r': 30})

        if show:
            fig.show(config={'displaylogo': False, 'staticPlot': False,
                             'toImageButtonOptions': {'height': None, 'width': None, },
                             'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d',
                                                        'zoom2d',
                                                        'pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d',
                                                        'toggleSpikelines', 'hoverCompareCartesian',
                                                        'hoverClosestCartesian']})
        else:
            return fig
