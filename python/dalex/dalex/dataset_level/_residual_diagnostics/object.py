import plotly.graph_objects as go

from .checks import *


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

        # are there predictions
        #:# add y_hat to the Explainer for the future
        if explainer.y_hat is None:
            explainer.y_hat = explainer.predict(explainer.data)

        result = result.assign(y_hat=explainer.y_hat)

        # are there residuals
        #:# add residuals to the Explainer for the future
        if explainer.residuals is None:
            explainer.residuals = explainer.residual(explainer.data, explainer.y)

        result = result.assign(
            y_hat=explainer.y_hat,
            residuals=explainer.residuals,
            abs_residuals=np.abs(explainer.residuals),
            label=explainer.label,
            ids=np.arange(result.shape[0])+1
        )
        self.result = result

    def plot(self, objects, variable="y_hat", yvariable="residuals", smooth=True):
        pass