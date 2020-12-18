import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import plot, utils
from ... import _theme, _global_checks


class ModelPerformance:
    """Calculate dataset level model performance measures

    Parameters
    -----------
    model_type : {'regression', 'classification'}
        Model task type that is used to choose the proper performance measures.
    cutoff : float, optional
        Cutoff for predictions in classification models. Needed for measures like
        recall, precision, acc, f1 (default is 0.5).

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    residuals : pd.DataFrame
        Residuals for `data`.
    model_type : {'regression', 'classification'}
        Model task type that is used to choose the proper performance measures.
    cutoff : float
        Cutoff for predictions in classification models.

    Notes
    --------
    - https://pbiecek.github.io/ema/modelPerformance.html
    """
    def __init__(self,
                 model_type,
                 cutoff=0.5):

        self.cutoff = cutoff
        self.model_type = model_type
        self.result = None
        self.residuals = None

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

        if explainer.y_hat is not None:
            y_pred = explainer.y_hat
        else:
            y_pred = explainer.predict(explainer.data)

        if explainer.residuals is not None:
            _residuals = explainer.residuals
        else:
            _residuals = explainer.residual(explainer.data, explainer.y)

        y_true = explainer.y

        if self.model_type == 'regression':
            _mse = utils.mse(y_pred, y_true)
            _rmse = utils.rmse(y_pred, y_true)
            _r2 = utils.r2(y_pred, y_true)
            _mae = utils.mae(y_pred, y_true)
            _mad = utils.mad(y_pred, y_true)

            self.result = pd.DataFrame(
                {
                    'mse': [_mse],
                    'rmse': [_rmse],
                    'r2': [_r2],
                    'mae': [_mae],
                    'mad': [_mad]
                }, index=[explainer.label])
        elif self.model_type == 'classification':
            tp = ((y_true == 1) * (y_pred >= self.cutoff)).sum()
            fp = ((y_true == 0) * (y_pred >= self.cutoff)).sum()
            tn = ((y_true == 0) * (y_pred < self.cutoff)).sum()
            fn = ((y_true == 1) * (y_pred < self.cutoff)).sum()

            _recall = utils.recall(tp, fp, tn, fn)
            _precision = utils.precision(tp, fp, tn, fn)
            _f1 = utils.f1(tp, fp, tn, fn)
            _accuracy = utils.accuracy(tp, fp, tn, fn)
            _auc = utils.auc(y_pred, y_true)

            self.result = pd.DataFrame({
                'recall': [_recall],
                'precision': [_precision],
                'f1': [_f1],
                'accuracy': [_accuracy],
                'auc': [_auc]
            }, index=[explainer.label])
        else:
            raise ValueError("'model_type' must be 'regression' or 'classification'")

        _residuals = pd.DataFrame({
            'y_hat': y_pred,
            'y': y_true,
            'residuals': _residuals,
            'label': explainer.label
        })

        self.residuals = _residuals

    def plot(self,
             objects=None,
             title="Reverse cumulative distribution of |residual|",
             show=False):
        """Plot the Model Performance explanation

        Parameters
        -----------
        objects : ModelPerformance object or array_like of ModelPerformance objects
            Additional objects to plot (default is None).
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

        # are there any other objects to plot?
        if objects is None:
            _df_list = [self.residuals.copy()]
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            _df_list = [self.residuals.copy(), objects.residuals.copy()]
        elif isinstance(objects, (list, tuple)):  # objects as tuple or array
            _df_list = [self.residuals.copy()]
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _df_list += [ob.residuals.copy()]
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)

        colors = _theme.get_default_colors(len(_df_list), 'line')
        fig = go.Figure()

        for i, _df in enumerate(_df_list):
            _abs_residuals = np.abs(_df['residuals'])
            _unique_abs_residuals = np.unique(_abs_residuals)

            fig.add_scatter(
                x=_unique_abs_residuals,
                y=1 - plot.ecdf(_abs_residuals)(_unique_abs_residuals),
                line_shape='hv',
                name=_df.iloc[0, _df.columns.get_loc('label')],
                marker=dict(color=colors[i])
            )

        fig.update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': 'outside',
                          'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'tickformat': ',.0%'})

        fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                          'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': '|residual|'})

        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          margin={'t': 78, 'b': 71, 'r': 30})

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
