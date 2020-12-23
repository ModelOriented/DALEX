import numpy as np
import pandas as pd

from . import plot, utils
from ... import _theme, _global_checks


class ModelPerformance:
    """Calculate model-level model performance measures

    Parameters
    -----------
    model_type : {'regression', 'classification'}
        Model task type that is used to choose the proper performance measures.
    cutoff : float, optional
        Cutoff for predictions in classification models. Needed for measures like
        recall, precision, acc, f1 (default is `0.5`).

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
             geom="ecdf",
             title=None,
             show=False):
        """Plot the Model Performance explanation

        Parameters
        -----------
        objects : ModelPerformance object or array_like of ModelPerformance objects
            Additional objects to plot (default is `None`).
        geom: {'ecdf', 'roc', 'lift'}
            Type of plot determines how residuals shall be summarized.
        title : str, optional
            Title of the plot (default depends on the `type` attribute).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object that can 
            be edited or saved using the `write_image()` method (default is `True`).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """

        if geom not in ("ecdf", "roc", "lift"):
            raise TypeError("geom should be one of {'ecdf', 'roc', 'lift'}")
        
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
        
        if geom == 'ecdf':
            fig = plot.plot_ecdf(_df_list, colors, title)
        elif geom == 'roc':
            fig = plot.plot_roc(_df_list, colors, title)
        elif geom == 'lift':
            fig = plot.plot_lift(_df_list, colors, title)
        else:
            raise TypeError("geom should be one of {'ecdf', 'roc', 'lift'}")

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
