import numpy as np
import narwhals.stable.v2 as nw

from . import plot, utils
from ... import _theme, _global_checks
from ..._explanation import Explanation, EmptyTable, CompatibilityTable, IndexedTable
from ..._dataframe import native_namespace
from ..._plotting import result_frame


class ModelPerformance(Explanation):
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
        self.result = EmptyTable()
        self.residuals = None

    def _repr_html_(self):
        return self.result._repr_html_()

    residuals = CompatibilityTable()

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
        namespace = native_namespace(explainer.data) if explainer.data is not None else 'pandas'

        if self.model_type == 'regression':
            _mse = utils.mse(y_pred, y_true)
            _rmse = utils.rmse(y_pred, y_true)
            _r2 = utils.r2(y_pred, y_true)
            _mae = utils.mae(y_pred, y_true)
            _mad = utils.mad(y_pred, y_true)

            metrics = {'mse': [_mse], 'rmse': [_rmse], 'r2': [_r2], 'mae': [_mae], 'mad': [_mad]}
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

            metrics = {
                'recall': [_recall],
                'precision': [_precision],
                'f1': [_f1],
                'accuracy': [_accuracy],
                'auc': [_auc]
            }
        else:
            raise ValueError("'model_type' must be 'regression' or 'classification'")

        metrics['label'] = [explainer.label]
        self.result = IndexedTable(nw.from_dict(metrics, backend=namespace).to_native(), index_column='label')
        _residuals = nw.from_dict({
            'y_hat': np.array(y_pred, copy=True),
            'y': np.array(y_true, copy=True),
            'residuals': np.array(_residuals, copy=True),
            'label': [explainer.label] * len(y_pred)
        }, backend=namespace).to_native()

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
            _df_list = [result_frame(self, table='residuals')]
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            _df_list = [result_frame(self, table='residuals'), result_frame(objects, table='residuals')]
        elif isinstance(objects, (list, tuple)):  # objects as tuple or array
            _df_list = [result_frame(self, table='residuals')]
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _df_list += [result_frame(ob, table='residuals')]
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
