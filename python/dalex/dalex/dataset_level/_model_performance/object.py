import plotly.graph_objects as go

from dalex.dataset_level._model_performance.plot import ecdf
from .utils import *


class ModelPerformance:
    def __init__(self,
                 model_type,
                 cutoff=0.5):
        """
        Constructor for ModelPerformance.

        :param model_type: either "regression" or "classification" determines measures to calculate
        :param cutoff: float, a cutoff for classification models, needed for measures like recall, precision, ACC, F1

        :return None
        """

        self.cutoff = cutoff
        self.model_type = model_type
        self.result = None
        self.residuals = None

    def fit(self, explainer):

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
            mse_ = mse(y_pred, y_true)
            rmse_ = rmse(y_pred, y_true)
            r2_ = r2(y_pred, y_true)
            mae_ = mae(y_pred, y_true)
            mad_ = mad(y_pred, y_true)

            self.result = pd.DataFrame(
                {
                    'mse': [mse_],
                    'rmse': [rmse_],
                    'r2': [r2_],
                    'mae': [mae_],
                    'mad': [mad_]
                })
        elif self.model_type == 'classification':
            tp = ((y_true == 1) * (y_pred >= self.cutoff)).sum()
            fp = ((y_true == 0) * (y_pred >= self.cutoff)).sum()
            tn = ((y_true == 0) * (y_pred < self.cutoff)).sum()
            fn = ((y_true == 1) * (y_pred < self.cutoff)).sum()

            recall_ = recall(tp, fp, tn, fn)
            precision_ = precision(tp, fp, tn, fn)
            f1_ = f1(tp, fp, tn, fn)
            accuracy_ = accuracy(tp, fp, tn, fn)
            auc_ = auc(y_pred, y_true)

            self.result = pd.DataFrame({
                'recall': [recall_],
                'precision': [precision_],
                'f1': [f1_],
                'accuracy': [accuracy_],
                'auc': [auc_]
            })
        else:
            raise ValueError("'model_type' must be 'regression' or 'classification'")

        _residuals = pd.DataFrame({
            'y_hat': y_pred,
            'y': y_true,
            'residuals': _residuals,
            'label': explainer.label
        })

        self.residuals = _residuals

    def plot(self, objects=None, title="Reverse cumulative distribution of |residual|", show=False):
        """
        Plot function for ModelPerformance class.

        :param objects: object of ModelPerformance class or list or tuple containing such objects
        :param title: str, the plot's title
        :param show: True shows the plot, False returns the plotly Figure object that can be edited or saved using `write_image()` method

        :return None or plotly Figure (see :param show)
        """

        # are there any other objects to plot?
        if objects is None:
            n = 1
            _residuals_df_list = [self.residuals.copy()]
        elif isinstance(objects, self.__class__):  # allow for objects to be a single element
            n = 2
            _residuals_df_list = [self.residuals.copy(), objects.residuals.copy()]
        else:  # objects as tuple or array
            n = len(objects) + 1
            _residuals_df_list = [self.residuals.copy()]
            for ob in objects:
                if not isinstance(ob, self.__class__):
                    raise TypeError("Some explanations aren't of ModelPerformance class")
                _residuals_df_list += [ob.residuals.copy()]

        fig = go.Figure()

        for i in range(n):
            _residuals_df = _residuals_df_list[i]
            _abs_residuals = np.abs(_residuals_df['residuals'])
            _unique_abs_residuals = np.unique(_abs_residuals)

            fig.add_scatter(
                x=_unique_abs_residuals,
                y=1 - ecdf(_abs_residuals)(_unique_abs_residuals),
                line_shape='hv',
                name=_residuals_df.iloc[0, _residuals_df.columns.get_loc('label')]
            )

        fig.update_yaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': 'outside',
                          'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'tickformat': ',.0%'})

        fig.update_xaxes({'type': 'linear', 'gridwidth': 2, 'zeroline': False, 'automargin': True, 'ticks': "outside",
                          'tickcolor': 'white', 'ticklen': 10, 'fixedrange': True, 'title_text': '|residual|'})

        fig.update_layout(title_text=title, title_x=0.15, font={'color': "#371ea3"}, template="none",
                          margin={'t': 78, 'b': 71, 'r': 30})

        if show:
            fig.show(config={'displaylogo': False, 'staticPlot': False,
                             'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d',
                                                        'zoom2d',
                                                        'pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d',
                                                        'toggleSpikelines', 'hoverCompareCartesian',
                                                        'hoverClosestCartesian']})
        else:
            return fig
