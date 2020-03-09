from .utils import *


class ModelPerformance:
    def __init__(self,
                 model_type,
                 cutoff=0.5):

        self.cutoff = cutoff
        self.model_type = model_type
        self.result = None

    def fit(self, explainer):

        if explainer.y_hat is not None:
            y_pred = explainer.y_hat
        else:
            y_pred = explainer.predict(explainer.data)

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
