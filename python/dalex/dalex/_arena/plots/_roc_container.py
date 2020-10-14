import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from .._plot_container import PlotContainer

class ROCContainer(PlotContainer):
    info = {
        'name': 'Receiver Operating Characterstic',
        'plotType': 'ROC',
        'plotCategory': 'Model Performance',
        'requiredParams': ['model']
    }
    options = {
        'grid_points': { 'default': 101, 'desc': 'Maximum number of points for ROC curve' },
    }
    def _fit(self, model):
        exp = model.explainer
        if exp.model_type != 'classification':
            self.set_message('ROC plot is only available for classificators')
            return

        y_hat = exp.predict(exp.data) if exp.y_hat is None else exp.y_hat
        df = pd.DataFrame({ 'y': exp.y.astype(bool), 'y_hat': y_hat }).sort_values('y_hat', ascending=False)
        P_n = np.sum(df.y)
        N_n = df.shape[0] - P_n
        if P_n == 0 or N_n == 0:
            self.set_message('Provided dataset contains only positive or only negative cases.', 'error')
            return
        grid_points = self.arena.get_option(self.plot_type, 'grid_points')
        df['TPR'] = np.cumsum(df.y) / P_n
        df['TNR'] = 1 - (np.cumsum(1 - df.y) / N_n)
        if df.shape[0] > grid_points:
            df = df.sample(grid_points).sort_values('y_hat', ascending=False)
        self.data = {
            'cutoff': df['y_hat'].tolist(),
            'specifity': df['TNR'].tolist(),
            'sensivity': df['TPR'].tolist()
        }
