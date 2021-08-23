import pandas as pd
import numpy as np
from .._plot_container import PlotContainer

class ROCContainer(PlotContainer):
    info = {
        'name': 'Receiver Operating Characterstic',
        'plotType': 'ROC',
        'plotCategory': 'Model Performance',
        'requiredParams': ['model']
    }
    options_category = 'ROC'
    options = {
        'grid_points': { 'default': 101, 'desc': 'Maximum number of points for ROC curve' },
    }

    def _fit(self, model):
        exp = model.explainer
        if exp.model_type != 'classification':
            self.set_message('ROC plot is only available for classificators')
            return

        y_hat = exp.predict(exp.data) if exp.y_hat is None else exp.y_hat
        df = pd.DataFrame({ 'y': exp.y.astype(bool), 'y_hat': y_hat })

        P_n = df.y.sum()
        N_n = df.shape[0] - P_n
        if P_n == 0 or N_n == 0:
            self.set_message('Provided dataset contains only positive or only negative cases.', 'error')
            return

        tpr_temp = df.groupby('y_hat').sum().reset_index().sort_values('y_hat', ascending=False)
        fpr_temp = df.assign(y=1-df.y).groupby('y_hat').sum().reset_index().sort_values('y_hat', ascending=False)
        df['TPR'] = tpr_temp.y.cumsum() / P_n
        df['TNR'] = 1 - (fpr_temp.y.cumsum() / N_n)

        grid_points = self.get_option('grid_points')
        if df.shape[0] > grid_points:
            df = df.sample(grid_points).sort_values('y_hat', ascending=False)

        self.data = {
            'cutoff': df['y_hat'].tolist(),
            'specifity': [1] + df['TNR'].tolist(),
            'sensivity': [0] + df['TPR'].tolist()
        }

    def test_arena(arena):
        if type(arena).__name__ != 'Arena' or type(arena).__module__ != 'dalex.arena.object':
            raise Exception('Invalid Arena argument')
        return next((True for model in arena.get_params('model') if model.explainer.model_type == 'classification'), False)
