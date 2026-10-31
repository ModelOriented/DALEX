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
        labels = np.asarray(exp.y, dtype=bool)
        scores = np.asarray(y_hat)
        positive = labels.sum()
        negative = len(labels) - positive
        if positive == 0 or negative == 0:
            self.set_message('Provided dataset contains only positive or only negative cases.', 'error')
            return
        valid = ~np.isnan(scores)
        scores, labels = scores[valid], labels[valid]
        order = np.argsort(scores, kind='stable')[::-1]
        scores, labels = scores[order], labels[order]
        ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1] if len(scores) else np.array([], dtype=int)
        tpr = labels.cumsum()[ends] / positive
        tnr = 1 - (~labels).cumsum()[ends] / negative
        cutoffs = scores[ends]
        maximum = self.get_option('grid_points')
        if len(cutoffs) > maximum:
            selected = np.sort(np.random.choice(len(cutoffs), maximum, replace=False))
            cutoffs, tpr, tnr = cutoffs[selected], tpr[selected], tnr[selected]
        self.data = {'cutoff': cutoffs.tolist(), 'specifity': [1] + tnr.tolist(),
                     'sensivity': [0] + tpr.tolist()}

    def test_arena(arena):
        if type(arena).__name__ != 'Arena' or type(arena).__module__ != 'dalex.arena.object':
            raise Exception('Invalid Arena argument')
        return next((True for model in arena.get_params('model') if model.explainer.model_type == 'classification'), False)
