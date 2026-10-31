import numpy as np
import narwhals.stable.v2 as nw
from .._plot_container import PlotContainer

class FeatureImportanceContainer(PlotContainer):
    info = {
        'name': "Variable Importance",
        'plotType': "FeatureImportance",
        'plotCategory': "Dataset Level",
        'requiredParams': ["model"]
    }
    options_category = 'VariableImportance'
    options = {
        'N': { 'default': None, 'desc': 'Number of observations to use. None for all.' },
        'B': { 'default': 10, 'desc': 'Number of permutation rounds to perform each variable' }
    }
    def _fit(self, model):
        importance = model.explainer.model_parts(N=self.get_option('N'), B=self.get_option('B'))
        frame = nw.from_native(importance.result_as(table='permutation'), eager_only=True)
        names = [name for name in frame.columns if name not in ('_baseline_', '_full_model_')]
        summaries = []
        for name in names:
            values = frame[name].to_numpy()
            values = values[~np.isnan(values)]
            if len(values):
                mean, low, high = values.mean(), values.min(), values.max()
                q1, q3 = np.quantile(values, [.25, .75])
            else:
                mean = low = high = q1 = q3 = np.nan
            summaries.append((name, mean, low, high, q1, q3))
        means = np.asarray([row[1] for row in summaries])
        valid = np.flatnonzero(~np.isnan(means))
        order = len(valid)-1-np.argsort(means[valid][::-1], kind='quicksort')[::-1]
        order = np.r_[valid[order], np.flatnonzero(np.isnan(means))]
        summaries = [summaries[i] for i in order]
        columns = ['variables', 'dropout_loss', 'min', 'max', 'q1', 'q3']
        self.data = {name: [row[i] for row in summaries] for i, name in enumerate(columns)}
        self.data['base'] = np.nanmean(frame['_full_model_'].to_numpy())
