import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype
from .._plot_container import PlotContainer
from dalex.fairness._group_fairness import utils, checks

def rm_nan(obj):
    return { k: (None if np.isnan(obj[k]) or np.isinf(obj[k]) else obj[k]) for k in obj.keys() }

class FairnessCheckContainer(PlotContainer):
    info = {
        'name': 'Fairness',
        'plotType': 'Fairness',
        'plotCategory': 'Dataset Level',
        'requiredParams': ['model', 'variable']
    }
    options_category = 'Fairness'
    options = {
        'cutoffs': { 'default': [x / 100 for x in range(5, 100, 5)], 'desc': 'List of tested cutoff levels' },
    }
    def _fit(self, model, variable):
        if not variable.variable in model.variables:
            raise Exception('Variable is not a column of explainer')
        exp = model.explainer
        y_hat = exp.predict(exp.data) if exp.y_hat is None else exp.y_hat
        protected = exp.data[variable.variable]
        if exp.model_type != 'classification':
            self.set_message('Fairness plot is only available for classificators')
            return
        if not is_object_dtype(protected):
            self.set_message('Select categorical variable to check fairness')
            return

        output_df = None
        for cutoff in self.get_option('cutoffs'):
            cutoff_dict = checks.check_cutoff(protected, cutoff, False)
            sub_confusion_matrix = utils.SubgroupConfusionMatrix(exp.y, y_hat, protected, cutoff_dict)
            sub_confusion_matrix_metrics = utils.SubgroupConfusionMatrixMetrics(sub_confusion_matrix)
            df = sub_confusion_matrix_metrics.to_vertical_DataFrame()
            df['cutoff'] = cutoff
            output_df = df if output_df is None else output_df.append(df)

        output = {}
        for (subgroup, x) in output_df.set_index('metric').groupby('subgroup'):
            output[subgroup] = {}
            for (cutoff, y) in x.groupby('cutoff'):
                output[subgroup][cutoff] = rm_nan(y['score'].to_dict())

        self.data = { 'subgroups': output }

    def test_arena(arena):
        if type(arena).__name__ != 'Arena' or type(arena).__module__ != 'dalex.arena.object':
            raise Exception('Invalid Arena argument')
        return next((True for model in arena.get_params('model') if model.explainer.model_type == 'classification'), False)
