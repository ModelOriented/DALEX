import numpy as np
import narwhals.stable.v2 as nw

from ...fairness._basics.checks import check_protected
from .._plot_container import PlotContainer
from ..._dataframe import column_values
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
        series = nw.from_native(exp.data, eager_only=True)[variable.variable]
        if not isinstance(series.dtype, (nw.String, nw.Categorical, nw.Enum, nw.Object)):
            self.set_message('Select categorical variable to check fairness')
            return
        protected = check_protected(column_values(exp.data, variable.variable, preserve_nulls=True), False)
        if exp.model_type == 'classification':
            output = {group: {} for group in np.unique(protected)}
            for cutoff in self.arena.get_option(self.plot_type, 'cutoffs'):
                cutoffs = checks.check_cutoff(protected, cutoff, False)
                matrix = utils.SubgroupConfusionMatrix(exp.y, y_hat, protected, cutoffs)
                metrics = utils.SubgroupConfusionMatrixMetrics(matrix).subgroup_confusion_matrix_metrics
                for group, scores in metrics.items():
                    output[group][cutoff] = rm_nan(scores)
            self.data = {'subgroups': output}
        else:
            self.plot_component = 'RegressionFairness'
            self.data = {}
            for subgroup in np.unique(protected):
                table = nw.from_native(exp.model_fairness(protected, subgroup).result_as(), eager_only=True)
                groups = table['subgroup'].to_list()
                self.data[subgroup] = {name: dict(zip(groups, table[name].to_list()))
                                      for name in table.columns if name != 'subgroup'}
