import pandas as pd
from .._plot_container import PlotContainer


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def iqr(x):
    return q3(x) - q1(x)


def lf(x):
    return max(q1(x) - (1.5 * iqr(x)), x.min())


def uf(x):
    return min(q3(x) + (1.5 * iqr(x)), x.max())


class ShapleyValuesVariableImportanceContainer(PlotContainer):
    info = {
        'name': "Shapley Variable Importance",
        'plotType': "ShapleyValuesVariableImportance",
        'plotCategory': "Dataset Level",
        'requiredParams': ["model"]
    }
    options_category = 'DatasetShapleyValues'
    options = {}
    def _fit(self, model):
        resource = self.arena.resource_manager.get_resource('DatasetShapleyValues', {'model': model}, cache=self.use_cache)
        try:
            data, progress, is_done = resource.get_result()
            if data.get('result') is None:
                resource.wait_for_update()
                data, progress, is_done = resource.get_result()
        except Exception as e:
            self.set_message(str(e))
            return
        self.is_done = is_done
        self.progress = progress
        result = data.get('result')

        stats = result.groupby(['variable_name', 'row']) \
            .agg({'contribution': ['mean']}) \
            .contribution
        stats['abs_contribution'] = stats['mean'].abs()
        stats = stats.reset_index()
        box_stats = stats.groupby(['variable_name']) \
            .agg({'abs_contribution': ['mean', 'max', 'min', 'median', q1, q3, lf, uf]}) \
            .abs_contribution.sort_values(by='mean', ascending=False)
        outliers1 = stats.loc[stats['abs_contribution'] > box_stats.uf.loc[stats['variable_name']].reset_index(drop=True)]
        outliers2 = stats.loc[stats['abs_contribution'] < box_stats.lf.loc[stats['variable_name']].reset_index(drop=True)]
        outliers = pd.concat([outliers1, outliers2]).groupby('variable_name')['abs_contribution'].apply(list).to_dict()
        self.data = {
            'variables': list(box_stats.index),
            'mean': box_stats['mean'].values.tolist(),
            'median': box_stats['median'].values.tolist(),
            'min': box_stats['min'].values.tolist(),
            'max': box_stats['max'].values.tolist(),
            'q1': box_stats['q1'].values.tolist(),
            'q3': box_stats['q3'].values.tolist(),
            'lf': box_stats['lf'].values.tolist(),
            'uf': box_stats['uf'].values.tolist(),
            'outliers': outliers,
            'intercept': data.get('intercept')
        }
