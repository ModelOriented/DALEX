import numpy as np
import narwhals.stable.v2 as nw
from .._native_resources import records, clean_contributions
from ...aspect._native_importance import descending
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
            if data.native('result') is None:
                resource.wait_for_update()
                data, progress, is_done = resource.get_result()
        except Exception as e:
            self.set_message(str(e))
            return
        self.is_done = is_done
        self.progress = progress
        result = clean_contributions(records(data.native('result')).numeric())
        result = result.group_by('variable_name', 'row').agg(nw.col('contribution').mean())
        result = result.sort(['variable_name', 'row']).with_columns(nw.col('contribution').abs().alias('abs_contribution'))
        output, outliers = [], {}
        for name in sorted(result['variable_name'].unique().to_list()):
            values = result.filter(nw.col('variable_name') == name)['abs_contribution'].to_numpy()
            values = values[~np.isnan(values)]
            if len(values):
                q1, median, q3 = np.quantile(values, [.25, .5, .75])
                low = max(q1 - 1.5*(q3-q1), values.min())
                high = min(q3 + 1.5*(q3-q1), values.max())
                mean, minimum, maximum = values.mean(), values.min(), values.max()
                unusual = values[values > high].tolist() + values[values < low].tolist()
            else:
                q1 = median = q3 = low = high = mean = minimum = maximum = np.nan
                unusual = []
            output.append((name, mean, median, minimum, maximum, q1, q3, low, high))
            if unusual:
                outliers[name] = unusual
        order = descending([row[1] for row in output])
        output = [output[i] for i in order]
        columns = ['variables', 'mean', 'median', 'min', 'max', 'q1', 'q3', 'lf', 'uf']
        self.data = {name: [row[i] for row in output] for i, name in enumerate(columns)}
        self.data.update(outliers=outliers, intercept=data.get('intercept'))
