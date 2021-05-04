from pandas.api.types import is_numeric_dtype
from .._plot_container import PlotContainer


class ShapleyValuesDependenceContainer(PlotContainer):
    info = {
        'name': "Shapley Values Dependence",
        'plotType': "ShapleyValuesDependence",
        'plotCategory': "Dataset Level",
        'requiredParams': ["model", "variable"]
    }
    options_category = 'DatasetShapleyValues'
    options = {}
    def _fit(self, model, variable):
        if variable.variable not in model.variables:
            raise Exception('Variable is not a column of explainer')
        is_numeric = is_numeric_dtype(model.explainer.data[variable.variable])
        if is_numeric and variable.levels is None:
            self.plot_component = 'LinearShapleyDependence'
        else:
            self.plot_component = 'CategoricalShapleyDependence'
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
        result = result[result['variable_name'] == variable.variable]
        stats = result.groupby(['variable_value', 'row']).agg({'contribution': ['mean', 'min', 'max']}).contribution
        if self.plot_component == 'LinearShapleyDependence':
            stats = stats.sort_index()
        transform_index = float if self.plot_component == 'LinearShapleyDependence' else str
        self.data = {
            'x': [transform_index(x[0]) for x in stats.index],
            'mean': stats['mean'].values.tolist(),
            'min': stats['min'].values.tolist(),
            'max': stats['max'].values.tolist(),
            'variable': variable.variable
        }
