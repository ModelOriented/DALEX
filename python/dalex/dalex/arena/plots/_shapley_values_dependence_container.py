import numpy as np
import narwhals.stable.v2 as nw
from .._native_resources import records, clean_contributions
from ...aspect._native_importance import descending
from ..._dataframe import numeric_columns
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
        is_numeric = variable.variable in numeric_columns(model.explainer.data)
        if is_numeric and variable.levels is None:
            self.plot_component = 'LinearShapleyDependence'
        else:
            self.plot_component = 'CategoricalShapleyDependence'
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
        from .._native_eda import display_categories
        value = records(data.native('result'))
        result = clean_contributions(nw.from_native(value.partitions[variable.variable], eager_only=True))
        if result['variable_value'].dtype.is_float():
            result = result.filter(~nw.col('variable_value').is_nan())
        if isinstance(result['variable_value'].dtype, (nw.Categorical, nw.Enum)):
            result = result.with_columns(nw.col('variable_value').cast(nw.String))
        result = result.group_by('variable_value', 'row', drop_null_keys=True).agg(
            nw.col('contribution').mean().alias('mean'), nw.col('contribution').min().alias('min'),
            nw.col('contribution').max().alias('max')).sort(['variable_value', 'row'])
        values = display_categories(result['variable_value']).to_list()
        transform = float if self.plot_component == 'LinearShapleyDependence' else str
        self.data = {'x': [transform(value) for value in values],
                     **{name: result[name].to_list() for name in ['mean', 'min', 'max']},
                     'variable': variable.variable}
