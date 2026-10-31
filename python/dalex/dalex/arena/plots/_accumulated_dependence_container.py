from ..._dataframe import numeric_columns
import narwhals.stable.v2 as nw
from .._native_eda import display_categories
from .._plot_container import PlotContainer

class AccumulatedDependenceContainer(PlotContainer):
    info = {
        'name': "Accumulated Dependence",
        'plotType': 'AccumulatedDependence',
        'plotCategory': 'Dataset Level',
        'requiredParams': ['model', 'variable']
    }
    options_category = 'AccumulatedDependence'
    options = {
        'grid_type': { 'default': 'quantile', 'desc': 'grid type "quantile" or "uniform"'},
        'grid_points': { 'default': 101, 'desc': 'Maximum number of points for profile' },
        'N': { 'default': 500, 'desc': 'Number of observations to use. None for all.' }
    }
    def _fit(self, model, variable):
        if not variable.variable in model.variables:
            raise Exception('Variable is not a column of explainer')
        if variable.variable in numeric_columns(model.explainer.data):
            self.plot_component = 'LinearDependence'
            variable_type = 'numerical'
        else:
            self.plot_component = 'CategoricalDependence'
            variable_type = 'categorical'
        profile = model.explainer.model_profile(
            type='accumulated',
            variables=variable.variable,
            variable_type=variable_type,
            center=False,
            grid_points=self.get_option('grid_points'),
            variable_splits_type=self.get_option('grid_type'),
            N=self.get_option('N'),
            verbose=False
        )
        result = nw.from_native(profile.result_as(), eager_only=True)
        self.data = {
            'x': display_categories(result['_x_']).to_list(),
            'y': result['_yhat_'].to_list(),
            'variable': variable.variable,
            'base': 0
        }
