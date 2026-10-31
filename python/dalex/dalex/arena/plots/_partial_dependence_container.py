from ..._dataframe import numeric_columns
import narwhals.stable.v2 as nw
from .._native_eda import display_categories
from .._plot_container import PlotContainer

class PartialDependenceContainer(PlotContainer):
    info = {
        'name': "Partial Dependence",
        'plotType': 'PartialDependence',
        'plotCategory': 'Dataset Level',
        'requiredParams': ['model', 'variable']
    }
    options_category = 'PartialDependence'
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
            profile = model.explainer.model_profile(
                type='partial',
                variables=variable.variable,
                variable_type='numerical',
                grid_points=self.get_option('grid_points'),
                variable_splits_type=self.get_option('grid_type'),
                N=self.get_option('N'),
                verbose=False
            )
        else:
            self.plot_component = 'CategoricalDependence'
            profile = model.explainer.model_profile(
                type='partial',
                variables=variable.variable,
                variable_type='categorical',
                N=self.get_option('N'),
                verbose=False
            )
        result = nw.from_native(profile.result_as(), eager_only=True)
        self.data = {
            'x': display_categories(result['_x_']).to_list(),
            'y': result['_yhat_'].to_list(),
            'variable': variable.variable,
            'base': profile.mean_prediction
        }
