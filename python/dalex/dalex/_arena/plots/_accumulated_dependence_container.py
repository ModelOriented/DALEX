from pandas.api.types import is_numeric_dtype
from dalex import Explainer
from .._plot_container import PlotContainer
from ..helper import get_variables

class AccumulatedDependenceContainer(PlotContainer):
    info = {
        'name': "Accumulated Dependence",
        'plotType': 'AccumulatedDependence',
        'plotCategory': 'Dataset Level',
        'requiredParams': ['model', 'variable']
    }
    options = {
        'grid_type': { 'default': 'quantile', 'desc': 'grid type "quantile" or "uniform"'},
        'grid_points': { 'default': 101, 'desc': 'Maximum number of points for profile' },
        'N': { 'default': 500, 'desc': 'Number of observations to use. None for all.' }
    }
    def __init__(self, arena, model, variable):
        super().__init__(
            arena,
            name=self.__class__.info.get('name'),
            plot_category=self.__class__.info.get('plotCategory'),
            plot_type=self.__class__.info.get('plotType')
        )
        if not isinstance(model, Explainer):
            raise Exception('Invalid Explainer argument')
        if not variable in get_variables(model):
            raise Exception('Variable is not a column of explainer')
        if is_numeric_dtype(model.data[variable]):
            self.plot_component = 'LinearDependence'
            variable_type = 'numerical'
        else:
            self.plot_component = 'CategoricalDependence'
            variable_type = 'categorical'
        profile = model.model_profile(
            type='accumulated',
            variables=variable,
            variable_type=variable_type,
            center=False,
            grid_points=arena.get_option(self.plot_type, 'grid_points'),
            # TODO
            #variable_splits_type=arena.get_option(self.plot_type, 'grid_type'),
            N=arena.get_option(self.plot_type, 'N')
        )
        self.data = {
            'x': profile.result['_x_'].tolist(),
            'y': profile.result['_yhat_'].tolist(),
            'variable': variable,
            'base': 0
        }
        self.params = {
            'model': model.label,
            'variable': variable
        }
