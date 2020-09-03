from pandas.api.types import is_numeric_dtype
from dalex import Explainer
from .._plot_container import PlotContainer
from ..helper import get_variables

class CeterisParibusContainer(PlotContainer):
    info = {
        'name': 'Ceteris Paribus',
        'plotType': 'CeterisParibus',
        'plotCategory': 'Observation Level',
        'requiredParams': ['model', 'variable', 'observation']
    }
    options = {
        'grid_points': { 'default': 101, 'desc': 'Maximum number of points for profile' },
        'grid_type': { 'default': 'quantile', 'desc': 'grid type "quantile" or "uniform"'}
    }
    def __init__(self, arena, model, variable, observation):
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
        cp = model.predict_profile(
            observation,
            variables=variable,
            grid_points=arena.get_option(self.plot_type, 'grid_points'),
            variable_splits_type=arena.get_option(self.plot_type, 'grid_type')
        )
        if is_numeric_dtype(observation[variable]):
            self.plot_component = 'NumericalCeterisParibus'
        else:
            self.plot_component = 'CategoricalCeterisParibus'
        self.data = {
            'x': cp.result[variable].tolist(),
            'y': cp.result['_yhat_'].tolist(),
            'variable': variable,
            'min': cp.result['_yhat_'].min(),
            'max': cp.result['_yhat_'].max(),
            'observation': cp.new_observation.iloc[0].to_dict()
        }
        self.params = {
            'model': model.label,
            'variable': variable,
            'observation': observation.index[0]
        }
