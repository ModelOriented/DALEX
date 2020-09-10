from pandas.api.types import is_numeric_dtype
from .._plot_container import PlotContainer

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
    def _fit(self, model, variable, observation):
        if not variable.variable in model.variables:
            raise Exception('Variable is not a column of explainer')
        row = observation.get_row()
        cp = model.explainer.predict_profile(
            row,
            variables=variable.variable,
            grid_points=self.arena.get_option(self.plot_type, 'grid_points'),
            variable_splits_type=self.arena.get_option(self.plot_type, 'grid_type'),
            verbose=False
        )
        if is_numeric_dtype(row[variable.variable]):
            self.plot_component = 'NumericalCeterisParibus'
        else:
            self.plot_component = 'CategoricalCeterisParibus'
        self.data = {
            'x': cp.result[variable.variable].tolist(),
            'y': cp.result['_yhat_'].tolist(),
            'variable': variable.variable,
            'min': cp.result['_yhat_'].min(),
            'max': cp.result['_yhat_'].max(),
            'observation': cp.new_observation.iloc[0].to_dict()
        }
