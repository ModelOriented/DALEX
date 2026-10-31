import narwhals.stable.v2 as nw
from ..._dataframe import observation_display_values
from .._native_eda import display_categories, numeric
from .._plot_container import PlotContainer

class CeterisParibusContainer(PlotContainer):
    info = {
        'name': 'Ceteris Paribus',
        'plotType': 'CeterisParibus',
        'plotCategory': 'Observation Level',
        'requiredParams': ['model', 'variable', 'observation']
    }
    options_category = 'CeterisParibus'
    options = {
        'grid_points': { 'default': 101, 'desc': 'Maximum number of points for profile' },
        'grid_type': { 'default': 'quantile', 'desc': 'grid type "quantile" or "uniform"'}
    }
    def _fit(self, model, variable, observation):
        if not variable.variable in model.variables:
            raise Exception('Variable is not a column of explainer')
        row = observation.get_row_for_model(model)
        if row is None:
            self.set_message('Observation is not valid for given model.')
            return
        cp = model.explainer.predict_profile(
            row,
            variables=variable.variable,
            grid_points=self.get_option('grid_points'),
            variable_splits_type=self.get_option('grid_type'),
            variable_splits_with_obs=False,
            verbose=False
        )
        if numeric(nw.from_native(row)[variable.variable]):
            self.plot_component = 'NumericalCeterisParibus'
        else:
            self.plot_component = 'CategoricalCeterisParibus'
        result = nw.from_native(cp.result_as(partition=variable.variable), eager_only=True)
        attributes = observation_display_values(cp._native_observation_frame())
        if '_ids_' in attributes:
            attributes['_ids_'] = observation.get_label()
        self.data = {
            'x': display_categories(result[variable.variable]).to_list(),
            'y': result['_yhat_'].to_list(),
            'variable': variable.variable,
            'min': result['_yhat_'].min(),
            'max': result['_yhat_'].max(),
            'observation': attributes
        }
