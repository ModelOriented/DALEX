from dalex import Explainer
from .._plot_container import PlotContainer

class ShapleyValuesContainer(PlotContainer):
    info = {
        'name': "Shapley Values",
        'plotType': "SHAPValues",
        'plotCategory': "Observation Level",
        'requiredParams': ["model", "observation"]
    }
    options = {
        'B': { 'default': 10, 'desc': 'Number of random paths' }
    }
    def _fit(self, model, observation):
        row = observation.get_row_for_model(model)
        if row is None:
            self.set_message('Observation is not valid for given model.')
            return
        shap = model.explainer.predict_parts(
            row,
            type='shap',
            B=self.arena.get_option(self.plot_type, 'B')
        )
        intercept = shap.intercept
        result = shap.result
        result = result[result.B != 0]
        def q1(x):
            return x.quantile(0.25)
        def q3(x):
            return x.quantile(0.75)
        stats = result.groupby(['variable_name', 'variable_value']) \
            .agg({'contribution': ['mean', 'max', 'min', q1, q3]}) \
            .contribution
        stats['abs'] = stats['mean'].abs()
        stats = stats.sort_values('abs', ascending=False).reset_index()
        self.data = {
            'variables': stats.variable_name.tolist(),
            'variables_value': stats.variable_value.tolist(),
            'mean': stats['mean'].tolist(),
            'min': stats['min'].tolist(),
            'max': stats['max'].tolist(),
            'q1': stats.q1.tolist(),
            'q3': stats.q3.tolist(),
            'intercept': intercept.astype(float)
        }
