import narwhals.stable.v2 as nw
from .._plot_container import PlotContainer

class BreakDownContainer(PlotContainer):
    info = {
        'name': "Break Down",
        'plotType': "Breakdown",
        'plotCategory': "Observation Level",
        'requiredParams': ["model", "observation"]
    }
    options_category = 'BreakDown'
    options = {
    }
    def _fit(self, model, observation):
        row = observation.get_row_for_model(model)
        if row is None:
            self.set_message('Observation is not valid for given model.')
            return
        bd = nw.from_native(model.explainer.predict_parts(row, type='break_down').result_as(), eager_only=True)
        self.data = {
            'variables': bd['variable_name'][1:-1].to_list(),
            'variables_value': bd['variable_value'][1:-1].to_list(),
            'contribution': bd['contribution'][1:-1].to_list(),
            'intercept': bd['contribution'][0],
            'prediction': bd['cumulative'][-1]
        }
