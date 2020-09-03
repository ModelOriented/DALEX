from dalex import Explainer
from .._plot_container import PlotContainer

class BreakDownContainer(PlotContainer):
    info = {
        'name': "Break Down",
        'plotType': "Breakdown",
        'plotCategory': "Observation Level",
        'requiredParams': ["model", "observation"]
    }
    options = {
    }
    def __init__(self, arena, model, observation):
        super().__init__(
            arena,
            name=self.__class__.info.get('name'),
            plot_category=self.__class__.info.get('plotCategory'),
            plot_type=self.__class__.info.get('plotType'),
            plot_component='Breakdown'
        )
        if not isinstance(model, Explainer):
            raise Exception('Invalid Explainer argument')
        bd = model.predict_parts(observation, type='break_down').result
        self.data = {
            'variables': bd[1:-1].variable_name.tolist(),
            'variables_value': bd[1:-1].variable_value.tolist(),
            'contribution': bd[1:-1].contribution.tolist(),
            'intercept': bd.contribution[0],
            'prediction': bd.cumulative.head(1)[0]
        }
        self.params = {
            'model': model.label,
            'observation': observation.index[0]
        }
