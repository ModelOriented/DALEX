import numpy as np
import narwhals.stable.v2 as nw
from .._native_resources import records, ShapStats
from ...aspect._native_importance import descending
from .._plot_container import PlotContainer


class ShapleyValuesContainer(PlotContainer):
    info = {
        'name': "Shapley Values",
        'plotType': "SHAPValues",
        'plotCategory': "Observation Level",
        'requiredParams': ["model", "observation"]
    }
    options_category = 'ShapleyValues'
    options = {}
    def _fit(self, model, observation):
        resource = self.arena.resource_manager.get_resource('ShapleyValues', {'model': model, 'observation': observation}, cache=self.use_cache)
        try:
            data, progress, is_done = resource.get_result()
            if data.native('stats') is None:
                resource.wait_for_update()
                data, progress, is_done = resource.get_result()
        except Exception as e:
            self.set_message(str(e))
            return
        self.is_done = is_done
        self.progress = progress
        value = data.native('stats')
        if isinstance(value, ShapStats):
            frame = nw.from_native(value.data, eager_only=True)
            display = value.display_values()
        else:
            frame = nw.from_native(value.reset_index(), eager_only=True)
            display = dict(zip(frame['variable_name'].to_list(), frame['variable_value'].to_list()))
        frame = frame[descending(frame['abs'].to_numpy()).tolist(), :]
        self.data = {'variables': frame['variable_name'].to_list(),
                     'variables_value': [display[name] for name in frame['variable_name'].to_list()],
                     **{name: frame[name].to_list() for name in ['mean', 'min', 'max', 'q1', 'q3']},
                     'intercept': float(resource.data.get('intercept'))}
