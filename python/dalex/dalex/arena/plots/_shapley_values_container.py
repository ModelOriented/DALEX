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
            if data.get('stats') is None:
                resource.wait_for_update()
                data, progress, is_done = resource.get_result()
        except Exception as e:
            self.set_message(str(e))
            return
        self.is_done = is_done
        self.progress = progress
        stats = data.get('stats').sort_values('abs', ascending=False).reset_index()
        self.data = {
            'variables': stats.variable_name.tolist(),
            'variables_value': stats.variable_value.tolist(),
            'mean': stats['mean'].tolist(),
            'min': stats['min'].tolist(),
            'max': stats['max'].tolist(),
            'q1': stats.q1.tolist(),
            'q3': stats.q3.tolist(),
            'intercept': resource.data.get('intercept').astype(float)
        }
