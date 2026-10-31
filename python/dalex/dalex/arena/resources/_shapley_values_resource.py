from .._native_resources import ShapRecords, records, stats
from .._resource import Resource


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


class ShapleyValuesResource(Resource):
    resource_type = 'ShapleyValues'
    required_params = ["model", "observation"]
    options_category = 'ShapleyValues'
    options = {
        'B': {'default': 20, 'desc': 'Number of random paths'},
        'cpus': {'default': 4, 'desc': 'Number of parallel processes'}
    }
    def add_to_data(self, shap):
        with self.mutex:
            if self.data.get('intercept') is None:
                self.data['intercept'] = shap.intercept
            result = ShapRecords.from_shap(shap)
            previous = self.data.native('result')
            if previous is not None:
                result = records(previous).append(result)
            self.data['result'] = result
            self.data['stats'] = stats(result)

    def _fit(self, model, observation):
        row = observation.get_row_for_model(model)
        if row is None:
            raise Exception('Observation is not valid for given model.')
        B = self.get_option('B')
        cpus = self.get_option('cpus')
        with self.mutex:
            self.progress = 0
        for i in range(B // cpus):
            with self.mutex:
                if self.cancel_signal:
                    return
            shap = model.explainer.predict_parts(
                row,
                type='shap',
                B=cpus,
                processes=cpus
            )
            self.add_to_data(shap)
            with self.mutex:
                self.progress = (i + 1) * cpus / B
            self._emit_update()
        if B % cpus > 0:
            with self.mutex:
                if self.cancel_signal:
                    return
            shap = model.explainer.predict_parts(
                row,
                type='shap',
                B=B % cpus,
                processes=B % cpus
            )
            self.add_to_data(shap)
        with self.mutex:
            self.progress = 1
            self.is_done = True
            self._emit_update()
