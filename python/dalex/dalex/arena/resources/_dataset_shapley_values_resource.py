import pandas as pd
import numpy as np
from .._resource import Resource


class DatasetShapleyValuesResource(Resource):
    resource_type = 'DatasetShapleyValues'
    required_params = ["model"]
    options_category = 'DatasetShapleyValues'
    options = {
        'B': {'default': 4, 'desc': 'Number of random paths'},
        'N': {'default': 500, 'desc': 'Number of randomly sampled rows from dataset'},
        'cpus': {'default': 4, 'desc': 'Number of parallel processes'}
    }
    def add_to_data(self, shaps):
        with self.mutex:
            if self.data.get('intercept') is None:
                self.data['intercept'] = shaps[0].intercept
            results = [shap.result[shap.result.B != 0] for shap in shaps]
            if self.data.get('result') is not None:
                result = pd.concat([self.data['result']] + results)
            else:
                result = pd.concat(results)
            self.data['result'] = result

    def _fit(self, model):
        B = self.get_option('B')
        cpus = self.get_option('cpus')
        with self.mutex:
            self.progress = 0
        dataset = model.explainer.data
        rows = dataset.shape[0]
        # Sample N rows
        N = self.get_option('N')
        if N < rows:
            sampled_rows = np.random.choice(np.arange(rows), N, replace=False)
            dataset = dataset.iloc[sampled_rows, :]
            rows = N
        buffor = []
        for i in range(rows):
            with self.mutex:
                if self.cancel_signal:
                    return
            shap = model.explainer.predict_parts(
                dataset.iloc[i],
                type='shap',
                B=B,
                processes=cpus
            )
            shap.result['row'] = i
            buffor.append(shap)
            # Append buffor to results if buffor size >= 10% of already appended
            if (len(buffor) >= 0.1 * (i + 1 - len(buffor))) or i == rows - 1:
                self.add_to_data(buffor)
                buffor = []
                with self.mutex:
                    self.progress = (i + 1) / rows
                self._emit_update()
        with self.mutex:
            self.progress = 1
            self.is_done = True
            self._emit_update()
