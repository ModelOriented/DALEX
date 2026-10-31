from .._native_resources import ShapRecords, records
import numpy as np
from ..._dataframe import take_rows
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
                first = shaps[0][0] if isinstance(shaps[0], tuple) else shaps[0]
                self.data['intercept'] = first.intercept
            result = self.data.native('result')
            result = records(result) if result is not None else None
            for item in shaps:
                shap, row = item if isinstance(item, tuple) else (item, None)
                current = ShapRecords.from_shap(shap, row=row)
                result = current if result is None else result.append(current)
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
            sampled_rows = np.random.choice(rows, N, replace=False)
            dataset = take_rows(dataset, sampled_rows)
            rows = N
        buffor = []
        for i in range(rows):
            with self.mutex:
                if self.cancel_signal:
                    return
            shap = model.explainer.predict_parts(
                take_rows(dataset, [i]),
                type='shap',
                B=B,
                processes=cpus
            )
            buffor.append((shap, i))
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
