"""External Kernel SHAP must honor per-call seeds after background sampling."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from concurrent.futures import ThreadPoolExecutor

import dalex as dx
from dalex._dataframe import to_numpy


class NonlinearModel:
    def predict(self, data):
        values = to_numpy(data)
        return values[:, 0]*values[:, 1] + np.sin(values[:, 2]*values[:, 3]) + np.cos(values[:, 4])*values[:, 5]**2


@pytest.mark.parametrize("sample", [None, 20])
def test_kernel_shap_seeded_backend_parity_and_rng_isolation(sample):
    import shap  # import-time library initialization is outside the RNG check

    data = pl.DataFrame(np.random.default_rng(123).normal(size=(40, 6)), schema=[f"x{i}" for i in range(6)])
    model = NonlinearModel()
    y = model.predict(data)
    before = np.random.get_state()
    results = []
    try:
        for frame in (data.to_pandas(), data, data.to_arrow(), data):
            exp = dx.Explainer(model, frame, y, model_type="regression", verbose=False)
            from dalex._dataframe import take_rows
            result = exp.predict_parts(
                take_rows(frame, [0]), type="shap_wrapper", N=sample, random_state=42,
                shap_explainer_type="KernelExplainer", nsamples=24, l1_reg=0, silent=True,
            )
            results.append(result.result)
            assert exp.data is frame
        shared = dx.Explainer(model, data, y, model_type="regression", verbose=False)

        def explain_seeded(_):
            return shared.predict_parts(
                data.head(1), type="shap_wrapper", N=sample, random_state=42,
                shap_explainer_type="KernelExplainer", nsamples=24, l1_reg=0, silent=True,
            ).result

        with ThreadPoolExecutor(max_workers=4) as pool:
            results.extend(pool.map(explain_seeded, range(4)))
        assert shared.data is data
        after = np.random.get_state()
    finally:
        np.random.set_state(before)
    for values in results[1:]:
        np.testing.assert_allclose(results[0], values, atol=1e-10, rtol=1e-10)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
