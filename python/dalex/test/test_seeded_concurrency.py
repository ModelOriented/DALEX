"""Explicitly seeded explanation calls must not share NumPy's global generator."""
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx


class Model:
    pass


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("family", ["importance", "shap", "profile", "aspect_default", "aspect_shap"])
def test_seeded_calls_are_independent_across_threads(backend, family):
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(64, 3)), columns=["x", "z", "u"])
    data = frame if backend == "pandas" else pl.from_pandas(frame)
    y = (2 * frame.x - frame.z).to_numpy() + rng.normal(size=64) * .1
    obs = data.iloc[[0]] if backend == "pandas" else data.head(1)

    def calculate(seed, barrier=None):
        entered = False

        def predict(model, batch):
            nonlocal entered
            # The legacy pandas constructor probes array acceptance. The barrier
            # belongs to explanation inference, after random sampling has begun.
            if isinstance(batch, np.ndarray):
                return 2 * batch[:, 0] - batch[:, 1]
            if barrier is not None and not entered:
                entered = True
                barrier.wait(timeout=15)
            return 2 * batch["x"].to_numpy() - batch["z"].to_numpy()

        exp = dx.Explainer(Model(), data, y, predict_function=predict,
                           model_type="regression", label="model", verbose=False,
                           precalculate=False)
        if family == "importance":
            result = exp.model_parts(N=32, B=5, random_state=seed).result
        elif family == "shap":
            result = exp.predict_parts(obs, type="shap", N=32, B=5, random_state=seed).result
        elif family.startswith("aspect_"):
            result = dx.Aspect(exp).predict_parts(
                obs, type=family.split("_")[1], N=32, B=5, random_state=seed,
                variable_groups={"pair": ["x", "z"], "single": ["u"]},
            ).result
        else:
            result = exp.model_profile(N=32, variables=["x"], grid_points=5,
                                       random_state=seed, verbose=False).result
        return result

    seeds = [7, 31, 42, 123]
    preserved = np.random.get_state()
    expected = [calculate(seed) for seed in seeds]
    np.random.seed(8675309)
    before = np.random.get_state()
    barrier = threading.Barrier(len(seeds))
    try:
        with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
            actual = list(pool.map(lambda seed: calculate(seed, barrier), seeds))
        after = np.random.get_state()
    finally:
        np.random.set_state(preserved)
    for reference, result in zip(expected, actual):
        pd.testing.assert_frame_equal(reference, result)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
