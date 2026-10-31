"""Explicit optional-framework job; no silently skipped neural dependencies."""
import os

import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx
from dalex._dataframe import to_numpy


@pytest.fixture(params=["trainable", "frozen"])
def neural_model(request):
    framework = os.environ.get("DALEX_NEURAL_FRAMEWORK", "torch")
    if framework == "torch":
        import torch
        torch.set_num_threads(1)
        model = torch.nn.Sequential(torch.nn.Linear(3, 1)).double()
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[2., -.5, .1]], dtype=torch.float64))
            model[0].bias.fill_(.3)
        if request.param == "frozen":
            # A frozen inference module can keep weights as buffers and expose
            # no parameters. Its input dtype/device still comes from the model.
            for name, parameter in list(model[0].named_parameters(recurse=False)):
                values = parameter.detach().clone()
                delattr(model[0], name)
                model[0].register_buffer(name, values)
            assert not list(model.parameters())

        def predict(model, frame):
            with torch.no_grad():
                return model(torch.tensor(to_numpy(frame), dtype=torch.float64)).numpy().reshape(-1)
    elif framework == "tensorflow":
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(3,), dtype="float64"),
            tf.keras.layers.Dense(1, dtype="float64",
                kernel_initializer=tf.keras.initializers.Constant([[2.], [-.5], [.1]]),
                bias_initializer=tf.keras.initializers.Constant(.3)),
        ])
        model.trainable = request.param == "trainable"

        def predict(model, frame):
            return model(to_numpy(frame), training=False).numpy().reshape(-1)
    else:
        raise ValueError(framework)
    return model, predict


@pytest.mark.parametrize("shap_type", ["DeepExplainer", "GradientExplainer"])
@pytest.mark.parametrize("api", ["predict_parts", "model_parts"])
def test_neural_shap_native_backend_parity(neural_model, shap_type, api):
    model, predict = neural_model
    if os.environ.get("DALEX_NEURAL_FRAMEWORK", "torch") == "torch":
        import torch
        default_dtype = torch.get_default_dtype()
        original_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    rng = np.random.default_rng(123)
    data = pl.DataFrame(rng.normal(size=(24, 3)), schema=["x", "z", "u"])
    target = 2 * data["x"].to_numpy() - .5 * data["z"].to_numpy() + .1 * data["u"].to_numpy() + .3
    results = []
    for backend in ("pandas", "polars", "arrow"):
        frame = {"pandas": data.to_pandas, "polars": lambda: data, "arrow": data.to_arrow}[backend]()

        def guarded(model, batch):
            assert type(batch) is type(frame), type(batch)
            return predict(model, batch)

        exp = dx.Explainer(model, frame, target, predict_function=guarded,
                           model_type="regression", label="model", verbose=False)
        kwargs = {"shap_explainer_type": shap_type}
        if shap_type == "GradientExplainer":
            kwargs.update(nsamples=512, rseed=42)
        if api == "predict_parts":
            obs = frame.iloc[[0]] if backend == "pandas" else (frame.head(1) if backend == "polars" else frame.slice(0, 1))
            result = exp.predict_parts(obs, type="shap_wrapper", **kwargs)
            values = np.asarray(result.result).reshape(1, 3)
            # Linear model additivity is an independent explanation invariant.
            difference = predict(model, obs)[0] - target.mean()
            np.testing.assert_allclose(values.sum(), difference,
                                       atol=1e-6 if shap_type == "DeepExplainer" else .3)
        else:
            result = exp.model_parts(type="shap_wrapper", N=5, random_state=42, **kwargs)
            values = np.asarray(result.result)
            assert values.shape[0] == 5
        assert exp.data is frame and np.isfinite(values).all()
        result.plot(show=False)
        results.append(values)
    for values in results[1:]:
        np.testing.assert_allclose(results[0], values, rtol=1e-6, atol=1e-6)
    if os.environ.get("DALEX_NEURAL_FRAMEWORK", "torch") == "torch":
        assert torch.get_default_dtype() == default_dtype
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original_weights[name], rtol=0, atol=0)
    import matplotlib.pyplot as plt
    plt.close("all")
