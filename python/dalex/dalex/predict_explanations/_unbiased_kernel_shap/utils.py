from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .protocol import Explainer
from .welford import WelfordState


def calculate_A(num_features: int) -> np.ndarray:
    """Calculate A parameter's exact form."""
    d = num_features
    p_coaccur = (np.sum((np.arange(2, d) - 1) / (d - np.arange(2, d)))) / (
        d * (d - 1) * np.sum(1 / (np.arange(1, d) * (d - np.arange(1, d))))
    )
    A = np.eye(d) * 0.5 + (1 - np.eye(d)) * p_coaccur
    return A


def calculate_b_sample(
    S: np.ndarray, S_predictions: np.ndarray, null_prediction: float
) -> np.ndarray:
    """given S - True/False matrix and its predictions calculate sample bs"""
    return S.astype(float) * (S_predictions - null_prediction)[:, None]


def sample_subsets(num_samples: int, num_features: int) -> np.ndarray:
    """Sample (`num_samples` x `num_features`) True/False matrix
    representing which feature we take into account."""

    # TODO: i dont think these weights are correct

    # Weighting kernel (probability of each subset size).
    weights = np.arange(1, num_features)
    weights = 1 / (weights * (num_features - weights))
    weights = weights / np.sum(weights)

    # TODO: need to optimize - we iterate over every sample...
    S = np.zeros((num_samples, num_features), dtype=bool)
    num_included = np.random.choice(num_features - 1, size=num_samples, p=weights) + 1
    for row, num in zip(S, num_included):
        inds = np.random.choice(num_features, size=num, replace=False)
        row[inds] = 1
    return S


def predict_on_subsets(
    feature_subsets: np.ndarray,
    observation: pd.DataFrame,
    data: pd.DataFrame,
    predict_function: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Make predictions using different subsets of features
    feature_subsets is a (`num_samples` x `num_features`) True/False matrix

    for absent features we take the average prediction over `approx_samples`
    random observations from data
    for present features we set them as they are"""

    num_samples, num_features = feature_subsets.shape

    # observarion_wide.shape == `approx_samples` x `num_samples` x `num_features`
    observation_wide = np.repeat(data.values[None, :, :], repeats=num_samples, axis=0).swapaxes(0, 1)
    for feature in range(num_features):
        observation_feature_value = observation.iloc[0, feature]
        observation_wide[:, feature_subsets[:, feature], feature] = observation_feature_value

    predictions = predict_function(observation_wide.reshape(-1, num_features)).reshape(
        -1, num_samples
    )
    predictions_approx = predictions.mean(0)
    return predictions_approx


def calculate_exact_result(
    A: np.ndarray,
    b: np.ndarray,
    full_prediction: float,
    null_prediction: float,
    b_cov: np.ndarray,
    num_observations: int,
) -> Tuple[np.ndarray, float]:
    """Calculate the regression coefficients and uncertainty estimates.
    A - 2d matrix
    b - 1d vector
    from: https://github.com/iancovert/shapley-regression/blob/master/shapreg/shapley_unbiased.py"""

    _, num_features = A.shape
    assert (num_features,) == b.shape

    A_inv = np.linalg.solve(A, np.eye(num_features))
    # TODO: they do not use null prediction for some reason
    betas = A_inv @ (b - ((A_inv @ b).sum() - full_prediction + null_prediction) / A_inv.sum())

    std = np.ones(num_features) * np.nan  # TODO: add actual compute

    return betas, std


def unbiased_kernel_shap(
    explainer: Explainer,
    new_observation: pd.DataFrame,
    # keep_distributions: bool,
    n_samples: int,
    batch_size: int = 10,
    # processes: int,
    # random_state: int,
    paired_sampling: bool = False,
) -> Tuple[pd.DataFrame, float, float, Optional[pd.DataFrame]]:
    """returns shap values result, prediction, intercept and optionally yhats_distributions"""

    num_features = new_observation.shape[1]
    eval_feature_subsets: Callable[[np.ndarray], np.ndarray] = partial(
        predict_on_subsets,
        observation=new_observation,
        data=explainer.data,
        predict_function=explainer.predict,
    )

    welford_state = WelfordState()
    null_prediction = eval_feature_subsets(np.zeros((1, num_features), dtype=bool))[0]
    # only one new_observation allowed
    full_prediction = explainer.predict(new_observation)[0]

    n_loops = int(np.ceil(n_samples / batch_size))
    A = calculate_A(num_features)
    for _ in range(n_loops):
        S = sample_subsets(batch_size, num_features)
        if paired_sampling:
            raise NotImplementedError
        else:
            b_sample = calculate_b_sample(S, eval_feature_subsets(S), null_prediction)

        welford_state.update(observations=b_sample)
        count, b, b_cov = welford_state
        shap_values, std = calculate_exact_result(
            A, b, full_prediction, null_prediction, b_cov, count
        )

    shap_result = wrap_shap_result(
        shap_values, new_observation, explainer.data.columns, explainer.label
    )
    return shap_result, full_prediction, null_prediction, None


def wrap_shap_result(
    shap_values: np.ndarray,
    new_observation: pd.DataFrame,
    variable_names: List[str],
    label: str,
) -> pd.DataFrame:
    new_observation_f = new_observation.iloc[0][variable_names].apply(nice_format)
    variable_values = new_observation.iloc[0][variable_names].values
    return pd.DataFrame.from_dict(
        {
            "variable": [" = ".join(pair) for pair in zip(variable_names, new_observation_f)],
            "contribution": shap_values,
            "variable_name": variable_names,
            "variable_value": variable_values,
            "sign": np.sign(shap_values),
            "label": label,
        }
    )


def nice_format(x) -> str:
    return str(x) if isinstance(x, str) else str(float(signif(x)))


#:# https://stackoverflow.com/a/59888924
def signif(x, p: int = 4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
