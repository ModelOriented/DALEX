from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .protocol import Explainer
from .welford import WelfordState


def calculate_A(num_features: int) -> np.ndarray:
    """Calculate A parameter's exact form.
    from: https://github.com/iancovert/shapley-regression/blob/master/shapreg/shapley_unbiased.py"""
    # TODO: refactor, it's not readable
    p_coaccur = (
        np.sum((np.arange(2, num_features) - 1) / (num_features - np.arange(2, num_features)))
    ) / (
        num_features
        * (num_features - 1)
        * np.sum(1 / (np.arange(1, num_features) * (num_features - np.arange(1, num_features))))
    )
    A = np.eye(num_features) * 0.5 + (1 - np.eye(num_features)) * p_coaccur
    return A


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
    approx_samples: int = 20,
) -> np.ndarray:
    """Make predictions using different subsets of features
    feature_subsets is a (`num_samples` x `num_features`) True/False matrix

    for absent features we take the average prediction over `approx_samples`
    random observations from data
    for present features we set them as they are"""

    approx_samples = min(approx_samples, len(data))
    num_samples, num_features = feature_subsets.shape
    # TODO: maybe better to work with the different set of observations?
    data_sample = data.sample(n=approx_samples, replace=False).values

    # observarion_wide.shape == `approx_samples` x `num_samples` x `num_features`
    observation_wide = np.repeat(data_sample[None, :, :], repeats=num_samples, axis=0).swapaxes(
        0, 1
    )
    for feature in range(num_features):
        observation_feature_value = observation.iloc[0, feature]
        observation_wide[:, feature_subsets[:, feature], feature] = observation_feature_value

    predictions = predict_function(observation_wide.reshape(-1, num_features)).reshape(
        approx_samples, num_samples
    )
    predictions_approx = predictions.mean(0)
    return predictions_approx


def calculate_exact_result(
    A: np.ndarray,
    b: float,
    full_prediction: float,
    b_sum_squares: float,
    num_observations: int,
) -> Tuple[np.ndarray, float]:
    """Calculate the regression coefficients and uncertainty estimates.
    A - 2d matrix
    b - 1d vector"""
    ...


def unbiased_kernel_shap(
    explainer: Explainer,
    new_observation: pd.DataFrame,
    # keep_distributions: bool,
    n_samples: int,
    batch_size: int = 512,
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
            # TODO: not readable
            # (A.T * (B - C)[:, np.newaxis].T).T == A * (B - C)[None, :]
            b_sample = (
                S.astype(float).T * (eval_feature_subsets(S) - null_prediction)[:, np.newaxis].T
            ).T

        welford_state.update(observations=b_sample)
        count, b, b_sum_squares = welford_state
        shap_values, std = calculate_exact_result(A, b, full_prediction, b_sum_squares, count)

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
    new_observation_f = new_observation.loc[0, variable_names].apply(nice_format)
    variable_values = new_observation.loc[0, variable_names].values
    return pd.DataFrame(
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
