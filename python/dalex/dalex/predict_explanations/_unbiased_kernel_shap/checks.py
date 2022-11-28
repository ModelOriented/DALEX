from copy import deepcopy
from typing import Optional, Union
from warnings import warn

import numpy as np
import pandas as pd

from .protocol import Explainer


def check_columns_in_new_observation(new_observation: pd.DataFrame, explainer: Explainer) -> None:
    if not set(new_observation.columns).issubset(explainer.data.columns):
        raise ValueError("Columns in the new observation do not match these in training dataset.")


def check_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """also returns DataFrame with casted types"""
    result_observation = data.copy()
    for column in result_observation.columns:
        try:
            result_observation[column] = result_observation[column].astype(float)
        except ValueError as e:
            raise TypeError(f"All types must be numerical. {column} is not.") from e
    return result_observation


def check_new_observation(
    new_observation: Union[np.ndarray, pd.Series], explainer: Explainer
) -> pd.DataFrame:
    new_observation_ = deepcopy(new_observation)
    if isinstance(new_observation_, pd.Series):
        new_observation_ = new_observation_.to_frame().T
        new_observation_.columns = explainer.data.columns
    elif isinstance(new_observation_, np.ndarray):
        if new_observation_.ndim == 1:
            # make 1D array 2D
            new_observation_ = new_observation_.reshape((1, -1))
        elif new_observation_.ndim > 2:
            raise ValueError(f"Wrong new_observation number of dimensions: {new_observation_.ndim}")
        elif new_observation.shape[0] != 1:
            raise ValueError(
                f"Wrong new_observation contains {new_observation.shape[0]} observations"
            )

        new_observation_ = pd.DataFrame(data=new_observation_, columns=explainer.data.columns)

    elif isinstance(new_observation_, list):
        new_observation_ = pd.DataFrame(new_observation_).T
        new_observation_.columns = explainer.data.columns

    elif isinstance(new_observation_, pd.DataFrame):
        if new_observation.shape[0] != 1:
            raise ValueError("Wrong new_observation dimension")

        new_observation_.columns = explainer.data.columns
    else:
        raise TypeError(
            "new_observation must be a numpy.ndarray or pandas.Series or pandas.DataFrame"
        )

    if pd.api.types.is_bool_dtype(new_observation_.index):
        raise ValueError("new_observation index is of boolean type")

    return new_observation_


def check_processes(processes: int) -> int:
    from multiprocessing import cpu_count

    if processes > cpu_count():
        warn("You have asked for too many processes. Truncated to the number of physical CPUs.")

        return cpu_count()

    else:
        return processes


def check_random_state(random_state: Optional[int]) -> int:
    if random_state is None:
        random_state = np.random.randint()
    np.random.seed(random_state)

    return random_state
