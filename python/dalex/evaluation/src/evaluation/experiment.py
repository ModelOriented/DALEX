from typing import Tuple

import pandas as pd

from models import Model


def run_experiment(
    model: Model, explainer, n_repeats: int, X: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    """runs `n_repeats` times model explanations for all observations of X for all columns
    returns:
        - pd.DataFrame with the following columns: `run_id`, `model_name`, `explanation_name (hparams incl.)`, *X.columns
        - the time of inference in seconds"""
    # TODO
