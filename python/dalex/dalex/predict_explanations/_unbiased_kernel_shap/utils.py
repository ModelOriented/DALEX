from typing import Optional, Tuple

import numpy as np
import pandas as pd

def unbiased_kernel_shap(
    explainer,
    new_observation: pd.DataFrame,
    keep_distributions: bool,
    n_samples: int,
    processes: int,
    random_state: int,
) -> Tuple[pd.DataFrame, float, float, Optional[pd.DataFrame]]:
    raise NotImplementedError


def calculate_yhats_distributions(explainer) -> pd.DataFrame:
    data_yhat = explainer.predict(explainer.data)

    return pd.DataFrame(
        {
            "variable_name": "all_data",
            "variable": "all data",
            "id": np.arange(explainer.data.shape[0]),
            "prediction": data_yhat,
            "label": explainer.label,
        }
    )
