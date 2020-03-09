import pandas as pd


def check_columns_in_new_observation(new_observation,
                                     explainer):
    if not set(new_observation.columns).issubset(explainer.data):
        raise ValueError("Columns in new observation does not match these in training dataset.")


def check_new_observation(new_observation):
    if not isinstance(new_observation, (pd.DataFrame, pd.Series)):
        raise TypeError("New observation must be DataFrame or Series")

    if isinstance(new_observation, pd.Series):
        new_observation = new_observation.to_frame().T

    return new_observation