import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_ceteris_paribus(explainer,
                              new_observation,
                              variable_splits,
                              y):
    """
    Inner method that calculates ceteris paribus and some additional fields.

    Set profile and new_profile fields to actual values.

    :return: None
    """
    
    profiles = calculate_variable_profile(explainer,
                                          new_observation,
                                          variable_splits)

    profiles.loc[:, '_label_'] = explainer.label

    # add points of interests
    predictions = explainer.predict(new_observation)

    new_observation.loc[:, '_yhat_'] = predictions

    new_observation.loc[:, '_label_'] = explainer.label

    new_observation.loc[:, '_ids_'] = np.arange(new_observation.shape[0])

    if y is not None:
        new_observation.loc[:, '_y_'] = y

    return profiles, new_observation


def calculate_variable_profile(explainer,
                               data,
                               variable_splits):
    """
    Inner function that calculates ceteris paribus for all variables.

    :param explainer:
    :param data: new observations
    :param variable_splits: dictionary of split points
    :return: ceteris paribus profile for all variable
    """

    profile = []
    for variable in tqdm(variable_splits, desc="Calculating ceteris paribus!"):
        split_points = variable_splits[variable]
        profile.append(single_variable_profile(explainer, data, variable, split_points))

    return pd.concat(profile)


def single_variable_profile(explainer,
                            data,
                            variable,
                            split_points):
    """
    Inner function for calculating variable profile. This function is iterated over all variables.

    :param explainer:
    :param data: new observations
    :param variable: considered variable
    :param split_points: dataset is split over these points
    :return: ceteris paribus profile for one variable
    """
    # remember ids of selected points
    ids = np.repeat(data.index, split_points.shape[0])

    new_data = data.loc[ids, :]
    new_data.loc[:, variable] = np.tile(split_points, data.shape[0])

    yhat = explainer.predict(new_data)

    new_data.loc[:, '_yhat_'] = yhat
    new_data.loc[:, '_vname_'] = variable
    new_data.loc[:, '_ids_'] = ids

    return new_data


def calculate_variable_split(data, variables, grid_points):
    """
    Calculate points for splitting the dataset

    :param data: dataset to split
    :param variables: variables to calculate ceteris paribus
    :param grid_points: how many points should split the dataset
    :return: dict, dictionary of split points for all variables
    """
    variable_splits = {}
    for variable in variables:
        if np.issubdtype(data.loc[:, variable].dtype, np.floating):
            # grid points might be larger than the number of unique values
            probs = np.linspace(0, 1, grid_points)

            variable_splits[variable] = np.unique(np.quantile(data.loc[:, variable], probs))
        else:
            variable_splits[variable] = np.unique(data.loc[:, variable])

    return variable_splits
