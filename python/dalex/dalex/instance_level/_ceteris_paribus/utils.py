import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_ceteris_paribus(explainer,
                              new_observation,
                              variable_splits,
                              y,
                              processes,
                              disable=False):
    """
    Inner method that calculates ceteris paribus and some additional fields.

    Set profile and new_profile fields to actual values.

    :return: None
    """

    profiles = calculate_variable_profile(explainer.predict_function,
                                          explainer.model,
                                          new_observation,
                                          variable_splits,
                                          processes,
                                          disable)

    profiles.loc[:, '_label_'] = explainer.label

    # add points of interests
    predictions = explainer.predict(new_observation)

    new_observation.loc[:, '_yhat_'] = predictions

    new_observation.loc[:, '_label_'] = explainer.label

    new_observation.loc[:, '_ids_'] = new_observation.index.values

    if y is not None:
        new_observation.loc[:, '_y_'] = y

    return profiles, new_observation


def calculate_variable_profile(predict_function,
                               model,
                               data,
                               variable_splits,
                               processes,
                               disable=False):
    """
    Inner function that calculates ceteris paribus for all variables.

    :param explainer:
    :param data: new observations
    :param variable_splits: dictionary of split points
    :return: ceteris paribus profile for all variable
    """

    if processes == 1:
        profile = []
        for variable in tqdm(variable_splits, desc="Calculating ceteris paribus", disable=disable):
            split_points = variable_splits[variable]
            profile.append(single_variable_profile(predict_function, model, data, variable, split_points))
    else:
        pool = mp.Pool(processes)

        profile = pool.starmap_async(single_variable_profile, [(predict_function, model, data, variable, split_points)
                                                               for variable, split_points in
                                                               variable_splits.items()]).get()
        pool.close()

    profiles = pd.concat(profile)
    # convert the variable types
    profiles.loc[:, list(variable_splits)] = profiles.loc[:, list(variable_splits)].convert_dtypes()

    return profiles


def single_variable_profile(predict,
                            model,
                            data,
                            variable,
                            split_points):
    """
    Inner function for calculating variable profile. This function is iterated over all variables.

    :param predict:
    :param data: new observations
    :param variable: considered variable
    :param split_points: dataset is split over these points
    :return: ceteris paribus profile for one variable
    """
    # remember ids of selected points
    ids = np.repeat(data.index, split_points.shape[0])
    new_data = data.loc[ids, :]
    original = new_data.loc[:, variable]
    new_data.loc[:, variable] = np.tile(split_points, data.shape[0])

    yhat = predict(model, new_data)

    new_data.loc[:, '_original_'] = original
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
        if pd.api.types.is_numeric_dtype(data.loc[:, variable]):
            # grid points might be larger than the number of unique values
            probs = np.linspace(0, 1, grid_points)

            variable_splits[variable] = np.unique(np.quantile(data.loc[:, variable], probs))
        else:
            variable_splits[variable] = np.unique(data.loc[:, variable])

    return variable_splits
