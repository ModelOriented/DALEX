import multiprocessing as mp
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_ceteris_paribus(explainer,
                              new_observation,
                              variable_splits,
                              y,
                              processes,
                              verbose=True):
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
                                          verbose)

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
                               verbose=True):
    """
    Inner function that calculates ceteris paribus for all variables.

    :param explainer:
    :param data: new observations
    :param variable_splits: dictionary of split points
    :return: ceteris paribus profile for all variable
    """

    if processes == 1:
        profile = []
        for variable in tqdm(variable_splits, desc="Calculating ceteris paribus", disable=not verbose):
            split_points = variable_splits[variable]
            profile.append(single_variable_profile(predict_function, model, data, variable, split_points))
    else:
        pool = mp.get_context('spawn').Pool(processes)

        profile = pool.starmap_async(single_variable_profile, [(predict_function, model, data, variable, split_points)
                                                               for variable, split_points in
                                                               variable_splits.items()]).get()
        pool.close()

    profiles = pd.concat(profile)
    # convert the variable types
    if LooseVersion(pd.__version__) >= LooseVersion('1.2.0'):
        # convert_floating=False since pandas v1.2 seem to have issues
        profiles.loc[:, list(variable_splits)] = profiles.loc[:, list(variable_splits)].convert_dtypes(convert_floating=False)
    else:
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
    ids = np.repeat(data.index.values, split_points.shape[0])
    new_data = data.loc[ids, :]
    original = new_data.loc[:, variable].copy()
    new_data.loc[:, variable] = np.tile(split_points, data.shape[0])

    yhat = predict(model, new_data)

    new_data.loc[:, '_original_'] = original
    new_data.loc[:, '_yhat_'] = yhat
    new_data.loc[:, '_vname_'] = variable
    new_data.loc[:, '_ids_'] = ids

    return new_data


def calculate_variable_split(data,
                             variables,
                             grid_points,
                             variable_splits_type='uniform',
                             variable_splits_with_obs=False,
                             new_observation=None):
    """
    Calculate points for splitting the dataset

    :param data: dataset to split
    :param variables: variables to calculate ceteris paribus
    :param grid_points: how many points should split the dataset
    :param variable_splits_type: {'uniform', 'quantiles'}, optional way of calculating
        `variable_splits`. Set 'quantiles' for percentiles.
    :param variable_splits_with_obs: bool, optional add variable values of `new_observation`
        data to the `variable_splits`.
    :param new_observation: pd.DataFrame or np.ndarray, Observations for which predictions
        need to be explained.
    :return: dict, dictionary of split points for all variables
    """
    variable_splits = {}
    # grid points might be larger than the number of unique values
    probs = np.linspace(0, 1, grid_points)

    for variable in variables:
        variable_column = data.loc[:, variable]
        if pd.api.types.is_numeric_dtype(variable_column):
            if variable_splits_type == 'uniform':
                column_splits = np.linspace(np.min(variable_column),
                                            np.max(variable_column),
                                            grid_points)
            else:
                column_splits = np.unique(np.quantile(variable_column, probs))
            if variable_splits_with_obs:
                column_splits = np.concatenate((column_splits, new_observation.loc[:, variable]))
                column_splits = np.unique(column_splits)
                column_splits = np.sort(column_splits, kind='mergesort')

            variable_splits[variable] = column_splits
        else:
            variable_splits[variable] = variable_column.unique()

    return variable_splits
