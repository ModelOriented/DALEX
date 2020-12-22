import numpy as np
import pandas as pd
from tqdm import tqdm


def aggregate_profiles(all_profiles, mean_prediction, type, groups, center, span, verbose=True):
    if type == 'partial':
        aggregated_profiles = \
            all_profiles.groupby(['_vname_', '_label_', '_x_'] + groups, sort=False)['_yhat_'].mean().reset_index()

    else:
        # split all_profiles into groups
        tqdm.pandas(desc='Calculating accumulated dependency', disable=not verbose) if type == 'accumulated' else tqdm.pandas(
            desc="Calculating conditional dependency", disable=not verbose)
        aggregated_profiles = \
            all_profiles. \
                loc[:, ["_vname_", "_label_", "_x_", "_yhat_", "_ids_", "_original_"] + groups]. \
                groupby(['_vname_', '_label_']). \
                progress_apply(lambda split_profile: split_over_variables_and_labels(split_profile.copy(deep=True),
                                                                                     type, groups, span)). \
                reset_index(level=[0, 1])  # remove level_2
        # deepcopy due to https://github.com/ModelOriented/DALEX/issues/278

    aggregated_profiles.loc[:, '_ids_'] = 0

    if type == 'accumulated' and center:
        aggregated_profiles.loc[:, '_yhat_'] = aggregated_profiles.loc[:, '_yhat_'] - \
                                               aggregated_profiles.loc[:, '_yhat_'].mean() + \
                                               mean_prediction

    # postprocessing
    if len(groups) != 0:
        aggregated_profiles['_groups_'] = aggregated_profiles.loc[:, groups].apply(
            lambda row: '_'.join(row.astype(str)), axis=1)
        aggregated_profiles.drop(columns=groups, inplace=True)

        aggregated_profiles.loc[:, '_label_'] = \
            aggregated_profiles.loc[:, ['_label_', '_groups_']].apply(lambda row: '_'.join(row), axis=1)

    return aggregated_profiles


def split_over_variables_and_labels(split_profile, type, groups, span):
    """
    Inner function that calculates actual conditional profiles for one variable only. Iterated over each variable and group.

    :param split_profile: pandas.DataFrame, one group of the dataset (with only one variable)
    :param groups: str, name of grouping variable
    :return: pd.DataFrame, dataframe with calculated conditional profile for only one variable
    """

    if split_profile.shape[0] == 0:
        return None

    if pd.api.types.is_numeric_dtype(split_profile['_x_']):
        # for continuous variables we will calculate weighted average
        # where weights come from gaussian kernel and distance between points
        # scaling factor, range if the range i > 0
        split_profile['_original_'] = split_profile['_original_'].astype('float')
        range_x = split_profile['_x_'].max() - split_profile['_x_'].min()

        if range_x == 0:
            range_x = 1

        # scalled differences
        diffs = (split_profile['_original_'] - split_profile['_x_']) / range_x

        split_profile['_w_'] = norm(diffs, 0, span)

    else:
        # for categorical variables we will calculate weighted average
        # but weights are 0-1, 1 if it's the same level and 0 otherwise
        split_profile['_w_'] = split_profile['_original_'] == split_profile['_x_']

    if type == 'accumulated':
        # diffs
        split_profile['_yhat_'] = split_profile. \
            groupby('_ids_')['_yhat_']. \
            transform(lambda column: column.diff())

        # diff causes NaNs at the beginning of each group
        split_profile.loc[np.isnan(split_profile['_yhat_']), '_yhat_'] = 0

    par_profile = split_profile.groupby(['_x_'] + groups, sort=False). \
        apply(lambda point: (point['_yhat_'] * point['_w_']).sum() / point['_w_'].sum() \
        if point['_w_'].sum() != 0 else 0)

    par_profile.name = '_yhat_'
    par_profile = par_profile.reset_index()

    if type == 'accumulated':
        if len(groups) == 0:
            par_profile['_yhat_'] = par_profile['_yhat_'].cumsum()
        else:
            par_profile['_yhat_'] = par_profile.groupby(groups, sort=False)['_yhat_'].transform(
                lambda column: column.cumsum())

    return par_profile


def norm(x, loc, scale):
    return np.exp(-((x - loc) / scale) ** 2 / 2) / np.pi / np.sqrt(2) / scale


def prepare_numerical_categorical(all_profiles, variables, variable_type):
    # variables to use
    all_variables = all_profiles['_vname_'].dropna().unique()  # variables do not need to be string

    if variables is not None:
        all_variables_intersect = set(all_variables).intersection(set(variables))
        if len(all_variables_intersect) == 0:
            raise ValueError("variables do not overlap with " + all_variables)
        all_variables = np.array(list(all_variables_intersect))

    # only numerical or only factors?
    is_numeric = np.empty_like(all_variables, bool)
    for i, var in enumerate(all_variables):
        is_numeric[i] = pd.api.types.is_numeric_dtype(all_profiles[var])

    if variable_type == 'numerical':
        vnames = all_variables[is_numeric]
        if vnames.shape[0] == 0:
            raise ValueError("There are no numerical variables")

        all_profiles['_x_'] = 0

    else:
        vnames = all_variables[~is_numeric]
        if variables is not None:
            # take all if the user wants to convert numerical into categorical
            vnames = all_variables
        elif vnames.shape[0] == 0:
            raise ValueError("There are no non-numerical variables")

        all_profiles['_x_'] = ""

    return all_profiles, vnames


def prepare_x(all_profiles, variable_type):
    # create _x_
    for variable in all_profiles['_vname_'].unique():
        where_variable = all_profiles['_vname_'] == variable
        all_profiles.loc[where_variable, '_x_'] = all_profiles.loc[where_variable, variable]

    # change x column to proper character values
    if variable_type == 'categorical':
        all_profiles.loc[:, '_x_'] = all_profiles.apply(lambda row: str(row[row['_vname_']]), axis=1)

    return all_profiles