from scipy.stats import norm
import numpy as np
import pandas as pd
from tqdm import tqdm


def aggregate_profiles(all_profiles, ceteris_paribus, type, groups, intercept, span):
    if type == 'partial':
        aggregated_profiles = \
            all_profiles.groupby(['_vname_', '_label_', '_x_'] + groups)['_yhat_'].mean().reset_index()

    else:
        observations = ceteris_paribus.new_observation

        # just initialisation
        if not pd.api.types.is_string_dtype(all_profiles['_x_']):
            all_profiles.loc[:, '_original_'] = \
                all_profiles.apply(lambda row: observations.loc[row['_ids_'], row['_vname_']], axis=1)

        else:
            all_profiles.loc[:, '_original_'] = \
                all_profiles.apply(lambda row: str(observations.loc[row['_ids_'], row['_vname_']]), axis=1)

        # split all_profiles into groups
        tqdm.pandas(desc='Calculating accumulated dependency!') if type == 'accumulated' else tqdm.pandas(
            desc="Calculating conditional dependency!")
        aggregated_profiles = \
            all_profiles. \
            loc[:, ["_vname_", "_label_", "_x_", "_yhat_", "_ids_", "_original_"] + groups]. \
            groupby(['_vname_', '_label_']). \
            progress_apply(lambda split_profile: split_over_variables_and_labels(split_profile, type, groups, span))

    # postprocessing
    if len(groups) != 0:
        aggregated_profiles.loc[:, '_label_'] = \
            aggregated_profiles.loc[:, ['_label_', '_groups_']].apply(lambda row: '_'.join(row), axis=1)

    aggregated_profiles.loc[:, '_ids_'] = 0

    if type == 'partial':
        if not intercept:
            aggregated_profiles.loc[:, '_yhat_'] = aggregated_profiles.loc[:, '_yhat_'] - all_profiles[
                '_yhat_'].mean()

        aggregated_profiles = aggregated_profiles
    elif type == 'conditional':
        if not intercept:
            aggregated_profiles.loc[:, '_yhat_'] = aggregated_profiles.loc[:, '_yhat_'] - all_profiles[
                '_yhat_'].mean()
        aggregated_profiles = aggregated_profiles.reset_index().rename(columns={'level_2': '_grid_'})
    else:
        if intercept:
            aggregated_profiles.loc[:, '_yhat_'] = aggregated_profiles.loc[:, '_yhat_'] + all_profiles[
                '_yhat_'].mean()
        aggregated_profiles = aggregated_profiles.reset_index().rename(columns={'level_2': '_grid_'})

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

    if not pd.api.types.is_string_dtype(split_profile['_x_']):
        # for continuous variables we will calculate weighted average
        # where weights come from gaussian kernel and distance between points
        # scaling factor, range if the range i > 0
        range_x = split_profile['_x_'].max() - split_profile['_x_'].min()

        if range_x == 0:
            range_x = 1

        # scalled differences
        diffs = (split_profile['_original_'] - split_profile['_x_']) / range_x

        split_profile['_w_'] = norm(0, span).pdf(diffs)

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

    par_profile = split_profile.groupby(['_x_'] + groups). \
        apply(lambda point: (point['_yhat_'] * point['_w_']).sum() / point['_w_'].sum() \
        if point['_w_'].sum() != 0 else 0)

    par_profile.name = '_yhat_'
    par_profile = par_profile.reset_index()

    if type == 'accumulated':
        if len(groups) == 0:
            par_profile['_yhat_'] = par_profile['_yhat_'].cumsum()
        else:
            par_profile['_yhat_'] = par_profile.groupby(groups)['_yhat_'].transform(
                lambda column: column.cumsum())

    return par_profile
