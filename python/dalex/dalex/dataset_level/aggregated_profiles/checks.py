import pandas as pd
import numpy as np


def check_variable_type(variable_type):
    if variable_type not in ['numerical', 'categorical']:
        raise ValueError("variable_type needs to be 'numerical' or 'categorical'")


def prepare_all_variables(all_profiles, variables):
    # variables to use
    all_variables = all_profiles['_vname_'].dropna().unique().astype(str)

    if variables is not None:
        all_variables_intersect = set(all_variables).intersection(set(variables))
        if len(all_variables_intersect) == 0:
            raise ValueError("variables do not overlap with " + all_variables)
        all_variables = np.array(list(all_variables_intersect))

    return all_variables


def prepare_numerical_categorical(all_variables, all_profiles, variable_type):
    # only numerical or only factors?
    is_numeric = np.empty_like(all_variables, bool)
    for i, var in enumerate(all_variables):
        is_numeric[i] = not pd.api.types.is_string_dtype(all_profiles[var])

    if variable_type == 'numerical':
        vnames = all_variables[is_numeric]
        if vnames.shape[0] == 0:
            raise ValueError("There are no numerical variables")

        all_profiles['_x_'] = 0

    else:
        vnames = all_variables[~is_numeric]
        if vnames.shape[0] == 0:
            raise ValueError("There are no non-numerical variables")

        all_profiles['_x_'] = ""

    return all_profiles, vnames


def create_x(all_profiles, variable_type):
    # create _x_
    for variable in all_profiles['_vname_'].unique():
        where_variable = all_profiles['_vname_'] == variable
        all_profiles.loc[where_variable, '_x_'] = all_profiles.loc[where_variable, variable]

    # change x column to proper character values
    if variable_type == 'categorical':
        all_profiles.loc[:, '_x_'] = all_profiles.apply(lambda row: str(row[row['_vname_']]), axis=1)

    return all_profiles


def check_groups(groups):
    if groups is not None and not isinstance(groups, str):
        raise TypeError("groups must be None or str")

    # treating groups as list simplifies code
    return [] if groups is None else [groups]
