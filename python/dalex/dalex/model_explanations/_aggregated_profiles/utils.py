"""Legacy profile preparation/publication adapters over native reductions."""
import numpy as np
import pandas as pd
import narwhals.stable.v2 as nw
from ..._dataframe import to_pandas, column_values


def aggregate_profiles(all_profiles, mean_prediction, type, groups, center, span, verbose=True):
    from ._native import aggregate_long_profiles
    return aggregate_long_profiles(all_profiles, mean_prediction, type, groups,
                                    center, span)._dalex_pandas_view()


def split_over_variables_and_labels(split_profile, type, groups, span):
    if not len(split_profile):
        return None
    from ._native import reduce_profiles
    # Historically this inner helper always applied local weights, even when
    # called directly with "partial"; only the outer helper chose plain means.
    kind = 'accumulated' if type == 'accumulated' else 'conditional'
    return to_pandas(reduce_profiles(split_profile, kind, groups, span=span))


def norm(x, loc, scale):
    return np.exp(-1 * ((x - loc) / scale) ** 2 / 2) / np.sqrt(2 * np.pi) / scale


def prepare_numerical_categorical(all_profiles, variables, variable_type):
    # variables to use
    frame = nw.from_native(all_profiles, eager_only=True)
    all_variables = frame['_vname_'].drop_nulls().unique(maintain_order=True).to_numpy()

    if variables is not None:
        all_variables_intersect = set(all_variables).intersection(set(variables))
        if len(all_variables_intersect) == 0:
            raise ValueError('variables do not overlap with profiles')
        all_variables = np.array(list(all_variables_intersect))

    # only numerical or only factors?
    is_numeric = np.empty_like(all_variables, bool)
    for i, var in enumerate(all_variables):
        dtype = frame.schema[var]
        is_numeric[i] = dtype.is_numeric() or dtype == nw.Boolean
        if dtype == nw.Unknown and isinstance(all_profiles, pd.DataFrame):
            # Complex pandas storage is outside the Narwhals dtype vocabulary.
            is_numeric[i] = all_profiles[var].dtype.kind == 'c'

    if variable_type == 'numerical':
        vnames = all_variables[is_numeric]
        if vnames.shape[0] == 0:
            raise ValueError("There are no numerical variables")

        all_profiles['_x_'] = 0.0

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
    if variable_type == 'categorical':
        labels, original_labels = categorical_profile_labels(all_profiles)
        # Complete-column assignment also supports pandas 3's StringDtype.
        all_profiles['_x_'] = labels
        all_profiles['_original_'] = original_labels
        return all_profiles
    frame = nw.from_native(all_profiles, eager_only=True)
    names = frame['_vname_'].to_numpy()
    values = np.asarray(column_values(all_profiles, '_x_', preserve_nulls=True), dtype=float).copy()
    for variable in frame['_vname_'].unique(maintain_order=True).to_list():
        positions = np.flatnonzero(names == variable)
        source = np.asarray(column_values(all_profiles, variable, preserve_nulls=True), dtype=float)
        values[positions] = source[positions]
    # Publish the prepared column into the same mutable legacy pandas object.
    all_profiles['_x_'] = values
    return all_profiles


def categorical_profile_labels(profiles):
    """Match categorical levels without merging missing and literal labels.

    Numeric codes and nullable booleans can have different pandas dtypes in the
    intervention/original columns. Missing values share the grid's label, with
    a suffix only if a real category already uses that display string.
    """
    names = profiles['_vname_'].to_numpy()
    original = profiles['_original_'].to_numpy()
    labels = np.empty(len(profiles), dtype=object)
    original_labels = np.empty(len(profiles), dtype=object)

    def missing(value):
        return pd.api.types.is_scalar(value) and pd.isna(value)

    for variable in dict.fromkeys(names):
        positions = np.flatnonzero(names == variable)
        values = profiles[variable].to_numpy()[positions]
        strings = [str(value) for value in values]
        missing_mask = [missing(value) for value in values]
        nonmissing = {label for label, absent in zip(strings, missing_mask) if not absent}
        # Numeric interventions can become nullable integers while originals
        # remain floats. Match their values before choosing display strings.
        level_labels = {value: label for value, label, absent in zip(values, strings, missing_mask) if not absent}
        missing_label = next((label for label, absent in zip(strings, missing_mask) if absent), '<missing>')
        while missing_label in nonmissing:
            missing_label += ' (missing)'
        labels[positions] = [missing_label if absent else label for label, absent in zip(strings, missing_mask)]
        original_labels[positions] = [missing_label if missing(value) else level_labels.get(value, str(value))
                                      for value in original[positions]]
    return labels, original_labels
