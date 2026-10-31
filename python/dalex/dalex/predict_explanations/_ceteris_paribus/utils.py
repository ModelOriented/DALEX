"""Grid helpers and legacy pandas-returning adapters to native CP kernels."""
import numpy as np
import pandas as pd
import narwhals.stable.v2 as nw
from ..._dataframe import as_eager_dataframe, column_values, to_pandas


def calculate_ceteris_paribus(explainer, new_observation, variable_splits, y,
                              processes, verbose=True):
    from ._native import calculate
    profiles, observations = calculate(explainer, new_observation, variable_splits,
                                       y, processes, verbose)
    return profiles._dalex_pandas_view(), observations._dalex_pandas_view()


def calculate_variable_profile(predict_function, model, data, variable_splits,
                               processes, verbose=True):
    from ._native import profile_tables
    return profile_tables(predict_function, model, data, variable_splits,
                           processes, verbose=verbose)._dalex_pandas_view()


def single_variable_profile(predict, model, data, variable, split_points):
    from ._native import single_profile
    frame, index = single_profile(predict, model, data, variable, split_points)
    result = to_pandas(frame)
    result.index = index
    return result


def native_numeric_missing_states(data, variable_splits):
    """Keep validity provenance outside the backward-compatible result table."""
    if isinstance(data, pd.DataFrame):
        return {}
    frame = as_eager_dataframe(data)
    states = {}
    for variable, splits in variable_splits.items():
        if not any(value is None or value is pd.NA or value is pd.NaT or
                   (isinstance(value, (float, np.floating)) and np.isnan(value)) for value in splits):
            continue
        column = frame.get_column(variable)
        kinds = set()
        if column.dtype.is_numeric():
            if column.null_count():
                kinds.add('null')
            if isinstance(column.dtype, (nw.Float32, nw.Float64)) and column.is_nan().any():
                kinds.add('nan')
        states[variable] = kinds
    return states


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
        column = as_eager_dataframe(data).get_column(variable)
        dtype = column.dtype
        # pandas 1.5 does not classify numeric ArrowDtype via is_numeric_dtype.
        numeric = data[variable].dtype.kind in 'iufc' if isinstance(data, pd.DataFrame) else (
            dtype.is_numeric() and not isinstance(dtype, nw.Decimal))
        if numeric:
            # Reference datasets can contain millions of rows even when only
            # a small CP sample is explained. Keep missing filtering and range
            # reductions native; exporting/scanning Python cells made grid
            # construction dominate the entire sampled explanation.
            observed = column.drop_nulls() if column.null_count() else column
            if dtype.is_float() and observed.is_nan().any():
                observed = observed.filter(~observed.is_nan())
            if not len(observed):
                # Native float validity distinguishes null from NaN. An
                # all-missing column can therefore have two interventions.
                if not isinstance(data, pd.DataFrame):
                    count = column.null_count()
                    levels = ([None] if count else []) + ([np.nan] if count < len(column) else [])
                    column_splits = np.array(levels, dtype=object if count else float)
                else:
                    column_splits = np.array([np.nan])
            elif variable_splits_type == 'uniform':
                column_splits = np.linspace(observed.min(),
                                            observed.max(),
                                            grid_points)
            else:
                column_splits = np.unique(np.quantile(observed.to_numpy(), probs))
            if variable_splits_with_obs:
                values = np.concatenate((column_splits, column_values(new_observation, variable, preserve_nulls=True)))
                nulls = [value is None or value is pd.NA or value is pd.NaT for value in values]
                if any(nulls):
                    numeric = np.unique(np.asarray([value for value, absent in zip(values, nulls) if not absent], dtype=float))
                    if isinstance(data, pd.DataFrame):
                        column_splits = np.unique(np.append(numeric, np.nan))
                    else:
                        missing_nan = np.isnan(numeric)
                        column_splits = np.array(numeric[~missing_nan].tolist() + [None]
                                                 + ([np.nan] if missing_nan.any() else []), dtype=object)
                else:
                    column_splits = np.sort(np.unique(values), kind='mergesort')

            variable_splits[variable] = column_splits
        else:
            # Narwhals keeps the first-observed order without a pandas Series
            # allocation. Use the exact scalar export for dictionary/temporal
            # columns rather than a backend's lossy generic NumPy conversion.
            unique = column.unique(maintain_order=True).to_frame().to_native()
            variable_splits[variable] = column_values(unique, variable, preserve_nulls=True)

    return variable_splits
