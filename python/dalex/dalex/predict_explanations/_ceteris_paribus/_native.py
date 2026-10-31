"""Native CP prediction tables; pandas is only a legacy publication boundary."""
import multiprocessing as mp
import numpy as np
import narwhals.stable.v2 as nw
from narwhals.dependencies import is_pandas_dataframe, get_pandas
from tqdm import tqdm

from ..._dataframe import take_rows, set_columns, as_eager_dataframe
from ._table import ProfileTables, ObservationTable


def _missing(value):
    pandas = get_pandas()
    return value is None or pandas is not None and (value is pandas.NA or value is pandas.NaT)


def single_profile(predict, model, data, variable, splits, label=None):
    positions = np.repeat(np.arange(data.shape[0], dtype=np.int64), len(splits))
    expanded = take_rows(data, positions)
    frame = as_eager_dataframe(expanded)
    namespace = nw.get_native_namespace(frame)
    original_column = frame.get_column(variable)
    # NumPy arrays of requested names contain np.str_ scalars. Preserve the
    # actual stored label when replacing a column: sklearn rejects a mixture of
    # Python str and np.str_ column-label types even when their values match.
    variable = original_column.name
    original = original_column.alias('_original_')
    index = expanded.index.copy() if is_pandas_dataframe(expanded) else positions.copy()
    ids = index.values if is_pandas_dataframe(expanded) else positions
    numeric = frame.schema[variable].is_numeric() and not isinstance(frame.schema[variable], nw.Decimal)
    if is_pandas_dataframe(expanded):
        numeric = expanded[variable].dtype.kind in 'iufc'
    if numeric:
        if is_pandas_dataframe(expanded):
            points = np.asarray([np.nan if _missing(value) else float(value) for value in splits])
            replacements = np.tile(points, data.shape[0])
            # A complete typed replacement avoids pandas' implicit narrowing
            # and nullable-integer assignment rules on older supported releases.
            working = frame.with_columns(nw.new_series(variable, replacements, dtype=nw.Float64,
                                                       backend=namespace)).to_native()
        else:
            integer_grid = original_column.dtype.is_integer() and all(
                _missing(value) or isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                for value in splits)
            if integer_grid:
                points = [None if _missing(value) else int(value) for value in splits]
                replacements = points * data.shape[0]
            else:
                # Normalize each grid point once, rather than rechecking the
                # same points for every observation in the expanded batch.
                if any(_missing(value) for value in splits):
                    points = [None if _missing(value) else float(value) for value in splits]
                    replacements = points * data.shape[0]
                else:
                    replacements = np.tile(np.asarray(splits, dtype=float), data.shape[0])
            working = set_columns(expanded, {variable: replacements})
    else:
        replacements = np.tile(splits, data.shape[0])
        working = set_columns(expanded, {variable: replacements})
    prediction = predict(model, working)
    metadata = [
        original,
        nw.new_series('_yhat_', prediction, backend=namespace),
        nw.lit(variable).alias('_vname_'),
        nw.new_series('_ids_', ids, backend=namespace),
    ]
    if label is not None:
        metadata.append(nw.lit(label).alias('_label_'))
    result = as_eager_dataframe(working).with_columns(metadata)
    return result.to_native(), index


def profile_tables(predict, model, data, variable_splits, processes=1, *, label=None, verbose=True):
    arguments = [(predict, model, data, variable, splits, label) for variable, splits in variable_splits.items()]
    if processes == 1:
        results = [single_profile(*args) for args in tqdm(arguments, desc='Calculating ceteris paribus', disable=not verbose)]
    else:
        with mp.get_context('spawn').Pool(processes) as pool:
            results = pool.starmap(single_profile, arguments)
    return ProfileTables([frame for frame, _ in results], [index for _, index in results], list(variable_splits))


def calculate(explainer, new_observation, variable_splits, y, processes, verbose):
    profiles = profile_tables(explainer.predict_function, explainer.model, new_observation,
                              variable_splits, processes, label=explainer.label, verbose=verbose)
    frame = as_eager_dataframe(new_observation)
    namespace = nw.get_native_namespace(frame)
    index = new_observation.index.copy() if is_pandas_dataframe(new_observation) else np.arange(len(frame))
    ids = index.values if is_pandas_dataframe(new_observation) else index
    prediction = explainer.predict(new_observation)
    columns = [nw.new_series('_yhat_', prediction, backend=namespace),
               nw.lit(explainer.label).alias('_label_'), nw.new_series('_ids_', ids, backend=namespace)]
    if y is not None:
        columns.append(nw.lit(np.asarray(y).item()).alias('_y_') if np.ndim(y) == 0
                       else nw.new_series('_y_', y, backend=namespace))
    observations = ObservationTable(frame.with_columns(columns).to_native(), index)
    return profiles, observations
